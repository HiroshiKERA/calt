"""Tests for the monomial-embedding model (model_type=monomial).

Covers:
  - create_monomial_config derives coefficient/exponent/separator token groups
    from a real calt tokenizer (and fails clearly without num_variables).
  - ModelRegistry resolves the new model_type (and its monomial_transformer
    alias) from the unified OmegaConf config.
  - forward() consumes the standard seq2seq collator batch, returns a scalar
    loss and flat predicted token ids in the labels' shape.
  - Misaligned data (wrong num_variables) raises instead of training silently.
  - generate() returns [BOS, monomials..., EOS] flat ids that batch_decode and
    the exact-match evaluation consume unchanged.
  - A tiny model memorizes a two-sample dataset (loss drops, generation
    reproduces the targets).
  - save_pretrained / from_pretrained round-trips the vocabulary structure.
"""

import torch
from omegaconf import OmegaConf

from calt.io.base import StandardDataCollator
from calt.io.tokenizer import get_tokenizer
from calt.io.vocabulary.config import VocabConfig
from calt.models.base import ModelRegistry
from calt.models.monomial import (
    MonomialTransformer,
    create_monomial_config,
)

N_VARS = 2
WIDTH = 1 + N_VARS + 1  # 1 coeff field + n_vars exponents + follow slot

# Two-variable toy task in C/E expanded form.
SAMPLES = [
    {"input": "C3 E1 E0 + C2 E0 E1 || C1 E0 E0", "target": "C5 E1 E1"},
    {"input": "C-4 E2 E0 || C2 E1 E1", "target": "C-2 E2 E1 + C1 E0 E0"},
]


def _tokenizer():
    vocab = (
        [f"C{i}" for i in range(-9, 10)] + [f"E{i}" for i in range(0, 4)] + ["+", "||"]
    )
    return get_tokenizer(VocabConfig(vocab=vocab, special_tokens={}))


def _model_cfg(**overrides):
    cfg = {
        "model_type": "monomial",
        "d_model": 32,
        "num_encoder_heads": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "encoder_ffn_dim": 64,
        "max_sequence_length": 64,
        "num_variables": N_VARS,
        "dropout": 0.0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _batch(tokenizer, samples=SAMPLES):
    return StandardDataCollator(tokenizer)(samples)


def _build_model(**overrides):
    tokenizer = _tokenizer()
    config = create_monomial_config(_model_cfg(**overrides), tokenizer)
    return MonomialTransformer(config), tokenizer


# --------------------------------------------------------------------------- #
# Config mapping                                                                #
# --------------------------------------------------------------------------- #


def test_config_mapping_derives_structure():
    tokenizer = _tokenizer()
    config = create_monomial_config(_model_cfg(), tokenizer)
    vocab = tokenizer.get_vocab()

    assert config.monomial_width == WIDTH
    # One unlabeled coefficient field, ordered by coefficient value.
    assert config.num_coeff_fields == 1
    assert config.coeff_token_ids[0] == [vocab[f"C{i}"] for i in range(-9, 10)]
    # Shared exponent tokens replicated over the two variable slots.
    assert config.num_variables == N_VARS
    exp_ids = [vocab[f"E{i}"] for i in range(0, 4)]
    assert config.exp_token_ids == [exp_ids, exp_ids]
    # Separators in follow-slot order; EOS is the implicit last follow class.
    assert config.separator_token_ids == [vocab["+"], vocab["||"]]
    assert config.num_follow_classes == 3
    assert config.pad_token_id == tokenizer.pad_token_id
    assert config.eos_token_id == tokenizer.eos_token_id


def test_config_mapping_requires_num_variables():
    cfg = _model_cfg()
    del cfg.num_variables
    try:
        create_monomial_config(cfg, _tokenizer())
        assert False, "expected ValueError for missing num_variables"
    except ValueError as e:
        assert "num_variables" in str(e)


def test_config_mapping_requires_tokenizer():
    try:
        create_monomial_config(_model_cfg(), None)
        assert False, "expected ValueError for missing tokenizer"
    except ValueError as e:
        assert "tokenizer" in str(e)


# --------------------------------------------------------------------------- #
# Registry                                                                      #
# --------------------------------------------------------------------------- #


def test_registry_resolves_monomial():
    tokenizer = _tokenizer()
    registry = ModelRegistry()
    for name in ("monomial", "monomial_transformer"):
        model = registry.create_from_config(
            _model_cfg(model_type=name), tokenizer, model_name=name
        )
        assert isinstance(model, MonomialTransformer)


# --------------------------------------------------------------------------- #
# Forward                                                                       #
# --------------------------------------------------------------------------- #


def test_forward_loss_and_flat_predictions():
    model, tokenizer = _build_model()
    batch = _batch(tokenizer)
    out = model(**batch)

    assert out.loss is not None
    assert out.loss.dim() == 0
    assert out.loss.requires_grad
    assert torch.isfinite(out.loss)
    # Flat predicted token ids in the labels' shape, like the generic model.
    assert out.logits.shape == batch["labels"].shape
    assert out.logits.dtype == torch.long


def test_misaligned_data_raises():
    # A model expecting 3 variables folds 2-variable data off-grid: separators
    # land outside the follow slot and the alignment check must fire.
    model, tokenizer = _build_model(num_variables=3)
    batch = _batch(tokenizer)
    try:
        model(**batch)
        assert False, "expected ValueError for misaligned data"
    except ValueError as e:
        assert "expanded form" in str(e)


# --------------------------------------------------------------------------- #
# Generation                                                                    #
# --------------------------------------------------------------------------- #


def test_generate_is_flat_and_monomial_aligned():
    model, tokenizer = _build_model()
    batch = _batch(tokenizer)
    generated = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_length=64,
    )

    assert generated.dtype == torch.long
    assert (generated[:, 0] == model.config.bos_token_id).all()
    # After BOS, output is whole monomials of WIDTH tokens each.
    assert (generated.size(1) - 1) % WIDTH == 0

    follow_ids = set(model.config.separator_token_ids) | {model.config.eos_token_id}
    for row in generated:
        grid = row[1:].reshape(-1, WIDTH)
        seen_eos = False
        for monomial in grid:
            if seen_eos:
                assert (monomial == model.config.pad_token_id).all()
            else:
                assert int(monomial[-1]) in follow_ids
                seen_eos = int(monomial[-1]) == model.config.eos_token_id

    # Decoding must not crash and yields plain expanded-form tokens. (Full
    # re-encodability is asserted after training, where generation terminates
    # with EOS instead of being cut at max_length.)
    tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


# --------------------------------------------------------------------------- #
# Learning smoke test                                                           #
# --------------------------------------------------------------------------- #


def test_tiny_model_memorizes_two_samples():
    torch.manual_seed(0)
    model, tokenizer = _build_model()
    batch = _batch(tokenizer)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    first_loss = None
    for _ in range(150):
        optimizer.zero_grad()
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = out.loss.item()

    assert out.loss.item() < 0.1 * first_loss

    model.eval()
    generated = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_length=64,
    )
    texts = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    assert [t.strip() for t in texts] == [s["target"] for s in SAMPLES]

    # EOS-terminated generations decode to well-formed expanded form:
    # re-encoding them must pass the model's own alignment check.
    reencoded = _batch(tokenizer, [{"input": t, "target": t} for t in texts])
    model(**reencoded)  # would raise if misaligned


# --------------------------------------------------------------------------- #
# Serialization                                                                 #
# --------------------------------------------------------------------------- #


def test_save_load_roundtrip(tmp_path):
    model, tokenizer = _build_model()
    batch = _batch(tokenizer)
    model.eval()
    before = model.generate(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )

    model.save_pretrained(tmp_path)
    reloaded = MonomialTransformer.from_pretrained(tmp_path)
    reloaded.eval()

    assert reloaded.config.coeff_token_ids == model.config.coeff_token_ids
    assert reloaded.config.exp_token_ids == model.config.exp_token_ids
    assert reloaded.config.separator_token_ids == model.config.separator_token_ids

    after = reloaded.generate(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    assert torch.equal(before, after)
