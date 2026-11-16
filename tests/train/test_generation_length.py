"""Tests that generation stays within each model's positional capacity.

An undertrained model never emits EOS, so evaluation generation used to run to
its default max_length of 512 and overflow the positional embedding with
"Sequence length N exceeds maximum length M". Each test here blocks EOS so
generation runs to the cap.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import BartConfig, BartForConditionalGeneration

from calt.io.base import StandardDataCollator
from calt.io.tokenizer import get_tokenizer
from calt.io.vocabulary.config import VocabConfig
from calt.models.decoder_only.model import DecoderOnlyConfig, DecoderOnlyTransformer
from calt.models.generic.model import Transformer, TransformerConfig
from calt.models.monomial import MonomialTransformer, create_monomial_config
from calt.models.monomial_decoder_only import (
    MonomialDecoderOnlyTransformer,
    create_monomial_decoder_only_config,
)
from calt.models.positional_embeddings import (
    RotaryPositionalEmbedding,
    register_positional_embedding,
)
from calt.trainer.trainer import Trainer

PAD, EOS, BOS = 0, 1, 2
NO_EOS = -1  # never produced, so generation runs to max_length
MAX_INPUT_LEN = 8
CAPACITY = 2 * MAX_INPUT_LEN


def _generic(use_positional_embedding="generic"):
    config = TransformerConfig(
        d_model=8,
        attention_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        vocab_size=20,
        max_input_len=MAX_INPUT_LEN,
        pad_token_id=PAD,
        eos_token_id=EOS,
        bos_token_id=BOS,
        use_positional_embedding=use_positional_embedding,
    )
    return Transformer(config).eval()


def _decoder_only():
    config = DecoderOnlyConfig(
        d_model=8,
        attention_heads=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        vocab_size=20,
        max_input_len=MAX_INPUT_LEN,
        pad_token_id=PAD,
        eos_token_id=EOS,
        bos_token_id=BOS,
    )
    return DecoderOnlyTransformer(config).eval()


def _inputs():
    input_ids = torch.tensor([[BOS, 5, 6, 7, EOS], [BOS, 8, 9, EOS, PAD]])
    return input_ids, (input_ids != PAD).long()


def test_generic_caps_max_length_at_capacity():
    input_ids, attention_mask = _inputs()
    generated = _generic().generate(
        input_ids, attention_mask, max_length=512, eos_token_id=NO_EOS
    )
    # BOS plus CAPACITY tokens; the decoder input never exceeds CAPACITY
    assert generated.shape[1] == 1 + CAPACITY


def test_generic_without_positional_embedding_is_not_capped():
    input_ids, attention_mask = _inputs()
    model = _generic(use_positional_embedding="none")
    assert model.generation_length_limit() is None
    generated = model.generate(
        input_ids, attention_mask, max_length=CAPACITY + 4, eos_token_id=NO_EOS
    )
    assert generated.shape[1] == 1 + CAPACITY + 4


def test_decoder_only_subtracts_the_problem_length():
    input_ids, attention_mask = _inputs()
    generated = _decoder_only().generate(
        input_ids, attention_mask, max_length=512, eos_token_id=NO_EOS
    )
    # The longest problem has 5 tokens, so 11 tokens fit after it
    assert generated.shape[1] == 1 + CAPACITY - 5


def test_decoder_only_rejects_a_problem_that_fills_the_capacity():
    input_ids = torch.full((1, CAPACITY), 5)
    with pytest.raises(ValueError, match="leaves no room"):
        _decoder_only().generate(input_ids, torch.ones_like(input_ids))


def test_none_positional_embedding_is_case_insensitive():
    assert _generic(use_positional_embedding="None").generation_length_limit() is None


class _PositionalWithoutMaxLen(nn.Module):
    """A custom positional embedding that sizes itself by max_len without storing it."""

    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, embeddings):
        positions = torch.arange(embeddings.size(1), device=embeddings.device)
        return embeddings + self.pos(positions).unsqueeze(0)


def test_custom_positional_embedding_is_capped_at_its_max_len():
    register_positional_embedding(
        "test_without_max_len",
        lambda d_model, max_len, **kw: _PositionalWithoutMaxLen(d_model, max_len),
    )
    model = _generic(use_positional_embedding="test_without_max_len")
    assert model.generation_length_limit() == CAPACITY
    input_ids, attention_mask = _inputs()
    generated = model.generate(
        input_ids, attention_mask, max_length=512, eos_token_id=NO_EOS
    )
    assert generated.shape[1] == 1 + CAPACITY


def test_rope_past_max_len_in_bfloat16_keeps_positions_apart():
    rope = RotaryPositionalEmbedding(d_model=8, max_len=4).to(torch.bfloat16)
    cos, _ = rope._cos_sin(512, torch.bfloat16)
    reference = RotaryPositionalEmbedding(d_model=8, max_len=512).cos_cached
    assert torch.allclose(cos.float(), reference, atol=1e-2)


def test_rope_past_max_len_keeps_its_buffers():
    rope = RotaryPositionalEmbedding(d_model=8, max_len=4)
    out = rope(torch.randn(2, 6, 8))
    assert out.shape == (2, 6, 8)
    # Growing the persistent buffers would break loading the checkpoint
    assert rope.cos_cached.shape == (1, 4, 8)
    assert rope.sin_cached.shape == (1, 4, 8)


# --------------------------------------------------------------------------- #
# Monomial models                                                               #
# --------------------------------------------------------------------------- #

N_VARS = 2
WIDTH = 1 + N_VARS + 1  # coefficient + exponents + follow slot
SAMPLES = [
    {"input": "C3 E1 E0 + C2 E0 E1 || C1 E0 E0", "target": "C5 E1 E1"},
    {"input": "C-4 E2 E0 || C2 E1 E1", "target": "C-2 E2 E1 + C1 E0 E0"},
]


class _NoEosHead(nn.Module):
    """Wrap a monomial head so the follow slot never predicts EOS."""

    def __init__(self, head):
        super().__init__()
        self.inner = head

    def forward(self, hidden):
        logits = self.inner(hidden)
        logits["follow"][..., -1] = float("-inf")  # EOS is the last class
        return logits


def _monomial(model_cls, create_config, model_type):
    vocab = (
        [f"C{i}" for i in range(-9, 10)] + [f"E{i}" for i in range(0, 4)] + ["+", "||"]
    )
    tokenizer = get_tokenizer(VocabConfig(vocab=vocab, special_tokens={}))
    model_cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "d_model": 32,
            "num_encoder_heads": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "encoder_ffn_dim": 64,
            "max_sequence_length": MAX_INPUT_LEN,
            "num_variables": N_VARS,
            "dropout": 0.0,
        }
    )
    model = model_cls(create_config(model_cfg, tokenizer)).eval()
    model.head = _NoEosHead(model.head)
    return model, StandardDataCollator(tokenizer)(SAMPLES)


def test_monomial_caps_monomials_at_capacity():
    model, batch = _monomial(MonomialTransformer, create_monomial_config, "monomial")
    assert model.generation_length_limit() == (CAPACITY + 1) * WIDTH
    generated = model.generate(
        batch["input_ids"], batch["attention_mask"], max_length=512
    )
    # BOS plus CAPACITY monomials (the capacity counts monomial rows)
    assert generated.shape[1] == 1 + CAPACITY * WIDTH


def test_monomial_decoder_only_fits_problem_and_answer():
    model, batch = _monomial(
        MonomialDecoderOnlyTransformer,
        create_monomial_decoder_only_config,
        "monomial_decoder_only",
    )
    generated = model.generate(
        batch["input_ids"], batch["attention_mask"], max_length=512
    )
    # The longest problem folds into 3 monomial rows, leaving 13 for the answer
    assert generated.shape[1] == 1 + (CAPACITY - 3) * WIDTH


# --------------------------------------------------------------------------- #
# BART                                                                          #
# --------------------------------------------------------------------------- #


def _bart(max_position_embeddings=8):
    config = BartConfig(
        encoder_layers=1,
        decoder_layers=1,
        d_model=16,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        vocab_size=25,
        pad_token_id=21,
        bos_token_id=22,
        eos_token_id=23,
        decoder_start_token_id=22,
        max_position_embeddings=max_position_embeddings,
    )
    return BartForConditionalGeneration(config).eval()


def test_bart_generates_up_to_max_position_embeddings_plus_one():
    model = _bart(max_position_embeddings=8)
    input_ids = torch.tensor([[22, 1, 2, 23]])
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_length=9,
        min_length=9,
    )
    assert generated.shape[1] == 9


def test_bart_generation_ends_with_the_real_eos_token():
    # Built like CALT does, BartConfig keeps its default forced_eos_token_id
    # of 2 (transformers 4.x and >= 5.4), an ordinary token in CALT vocabularies
    model = _bart()
    generate_kwargs = Trainer._generate_kwargs(SimpleNamespace(model=model))
    assert generate_kwargs == {"forced_eos_token_id": 23}

    input_ids = torch.tensor([[22, 1, 2, 23]])
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_length=6,
        min_length=6,
        **generate_kwargs,
    )
    assert generated[0, -1].item() == 23


def test_other_models_get_no_extra_generate_kwargs():
    assert Trainer._generate_kwargs(SimpleNamespace(model=_generic())) == {}


# --------------------------------------------------------------------------- #
# Trainer.evaluate_and_save_generation's max_length                            #
# --------------------------------------------------------------------------- #


def _resolve(model, max_length):
    return Trainer._generation_max_length(SimpleNamespace(model=model), max_length)


def test_default_max_length_is_capped_at_capacity():
    assert _resolve(_generic(), None) == CAPACITY


def test_default_max_length_is_512_without_a_limit():
    assert _resolve(_generic(use_positional_embedding="none"), None) == 512


def test_default_max_length_for_bart():
    assert _resolve(_bart(max_position_embeddings=8), None) == 9
    assert _resolve(_bart(max_position_embeddings=1024), None) == 512


def test_smaller_max_length_is_kept():
    assert _resolve(_generic(), 4) == 4


def test_too_large_max_length_is_capped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert _resolve(_generic(), 512) == CAPACITY
    assert "max_length=512" in caplog.text


@pytest.mark.parametrize("pe", ["generic", "sinusoidal", "rope"])
def test_generic_does_not_overflow_any_bounded_embedding(pe):
    input_ids, attention_mask = _inputs()
    generated = _generic(use_positional_embedding=pe).generate(
        input_ids, attention_mask, max_length=512, eos_token_id=NO_EOS
    )
    assert generated.shape[1] == 1 + CAPACITY
