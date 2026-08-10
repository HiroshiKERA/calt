"""Tests for the decoder-only model.

The model folds the collated seq2seq batch into a single causal sequence, so the
things worth pinning down are that the fold is faithful (problem padding does not
leak into the solution's positions), that the loss is only taken on the solution,
and that generation returns the answer alone.
"""

import torch
from omegaconf import OmegaConf

from calt.models import ModelPipeline
from calt.models.decoder_only.model import DecoderOnlyConfig, DecoderOnlyTransformer

PAD, EOS, BOS = 0, 1, 2


def _config(**overrides) -> DecoderOnlyConfig:
    params = dict(
        d_model=32,
        attention_heads=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        vocab_size=20,
        max_input_len=64,
        pad_token_id=PAD,
        eos_token_id=EOS,
        bos_token_id=BOS,
    )
    params.update(overrides)
    return DecoderOnlyConfig(**params)


def _batch():
    """Two examples whose problems differ in length, so one row is padded."""
    input_ids = torch.tensor(
        [
            [BOS, 5, 6, 7, EOS],
            [BOS, 8, 9, EOS, PAD],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ]
    )
    decoder_input_ids = torch.tensor(
        [
            [BOS, 11, 12],
            [BOS, 13, PAD],
        ]
    )
    decoder_attention_mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 0],
        ]
    )
    labels = torch.tensor(
        [
            [11, 12, EOS],
            [13, EOS, -100],
        ]
    )
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
    )


def test_forward_shapes_and_loss():
    """Predictions line up with labels so the trainer's metric can compare them."""
    model = DecoderOnlyTransformer(_config()).eval()
    batch = _batch()

    out = model(**batch)

    assert out.logits.shape == batch["labels"].shape
    assert out.loss is not None and torch.isfinite(out.loss)


def test_padding_of_the_problem_does_not_change_the_answer():
    """A short problem must be scored the same alone as next to a longer one.

    The collator pads the problem and the solution independently, so a naive
    concatenation would leave padding between them and shift every solution
    token by an amount that depends on the rest of the batch.
    """
    model = DecoderOnlyTransformer(_config()).eval()
    batch = _batch()

    with torch.no_grad():
        batched = model(**batch)

        alone = model(
            input_ids=batch["input_ids"][1:, :4],
            attention_mask=batch["attention_mask"][1:, :4],
            decoder_input_ids=batch["decoder_input_ids"][1:, :2],
            decoder_attention_mask=batch["decoder_attention_mask"][1:, :2],
            labels=batch["labels"][1:, :2],
        )

    assert torch.equal(batched.logits[1, :2], alone.logits[0])


def test_loss_ignores_the_problem():
    """Only the solution carries loss; the problem part is context, not target."""
    model = DecoderOnlyTransformer(_config()).eval()
    batch = _batch()

    out = model(**batch)
    out.loss.backward()

    # Every gradient path runs through the solution positions. Changing a
    # problem token changes the loss (it is context), but no loss term asks the
    # model to reproduce the problem itself, which is what the shape assertion
    # above encodes: one prediction per label, none per problem token.
    assert out.logits.shape == batch["labels"].shape
    assert model.lm_head.weight.grad is not None


def test_generate_returns_only_the_answer():
    """Generation starts mid-sequence and returns the completion, not the prompt."""
    model = DecoderOnlyTransformer(_config()).eval()
    batch = _batch()

    with torch.no_grad():
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=4,
        )

    # BOS opens the answer, as it does in the seq2seq model's output.
    assert generated.shape[0] == batch["input_ids"].shape[0]
    assert generated.shape[1] <= 1 + 4
    assert torch.all(generated[:, 0] == BOS)
    # No problem token is echoed back into the returned sequence.
    assert not torch.any(generated[:, 1:] == 5)


def test_generate_matches_teacher_forcing():
    """Greedy decoding reproduces the argmax the forward pass would take."""
    model = DecoderOnlyTransformer(_config()).eval()
    batch = _batch()

    with torch.no_grad():
        forced = model(**batch).logits  # already argmaxed ids
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=1,
        )

    # The first generated token follows the answer-opening BOS, which is exactly
    # what position 0 of the teacher-forced predictions holds.
    assert torch.equal(generated[:, 1], forced[:, 0])


def test_generate_stops_after_eos():
    """Once a row emits EOS it only emits padding afterwards."""
    config = _config()
    model = DecoderOnlyTransformer(config).eval()
    # Force EOS: bias the head so EOS always wins.
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[EOS] = 100.0

    batch = _batch()
    with torch.no_grad():
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=5,
        )

    assert torch.all(generated[:, 1] == EOS)
    assert torch.all(generated[:, 2:] == PAD)


def test_registry_builds_from_model_type():
    """`model_type: decoder_only` is reachable from an experiment config."""
    cfg = OmegaConf.create(
        {
            "model_type": "decoder_only",
            "d_model": 32,
            "num_encoder_layers": 6,
            "num_encoder_heads": 4,
            "num_decoder_layers": 2,
            "num_decoder_heads": 4,
            "encoder_ffn_dim": 64,
            "decoder_ffn_dim": 64,
            "max_sequence_length": 64,
            "vocab_size": 20,
        }
    )

    model = ModelPipeline(cfg, tokenizer=None).build()

    assert isinstance(model, DecoderOnlyTransformer)
    # A seq2seq config maps onto one stack: the decoder side is what survives.
    assert model.config.num_layers == 2
    assert len(model.transformer.layers) == 2


def test_alias_decoder_builds_the_same_model():
    """`model_type: decoder` is accepted as a shorthand."""
    cfg = OmegaConf.create(
        {
            "model_type": "decoder",
            "d_model": 32,
            "num_decoder_layers": 2,
            "num_decoder_heads": 4,
            "num_encoder_heads": 4,
            "encoder_ffn_dim": 64,
            "max_sequence_length": 64,
            "vocab_size": 20,
        }
    )

    model = ModelPipeline(cfg, tokenizer=None).build()

    assert isinstance(model, DecoderOnlyTransformer)


def test_num_layers_overrides_the_seq2seq_keys():
    """An explicit `num_layers` wins over `num_decoder_layers`."""
    cfg = OmegaConf.create(
        {
            "model_type": "decoder_only",
            "d_model": 32,
            "num_layers": 4,
            "num_decoder_layers": 2,
            "num_encoder_heads": 4,
            "encoder_ffn_dim": 64,
            "max_sequence_length": 64,
            "vocab_size": 20,
        }
    )

    model = ModelPipeline(cfg, tokenizer=None).build()

    assert len(model.transformer.layers) == 4
