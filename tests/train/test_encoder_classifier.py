"""Tests for the encoder-only classification model (model_type=encoder_classifier).

Covers:
  - ModelRegistry resolves the new model_type (and its encoder_only alias).
  - forward() consumes the standard seq2seq batch and returns a scalar loss plus
    single-token predictions, deriving the target class from labels.
  - generate() wraps the predicted token as [BOS, token, EOS] for the existing
    exact-match generation evaluation.
  - create_encoder_classifier_config maps the unified cfg.model (+ tokenizer).
  - Trainer._compute_metrics scores the classification output as accuracy.
"""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from calt.models.base import ModelRegistry
from calt.models.encoder_classifier import (
    EncoderClassifier,
    EncoderClassifierConfig,
    create_encoder_classifier_config,
)
from calt.trainer.trainer import Trainer

VOCAB = 20
PAD, EOS, BOS = 0, 1, 2


def _config():
    return EncoderClassifierConfig(
        d_model=16,
        attention_heads=4,
        num_encoder_layers=2,
        dim_feedforward=32,
        vocab_size=VOCAB,
        max_input_len=32,
        pad_token_id=PAD,
        eos_token_id=EOS,
        bos_token_id=BOS,
    )


def _toy_batch():
    # Two samples; answer tokens are 7 and 9 (first non-(-100) label per row).
    input_ids = torch.tensor([[BOS, 5, 6, EOS], [BOS, 8, EOS, PAD]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = torch.tensor([[7, EOS, -100], [9, EOS, -100]])
    return input_ids, attention_mask, labels


def test_registry_resolves_encoder_classifier():
    reg = ModelRegistry()
    for name in ("encoder_classifier", "encoder_only"):
        model = reg.create(name, _config())
        assert isinstance(model, EncoderClassifier)


def test_forward_loss_and_predictions():
    model = EncoderClassifier(_config())
    input_ids, attention_mask, labels = _toy_batch()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None
    assert out.loss.requires_grad
    assert out.loss.dim() == 0  # scalar
    # Predicted token id per sample, shape (batch, 1).
    assert out.logits.shape == (2, 1)
    assert out.logits.dtype == torch.long


def test_forward_ignores_decoder_inputs():
    """Passing seq2seq-only keys (decoder_input_ids) must not break the model."""
    model = EncoderClassifier(_config())
    input_ids, attention_mask, labels = _toy_batch()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_input_ids=torch.tensor([[BOS, 7], [BOS, 9]]),
        decoder_attention_mask=torch.tensor([[1, 1], [1, 1]]),
    )
    assert out.loss is not None


def test_generate_wraps_bos_eos():
    model = EncoderClassifier(_config())
    input_ids, attention_mask, _ = _toy_batch()
    gen = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    assert gen.shape == (2, 3)
    assert torch.all(gen[:, 0] == BOS)
    assert torch.all(gen[:, 2] == EOS)


def test_config_mapping_from_unified():
    model_cfg = OmegaConf.create(
        {
            "d_model": 16,
            "num_encoder_heads": 4,
            "num_encoder_layers": 2,
            "encoder_ffn_dim": 32,
            "max_sequence_length": 64,
        }
    )
    tokenizer = SimpleNamespace(
        vocab={f"t{i}": i for i in range(VOCAB)},
        pad_token_id=PAD,
        eos_token_id=EOS,
        bos_token_id=BOS,
    )
    cfg = create_encoder_classifier_config(model_cfg, tokenizer)
    assert isinstance(cfg, EncoderClassifierConfig)
    assert cfg.vocab_size == VOCAB
    assert (cfg.pad_token_id, cfg.eos_token_id, cfg.bos_token_id) == (PAD, EOS, BOS)
    assert cfg.is_classification is True


def test_compute_metrics_classification_branch():
    # Predicted token ids (batch, 1); labels are seq2seq-shaped with -100 padding.
    predictions = torch.tensor([[7], [3]])  # first correct (7), second wrong (3 vs 9)
    labels = torch.tensor([[7, EOS, -100], [9, EOS, -100]])
    fake_self = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(is_classification=True))
    )
    metrics = Trainer._compute_metrics(fake_self, (predictions, labels))
    assert metrics["token_accuracy"] == 0.5
    assert metrics["success_rate"] == 0.5
