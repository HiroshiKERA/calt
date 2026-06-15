"""Tests for pluggable input + positional embeddings.

Covers:
  - get_input_embedding: default "token" is an nn.Embedding; unknown raises.
  - register_input_embedding: a custom factory is selected by config and ends up
    as the model's input embedding (generic and encoder_classifier).
  - register_positional_embedding: a custom positional type resolves; built-ins
    still work; unknown raises.
  - A forward pass works with both custom embeddings selected.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from calt.models import (
    get_input_embedding,
    get_positional_embedding,
    register_input_embedding,
    register_positional_embedding,
)
from calt.models.encoder_classifier import (
    EncoderClassifier,
    EncoderClassifierConfig,
    create_encoder_classifier_config,
)
from calt.models.generic.config_mapping import create_transformer_config
from calt.models.generic.model import Transformer, TransformerConfig

# --------------------------------------------------------------------------- #
# Custom modules used by the tests                                              #
# --------------------------------------------------------------------------- #


class ScaledEmbedding(nn.Module):
    """A trivial custom input embedding (token table scaled by sqrt(d_model))."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.scale = d_model**0.5

    def forward(self, input_ids):
        return self.emb(input_ids) * self.scale


class AddOnePositional(nn.Module):
    """A trivial custom positional embedding (adds a learned per-position bias)."""

    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, embeddings):
        seq_len = embeddings.size(1)
        positions = torch.arange(seq_len, device=embeddings.device)
        return embeddings + self.pos(positions).unsqueeze(0)


@pytest.fixture(autouse=True)
def _register_customs():
    register_input_embedding(
        "scaled", lambda vocab_size, d_model, **kw: ScaledEmbedding(vocab_size, d_model)
    )
    register_positional_embedding(
        "addone", lambda d_model, max_len, **kw: AddOnePositional(d_model, max_len)
    )


# --------------------------------------------------------------------------- #
# Input embeddings                                                              #
# --------------------------------------------------------------------------- #


def test_default_input_embedding_is_plain_table():
    emb = get_input_embedding("token", vocab_size=10, d_model=8)
    assert isinstance(emb, nn.Embedding)
    assert emb.weight.shape == (10, 8)


def test_unknown_input_embedding_raises():
    with pytest.raises(ValueError, match="Unsupported input embedding type"):
        get_input_embedding("does_not_exist", vocab_size=10, d_model=8)


def test_custom_input_embedding_selected_in_generic_model():
    cfg = TransformerConfig(
        d_model=8,
        attention_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        vocab_size=10,
        input_embedding_type="scaled",
    )
    model = Transformer(cfg)
    assert isinstance(model.embedding, ScaledEmbedding)


def test_default_input_embedding_in_generic_model():
    cfg = TransformerConfig(d_model=8, attention_heads=2, vocab_size=10)
    model = Transformer(cfg)
    assert isinstance(model.embedding, nn.Embedding)


# --------------------------------------------------------------------------- #
# Positional embeddings                                                         #
# --------------------------------------------------------------------------- #


def test_builtin_positional_still_resolve():
    assert get_positional_embedding("none", d_model=8, max_len=16) is None
    assert get_positional_embedding("generic", d_model=8, max_len=16) is not None
    assert get_positional_embedding("sinusoidal", d_model=8, max_len=16) is not None
    assert get_positional_embedding("rope", d_model=8, max_len=16) is not None


def test_custom_positional_resolves():
    pe = get_positional_embedding("addone", d_model=8, max_len=16)
    assert isinstance(pe, AddOnePositional)


def test_unknown_positional_raises():
    with pytest.raises(ValueError, match="Unsupported positional embedding type"):
        get_positional_embedding("does_not_exist", d_model=8, max_len=16)


# --------------------------------------------------------------------------- #
# config mapping passes the new field through                                   #
# --------------------------------------------------------------------------- #


def test_config_mapping_passes_input_embedding_type():
    model_cfg = OmegaConf.create(
        {
            "d_model": 8,
            "num_encoder_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "encoder_ffn_dim": 16,
            "max_sequence_length": 32,
            "input_embedding_type": "scaled",
            "use_positional_embedding": "addone",
        }
    )
    t_cfg = create_transformer_config(model_cfg, tokenizer=None)
    assert t_cfg.input_embedding_type == "scaled"
    assert t_cfg.use_positional_embedding == "addone"
    # default when omitted
    bare = OmegaConf.create(
        {
            "d_model": 8,
            "num_encoder_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "encoder_ffn_dim": 16,
            "max_sequence_length": 32,
        }
    )
    assert create_transformer_config(bare, None).input_embedding_type == "token"


# --------------------------------------------------------------------------- #
# End-to-end: both custom embeddings in one model, forward works                #
# --------------------------------------------------------------------------- #


def test_forward_with_custom_input_and_positional():
    cfg = EncoderClassifierConfig(
        d_model=8,
        attention_heads=2,
        num_encoder_layers=1,
        dim_feedforward=16,
        vocab_size=10,
        input_embedding_type="scaled",
        use_positional_embedding="addone",
    )
    model = EncoderClassifier(cfg)
    assert isinstance(model.embedding, ScaledEmbedding)
    assert isinstance(model.positional_embedding, AddOnePositional)
    input_ids = torch.tensor([[2, 5, 6, 1], [2, 7, 1, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = torch.tensor([[3, 1, -100], [4, 1, -100]])
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None
    assert out.loss.requires_grad


def test_encoder_classifier_config_mapping_input_embedding():
    model_cfg = OmegaConf.create(
        {
            "d_model": 8,
            "num_encoder_heads": 2,
            "num_encoder_layers": 1,
            "encoder_ffn_dim": 16,
            "max_sequence_length": 32,
            "input_embedding_type": "scaled",
        }
    )
    cfg = create_encoder_classifier_config(model_cfg, tokenizer=None)
    assert cfg.input_embedding_type == "scaled"
