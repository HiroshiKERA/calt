"""``model.input_embedding_kwargs`` reaches the input-embedding factory.

The three model families that use ``register_input_embedding`` pass the
mapping to the factory, store it in the model config, and rebuild the same
embedding from a saved checkpoint.
"""

import json

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from calt.models import ModelPipeline, register_input_embedding
from calt.models.bart.config_mapping import create_bart_config
from calt.models.decoder_only.model import DecoderOnlyTransformer
from calt.models.encoder_classifier import EncoderClassifier
from calt.models.generic.model import Transformer
from calt.models.monomial.config_mapping import create_monomial_config
from calt.models.monomial_decoder_only.config_mapping import (
    create_monomial_decoder_only_config,
)

MODEL_CLASSES = {
    "generic": Transformer,
    "decoder_only": DecoderOnlyTransformer,
    "encoder_only": EncoderClassifier,
}
KINDS = sorted(MODEL_CLASSES)


class ScaledEmbedding(nn.Module):
    """A token embedding whose output is multiplied by a configurable scale."""

    def __init__(self, vocab_size, d_model, scale=1.0, digit_ids=(), **kwargs):
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model)
        self.scale = scale
        self.digit_ids = list(digit_ids)

    def forward(self, input_ids):
        return self.table(input_ids) * self.scale


@pytest.fixture(autouse=True)
def _registered():
    register_input_embedding("scaled_kw", ScaledEmbedding)


def _model_cfg(kind, **overrides):
    cfg = OmegaConf.create(
        {
            "model_type": kind,
            "vocab_size": 20,
            "d_model": 16,
            "num_encoder_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "encoder_ffn_dim": 32,
            "max_sequence_length": 32,
            "dropout": 0.0,
        }
    )
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _scaled_cfg(kind):
    return _model_cfg(
        kind,
        input_embedding_type="scaled_kw",
        input_embedding_kwargs={"scale": 3.0, "digit_ids": [4, 5, 6]},
    )


@pytest.mark.parametrize("kind", KINDS)
def test_kwargs_reach_the_factory(kind):
    model = ModelPipeline(_scaled_cfg(kind), tokenizer=None).build()
    assert isinstance(model.embedding, ScaledEmbedding)
    assert model.embedding.scale == 3.0
    assert model.embedding.digit_ids == [4, 5, 6]


@pytest.mark.parametrize("kind", KINDS)
def test_kwargs_are_stored_as_a_plain_dict(kind):
    model = ModelPipeline(_scaled_cfg(kind), tokenizer=None).build()
    # A DictConfig here would make save_pretrained fail to write config.json
    assert type(model.config.input_embedding_kwargs) is dict
    assert model.config.input_embedding_kwargs == {
        "scale": 3.0,
        "digit_ids": [4, 5, 6],
    }


@pytest.mark.parametrize("kind", KINDS)
def test_saved_model_rebuilds_the_same_embedding(kind, tmp_path):
    model = ModelPipeline(_scaled_cfg(kind), tokenizer=None).build()
    model.save_pretrained(tmp_path)
    reloaded = MODEL_CLASSES[kind].from_pretrained(tmp_path)
    assert reloaded.embedding.scale == 3.0
    assert reloaded.embedding.digit_ids == [4, 5, 6]
    for key, value in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], value), key


@pytest.mark.parametrize("kind", KINDS)
def test_checkpoint_without_the_key_still_loads(kind, tmp_path):
    model = ModelPipeline(_model_cfg(kind), tokenizer=None).build()
    model.save_pretrained(tmp_path)
    # Older versions never wrote this key
    config_path = tmp_path / "config.json"
    saved = json.loads(config_path.read_text())
    saved.pop("input_embedding_kwargs", None)
    config_path.write_text(json.dumps(saved))

    reloaded = MODEL_CLASSES[kind].from_pretrained(tmp_path)
    assert reloaded.config.input_embedding_kwargs == {}


@pytest.mark.parametrize("kind", KINDS)
def test_default_model_is_unchanged(kind):
    default = ModelPipeline(_model_cfg(kind), tokenizer=None).build()
    explicit = ModelPipeline(
        _model_cfg(kind, input_embedding_kwargs={}), tokenizer=None
    ).build()
    assert default.config.input_embedding_kwargs == {}
    assert isinstance(default.embedding, nn.Embedding)
    # Both builds use the config seed, so the weights must match exactly
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        assert torch.equal(explicit.state_dict()[key], value), key


@pytest.mark.parametrize("key", ["vocab_size", "d_model"])
def test_kwargs_cannot_override_model_sizes(key):
    cfg = _model_cfg("generic", input_embedding_kwargs={key: 8})
    with pytest.raises(ValueError, match="must not set"):
        ModelPipeline(cfg, tokenizer=None).build()


def test_kwargs_must_be_a_mapping():
    cfg = _model_cfg("generic", input_embedding_kwargs=[1, 2])
    with pytest.raises(TypeError, match="must be a mapping"):
        ModelPipeline(cfg, tokenizer=None).build()


def test_kwargs_keys_must_be_strings():
    cfg = _model_cfg("generic")
    cfg.input_embedding_kwargs = OmegaConf.create({3: 1.0})
    with pytest.raises(TypeError, match="keys must be strings"):
        ModelPipeline(cfg, tokenizer=None).build()


def test_kwargs_must_be_json_serializable():
    cfg = _model_cfg("generic", input_embedding_kwargs={"scale": b"x"})
    with pytest.raises(TypeError, match="JSON-serializable"):
        ModelPipeline(cfg, tokenizer=None).build()


def test_factory_error_names_the_kwargs():
    register_input_embedding(
        "no_kwargs_kw", lambda vocab_size, d_model: nn.Embedding(vocab_size, d_model)
    )
    cfg = _model_cfg(
        "generic",
        input_embedding_type="no_kwargs_kw",
        input_embedding_kwargs={"unknown": 1},
    )
    with pytest.raises(TypeError, match="model.input_embedding_kwargs"):
        ModelPipeline(cfg, tokenizer=None).build()


@pytest.mark.parametrize(
    "create_config,model_type",
    [
        (create_bart_config, "bart"),
        (create_monomial_config, "monomial"),
        (create_monomial_decoder_only_config, "monomial_decoder_only"),
    ],
)
def test_unsupported_models_reject_the_key(create_config, model_type):
    cfg = _model_cfg(model_type, input_embedding_kwargs={"scale": 2.0})
    with pytest.raises(ValueError, match="does not support input_embedding_kwargs"):
        create_config(cfg, None)
