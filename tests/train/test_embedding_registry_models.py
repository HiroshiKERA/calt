"""Registered embeddings behave the same way in the three model families.

``register_input_embedding`` and ``register_positional_embedding`` are meant to
be used identically with the encoder-decoder (``generic``), decoder-only
(``decoder_only``) and encoder-only (``encoder_only``) models: register a
factory, name it in the YAML, build with ``ModelPipeline``.  The decoder-only
model used to build its input embedding directly and ignore
``input_embedding_type`` without an error, so every check here runs on all
three families.

The modules below count their calls, which is how the tests tell a module that
was merely constructed from one the model actually runs.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from calt.models import (
    ModelPipeline,
    register_input_embedding,
    register_positional_embedding,
)
from calt.models.decoder_only.model import DecoderOnlyTransformer
from calt.models.encoder_classifier import EncoderClassifier
from calt.models.generic.model import Transformer

PAD, EOS, BOS = 0, 1, 2

MODEL_CLASSES = {
    "generic": Transformer,
    "decoder_only": DecoderOnlyTransformer,
    "encoder_only": EncoderClassifier,
}
KINDS = sorted(MODEL_CLASSES)


class ProbeInput(nn.Module):
    """Custom input embedding: a learnable table that counts its calls."""

    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model)
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        return self.table(input_ids)


class ProbePosition(nn.Module):
    """Custom positional embedding: a learnable per-position bias, counting calls."""

    def __init__(self, d_model, max_len, **kwargs):
        super().__init__()
        self.table = nn.Embedding(max_len, d_model)
        self.calls = 0

    def forward(self, embeddings):
        self.calls += 1
        positions = torch.arange(embeddings.size(1), device=embeddings.device)
        return embeddings + self.table(positions)


def _register_probes():
    register_input_embedding("probe_input", ProbeInput)
    register_positional_embedding("probe_position", ProbePosition)


@pytest.fixture(autouse=True)
def _probes_registered():
    _register_probes()


def _model_cfg(kind, **overrides):
    """The ``model:`` section of a train.yaml, parsed from YAML text."""
    cfg = OmegaConf.create(
        f"""
        model_type: {kind}
        vocab_size: 20
        d_model: 16
        num_encoder_heads: 2
        num_encoder_layers: 1
        num_decoder_layers: 1
        encoder_ffn_dim: 32
        max_sequence_length: 32
        dropout: 0.0
        """
    )
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _probe_cfg(kind):
    return _model_cfg(
        kind,
        input_embedding_type="probe_input",
        use_positional_embedding="probe_position",
    )


def _batch(kind, device="cpu"):
    """Two examples of different lengths, in the layout each family expects."""
    batch = dict(
        input_ids=torch.tensor([[BOS, 5, 6, 7, EOS], [BOS, 8, 9, EOS, PAD]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
    )
    if kind == "encoder_only":
        # One class token per example, then EOS; the rest is ignored.
        batch["labels"] = torch.tensor([[11, EOS, -100], [13, EOS, -100]])
    else:
        batch["decoder_input_ids"] = torch.tensor([[BOS, 11, 12], [BOS, 13, PAD]])
        batch["decoder_attention_mask"] = torch.tensor([[1, 1, 1], [1, 1, 0]])
        batch["labels"] = torch.tensor([[11, 12, EOS], [13, EOS, -100]])
    return {name: tensor.to(device) for name, tensor in batch.items()}


def _build(cfg):
    return ModelPipeline(cfg, tokenizer=None).build()


# --------------------------------------------------------------------------- #
# 1. YAML -> ModelPipeline builds the registered modules                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", KINDS)
def test_yaml_selects_the_registered_modules(kind):
    model = _build(_probe_cfg(kind))

    assert isinstance(model, MODEL_CLASSES[kind])
    assert isinstance(model.embedding, ProbeInput)
    assert isinstance(model.positional_embedding, ProbePosition)
    assert model.config.input_embedding_type == "probe_input"
    assert model.config.use_positional_embedding == "probe_position"


# --------------------------------------------------------------------------- #
# 2. forward runs both modules                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", KINDS)
def test_forward_runs_both_modules(kind):
    model = _build(_probe_cfg(kind)).eval()
    batch = _batch(kind)

    out = model(**batch)

    assert model.embedding.calls > 0
    assert model.positional_embedding.calls > 0
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.logits.shape[0] == batch["input_ids"].shape[0]
    if kind != "encoder_only":
        assert out.logits.shape == batch["labels"].shape


# --------------------------------------------------------------------------- #
# 3. gradients reach the custom parameters; the model moves between devices     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_reach_the_custom_parameters(kind):
    model = _build(_probe_cfg(kind))

    model(**_batch(kind)).loss.backward()

    for module in (model.embedding, model.positional_embedding):
        grad = module.table.weight.grad
        assert grad is not None
        assert torch.any(grad != 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("kind", KINDS)
def test_custom_modules_follow_the_model_to_another_device(kind):
    model = _build(_probe_cfg(kind)).to("cuda")

    assert model.embedding.table.weight.is_cuda
    assert model.positional_embedding.table.weight.is_cuda
    out = model(**_batch(kind, device="cuda"))
    assert torch.isfinite(out.loss)


# --------------------------------------------------------------------------- #
# 4. generation runs the registered modules too                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", KINDS)
def test_generation_runs_the_registered_modules(kind):
    model = _build(_probe_cfg(kind)).eval()
    batch = _batch(kind)

    with torch.no_grad():
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=4,
        )

    assert model.embedding.calls > 0
    assert model.positional_embedding.calls > 0
    assert generated.shape[0] == batch["input_ids"].shape[0]
    assert torch.all(generated[:, 0] == BOS)


# --------------------------------------------------------------------------- #
# 5. save, then rebuild and load in another process                             #
# --------------------------------------------------------------------------- #

# The registry lives in the process, not in the checkpoint: a fresh process has
# to register the same names before it can rebuild the model.
_CHILD = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("embedding_probes", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._reload_and_run(*sys.argv[2:])
"""


def _reload_and_run(kind, checkpoint, result_path, register):
    """Entry point of the child process (see ``_CHILD``)."""
    if register == "register":
        _register_probes()
    model = MODEL_CLASSES[kind].from_pretrained(checkpoint).eval()
    with torch.no_grad():
        out = model(**_batch(kind))
    torch.save({"loss": out.loss, "logits": out.logits}, result_path)


def _run_child(kind, checkpoint, result_path, register):
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(Path(__file__).resolve()),
            kind,
            str(checkpoint),
            str(result_path),
            register,
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_reload_in_another_process_gives_the_same_output(kind, tmp_path):
    model = _build(_probe_cfg(kind)).eval()
    with torch.no_grad():
        # Move every weight away from its initial value, so the child can only
        # match by loading the saved weights, not by repeating the same init.
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
        expected = model(**_batch(kind))
    model.save_pretrained(tmp_path / "checkpoint")

    saved = json.loads((tmp_path / "checkpoint" / "config.json").read_text())
    assert saved["input_embedding_type"] == "probe_input"
    assert saved["use_positional_embedding"] == "probe_position"

    done = _run_child(kind, tmp_path / "checkpoint", tmp_path / "out.pt", "register")
    assert done.returncode == 0, done.stderr
    reloaded = torch.load(tmp_path / "out.pt")

    assert torch.allclose(reloaded["loss"], expected.loss, atol=1e-6)
    assert torch.equal(reloaded["logits"], expected.logits)


def test_reload_without_registering_fails_with_the_registry_error(tmp_path):
    model = _build(_probe_cfg("decoder_only"))
    model.save_pretrained(tmp_path / "checkpoint")

    done = _run_child(
        "decoder_only", tmp_path / "checkpoint", tmp_path / "out.pt", "skip"
    )

    assert done.returncode != 0
    assert "Unsupported input embedding type" in done.stderr


# --------------------------------------------------------------------------- #
# 6. unknown names are errors, never a silent fallback                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_input_embedding_name_raises(kind):
    cfg = _model_cfg(kind, input_embedding_type="not_registered")
    with pytest.raises(ValueError, match="Unsupported input embedding type"):
        _build(cfg)


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_positional_embedding_name_raises(kind):
    cfg = _model_cfg(kind, use_positional_embedding="not_registered")
    with pytest.raises(ValueError, match="Unsupported positional embedding type"):
        _build(cfg)


# --------------------------------------------------------------------------- #
# 7. decoder-only: omitting the key is the same as `token`                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["token", "default", "learned"])
def test_decoder_only_default_is_the_token_embedding(name):
    torch.manual_seed(0)
    omitted = _build(_model_cfg("decoder_only")).eval()
    torch.manual_seed(0)
    named = _build(_model_cfg("decoder_only", input_embedding_type=name)).eval()

    assert omitted.config.input_embedding_type == "token"
    assert type(omitted.embedding) is nn.Embedding
    assert type(named.embedding) is nn.Embedding

    omitted_state, named_state = omitted.state_dict(), named.state_dict()
    assert omitted_state.keys() == named_state.keys()
    for key in omitted_state:
        assert torch.equal(omitted_state[key], named_state[key]), key

    batch = _batch("decoder_only")
    with torch.no_grad():
        first, second = omitted(**batch), named(**batch)
    assert torch.equal(first.loss, second.loss)
    assert torch.equal(first.logits, second.logits)


# --------------------------------------------------------------------------- #
# 8. decoder-only: checkpoints from before the key existed still load           #
# --------------------------------------------------------------------------- #


def test_decoder_only_loads_a_checkpoint_without_the_key(tmp_path):
    model = _build(_model_cfg("decoder_only")).eval()
    assert "embedding.weight" in model.state_dict()
    model.save_pretrained(tmp_path)

    # Older versions never wrote this key; remove it to get such a config.
    config_path = tmp_path / "config.json"
    saved = json.loads(config_path.read_text())
    del saved["input_embedding_type"]
    config_path.write_text(json.dumps(saved))

    reloaded = DecoderOnlyTransformer.from_pretrained(tmp_path).eval()

    assert reloaded.config.input_embedding_type == "token"
    for key, value in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], value), key
    batch = _batch("decoder_only")
    with torch.no_grad():
        assert torch.equal(reloaded(**batch).logits, model(**batch).logits)
