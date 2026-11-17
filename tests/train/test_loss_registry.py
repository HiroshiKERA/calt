"""Tests for register_loss, model.loss_type / model.loss_kwargs and extra labels.

The generic, decoder_only and encoder_classifier models compute their loss
through calt.models.losses; the default must stay exactly the old
cross-entropy, and a registered loss gets logits, labels, hidden and the extra
labels returned by the load preprocessor.
"""

import functools
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers.utils import find_labels

from calt.io import IOPipeline, UserCallableLoadPreprocessor
from calt.io.extra_labels import split_sample
from calt.io.preprocess import _texts_only
from calt.io.preprocessor import ChainLoadPreprocessor
from calt.models import ModelPipeline, register_loss
from calt.models.bart.config_mapping import create_bart_config
from calt.models.config_utils import EXTENSION_KEYS, reject_unsupported_keys
from calt.models.decoder_only.model import DecoderOnlyConfig, DecoderOnlyTransformer
from calt.models.encoder_classifier import EncoderClassifier
from calt.models.encoder_classifier.model import EncoderClassifierConfig
from calt.models.generic.model import Transformer, TransformerConfig
from calt.models.losses import LOSS_REGISTRY
from calt.models.monomial.config_mapping import create_monomial_config
from calt.models.monomial_decoder_only.config_mapping import (
    create_monomial_decoder_only_config,
)
from calt.trainer import TrainerPipeline
from calt.trainer.pipeline import _check_extra_labels_reach_a_loss
from calt.trainer.trainer import Trainer

V, D = 20, 16
PAD, EOS, BOS = 0, 1, 2

CONFIGS = {
    "generic": lambda **kw: TransformerConfig(
        d_model=D,
        attention_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        vocab_size=V,
        max_input_len=16,
        **kw,
    ),
    "decoder_only": lambda **kw: DecoderOnlyConfig(
        d_model=D,
        attention_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        vocab_size=V,
        max_input_len=16,
        **kw,
    ),
    "encoder_only": lambda **kw: EncoderClassifierConfig(
        d_model=D,
        attention_heads=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        vocab_size=V,
        max_input_len=16,
        **kw,
    ),
}
MODELS = {
    "generic": Transformer,
    "decoder_only": DecoderOnlyTransformer,
    "encoder_only": EncoderClassifier,
}
KINDS = sorted(MODELS)


def _model(kind, **config_kwargs):
    return MODELS[kind](CONFIGS[kind](**config_kwargs))


def _batch():
    """Two rows; the second answer is one token shorter (padded with -100)."""
    input_ids = torch.tensor([[BOS, 5, 6, 7, EOS], [BOS, 8, 9, EOS, PAD]])
    decoder_input_ids = torch.tensor([[BOS, 10, 11, 12], [BOS, 13, 14, PAD]])
    labels = torch.tensor([[10, 11, 12, EOS], [13, 14, EOS, -100]])
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != PAD).long(),
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": (decoder_input_ids != PAD).long(),
        "labels": labels,
    }


def _logits_of(model, batch):
    """Run forward and capture the real logits (forward returns argmax ids)."""
    head = model.classifier if isinstance(model, EncoderClassifier) else model.lm_head
    captured = {}
    handle = head.register_forward_hook(
        lambda module, args, output: captured.update(logits=output)
    )
    try:
        output = model(**batch)
    finally:
        handle.remove()
    return output, captured["logits"]


class CarryLoss(nn.Module):
    """README §6 style: CE plus a weighted auxiliary head on `hidden`."""

    def __init__(self, d_model, weight=0.1, **_):
        super().__init__()
        self.head = nn.Linear(d_model, 1)
        self.weight = weight

    def forward(self, logits, labels, hidden, carry_labels=None, **_):
        main = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
        )
        if carry_labels is None:
            return main
        mask = labels != -100
        aux = F.binary_cross_entropy_with_logits(
            self.head(hidden).squeeze(-1)[mask], carry_labels[mask].float()
        )
        return main + self.weight * aux


class TwoHeadLoss(nn.Module):
    """A per-token head and a per-sample head (README §6 MultiTaskLoss)."""

    def __init__(self, d_model):
        super().__init__()
        self.carry_head = nn.Linear(d_model, 1)
        self.length_head = nn.Linear(d_model, 7)

    def forward(self, logits, labels, hidden, carry_labels, length_label):
        main = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
        )
        mask = labels != -100
        carry = F.binary_cross_entropy_with_logits(
            self.carry_head(hidden).squeeze(-1)[mask], carry_labels[mask].float()
        )
        length = F.cross_entropy(self.length_head(hidden[:, 0]), length_label)
        return main + carry + length


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(LOSS_REGISTRY)
    yield
    LOSS_REGISTRY.clear()
    LOSS_REGISTRY.update(saved)


# --------------------------------------------------------------------------- #
# Default loss (R4)                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", KINDS)
def test_default_loss_equals_the_previous_formula(kind):
    model = _model(kind).eval()
    batch = _batch()
    output, logits = _logits_of(model, batch)
    labels = batch["labels"]
    if kind == "generic":
        expected = nn.CrossEntropyLoss()(logits.view(-1, V), labels.view(-1))
    elif kind == "decoder_only":
        expected = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.reshape(-1, V), labels.reshape(-1)
        )
    else:
        target = EncoderClassifier._target_classes(labels)
        expected = nn.CrossEntropyLoss(ignore_index=-100)(logits, target)
    assert torch.equal(output.loss, expected)


@pytest.mark.parametrize("kind", KINDS)
def test_default_model_adds_no_module(kind):
    model = _model(kind)
    assert not hasattr(model, "loss_module")
    assert not any(key.startswith("loss_module") for key in model.state_dict())
    assert model.config.loss_type == "cross_entropy"
    assert model.config.loss_kwargs == {}


def _seeded_model(kind, **config_kwargs):
    # As a train script does with set_seed: the same seed before each build
    torch.manual_seed(0)
    return _model(kind, **config_kwargs)


@pytest.mark.parametrize("kind", KINDS)
def test_loss_module_does_not_change_the_initial_weights(kind):
    register_loss("carry", CarryLoss)
    default = _seeded_model(kind)
    custom = _seeded_model(kind, loss_type="carry")
    for key, value in default.state_dict().items():
        assert torch.equal(custom.state_dict()[key], value), key
    assert any(key.startswith("loss_module.") for key in custom.state_dict())


@pytest.mark.parametrize("kind", ["generic", "decoder_only"])
def test_generation_does_not_depend_on_the_loss(kind):
    register_loss("carry", CarryLoss)
    batch = _batch()
    default = _seeded_model(kind).eval()
    custom = _seeded_model(kind, loss_type="carry").eval()
    args = dict(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_length=5,
    )
    assert torch.equal(default.generate(**args), custom.generate(**args))


# --------------------------------------------------------------------------- #
# Registered losses (R1)                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", KINDS)
def test_function_loss_gets_declared_arguments(kind):
    seen = {}

    def probe(logits, labels, hidden, scale):
        seen.update(logits=logits, labels=labels, hidden=hidden, scale=scale)
        return logits.float().mean() * scale

    register_loss("probe", probe)
    model = _model(kind, loss_type="probe", loss_kwargs={"scale": 2.0})
    output, logits = _logits_of(model, _batch())

    assert seen["scale"] == 2.0
    assert torch.equal(seen["logits"], logits)
    assert torch.allclose(output.loss, logits.mean() * 2.0)
    if kind == "encoder_only":
        assert seen["logits"].shape == (2, V)
        assert seen["labels"].tolist() == [10, 13]  # the answer token per row
        assert seen["hidden"].shape == (2, D)
    else:
        assert seen["logits"].shape == (2, 4, V)
        assert torch.equal(seen["labels"], _batch()["labels"])
        assert seen["hidden"].shape == (2, 4, D)


def test_misspelled_loss_kwarg_raises():
    register_loss("strict", lambda logits, labels, strength=1.0: logits.mean())
    model = _model("decoder_only", loss_type="strict", loss_kwargs={"strenght": 2})
    with pytest.raises(TypeError, match="strenght"):
        model(**_batch())


def test_readme_sequence_loss_with_double_underscore_kwargs():
    def sequence_loss(logits, labels, strength=1.0, power=2.0, **_):
        return (
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            * strength
        )

    register_loss("sequence", sequence_loss)
    model = _model("decoder_only", loss_type="sequence", loss_kwargs={"strength": 3})
    default = _model("decoder_only")
    batch = _batch()
    assert torch.allclose(model(**batch).loss, default(**batch).loss * 3)


def test_class_loss_is_built_once_as_a_submodule():
    calls = []

    class Counting(CarryLoss):
        def __init__(self, d_model, weight=0.1):
            calls.append((d_model, weight))
            super().__init__(d_model, weight)

    register_loss("counting", Counting)
    model = _model("decoder_only", loss_type="counting", loss_kwargs={"weight": 0.5})
    model(**_batch())
    model(**_batch())
    assert calls == [(D, 0.5)]
    assert isinstance(model.loss_module, Counting)
    assert model.loss_module.weight == 0.5


def test_partial_with_keyword_arguments_is_a_class_loss():
    register_loss("half", functools.partial(CarryLoss, weight=0.5))
    model = _model("generic", loss_type="half")
    assert model.loss_module.weight == 0.5


def test_both_heads_are_trained_and_saved(tmp_path):
    register_loss("two_heads", TwoHeadLoss)
    model = _model("decoder_only", loss_type="two_heads")
    batch = _batch()
    batch["extra_targets"] = {
        "carry_labels": torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]]),
        "length_label": torch.tensor([3, 2]),
    }
    before = {k: v.clone() for k, v in model.loss_module.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model(**batch).loss.backward()
    optimizer.step()
    for key, value in model.loss_module.state_dict().items():
        assert not torch.equal(value, before[key]), key

    model.save_pretrained(tmp_path)
    reloaded = DecoderOnlyTransformer.from_pretrained(tmp_path)
    assert isinstance(reloaded.loss_module, TwoHeadLoss)
    for key, value in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], value), key
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["loss_type"] == "two_heads"
    assert saved["loss_kwargs"] == {}


def test_unregistered_loss_on_load_warns_and_still_generates(tmp_path, caplog):
    register_loss("carry", CarryLoss)
    model = _model("decoder_only", loss_type="carry")
    model.save_pretrained(tmp_path)
    del LOSS_REGISTRY["carry"]

    with caplog.at_level(logging.WARNING, logger="calt.models.losses"):
        reloaded = DecoderOnlyTransformer.from_pretrained(tmp_path)
    assert "not registered" in caplog.text
    batch = _batch()
    out = reloaded.generate(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    assert out.size(0) == 2
    with pytest.raises(RuntimeError, match="register_loss"):
        reloaded(**batch)


def test_checkpoint_without_loss_keys_loads(tmp_path):
    model = _model("generic")
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    saved = json.loads(config_path.read_text())
    saved.pop("loss_type")
    saved.pop("loss_kwargs")
    config_path.write_text(json.dumps(saved))
    reloaded = Transformer.from_pretrained(tmp_path)
    assert reloaded.config.loss_type == "cross_entropy"
    assert torch.equal(reloaded(**_batch()).loss, model(**_batch()).loss)


def test_registering_errors():
    with pytest.raises(ValueError, match="built-in"):
        register_loss("cross_entropy", lambda **kw: None)
    with pytest.raises(TypeError, match="instance"):
        register_loss("inst", CarryLoss(D))


def test_function_returning_a_module_raises():
    register_loss("builder", lambda **kw: CarryLoss(D))
    model = _model("decoder_only", loss_type="builder")
    with pytest.raises(TypeError, match="register the class"):
        model(**_batch())


def test_non_scalar_loss_raises():
    register_loss("vector", lambda logits, labels: logits.mean(dim=-1))
    model = _model("decoder_only", loss_type="vector")
    with pytest.raises(TypeError, match="scalar tensor"):
        model(**_batch())


def test_default_loss_takes_no_kwargs():
    with pytest.raises(ValueError, match="takes no model.loss_kwargs"):
        _model("generic", loss_kwargs={"weight": 1})


# --------------------------------------------------------------------------- #
# Config mapping and unsupported models (R3)                                   #
# --------------------------------------------------------------------------- #


def _model_cfg(kind, **overrides):
    cfg = OmegaConf.create(
        {
            "model_type": kind,
            "vocab_size": V,
            "d_model": D,
            "num_encoder_heads": 2,
            "num_decoder_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "max_sequence_length": 16,
            "dropout": 0.0,
        }
    )
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


@pytest.mark.parametrize("kind", ["generic", "decoder_only", "encoder_only"])
def test_config_mapping_passes_loss_settings(kind):
    register_loss("carry", CarryLoss)
    cfg = _model_cfg(kind, loss_type="carry", loss_kwargs={"weight": 0.3})
    model = ModelPipeline(cfg, tokenizer=None).build()
    assert model.config.loss_type == "carry"
    assert type(model.config.loss_kwargs) is dict
    assert model.loss_module.weight == 0.3


def test_unknown_loss_type_raises_when_building():
    cfg = _model_cfg("decoder_only", loss_type="nope")
    with pytest.raises(ValueError, match="Unknown model.loss_type 'nope'"):
        ModelPipeline(cfg, tokenizer=None).build()


@pytest.mark.parametrize("key", ["logits", "labels", "hidden", "d_model"])
def test_loss_kwargs_cannot_shadow_loss_arguments(key):
    register_loss("carry", CarryLoss)
    cfg = _model_cfg("generic", loss_type="carry", loss_kwargs={key: 1})
    with pytest.raises(ValueError, match="must not set"):
        ModelPipeline(cfg, tokenizer=None).build()


@pytest.mark.parametrize("model_type", ["bart", "monomial", "monomial_decoder_only"])
@pytest.mark.parametrize(
    "overrides",
    [{"loss_type": "carry"}, {"loss_kwargs": {"weight": 1}}],
)
def test_unsupported_models_reject_loss_settings(model_type, overrides):
    create_config = {
        "bart": create_bart_config,
        "monomial": create_monomial_config,
        "monomial_decoder_only": create_monomial_decoder_only_config,
    }[model_type]
    cfg = _model_cfg(model_type, **overrides)
    with pytest.raises(ValueError, match="does not support loss_"):
        create_config(cfg, None)


def test_unsupported_model_allows_explicit_default_loss():
    cfg = _model_cfg("bart", loss_type="cross_entropy")
    reject_unsupported_keys(cfg, EXTENSION_KEYS, "bart")  # no error
    cfg.loss_type = "CROSS_ENTROPY"
    reject_unsupported_keys(cfg, EXTENSION_KEYS, "monomial")  # no error
    cfg.loss_type = ""  # empty means the default, as in loss_settings
    reject_unsupported_keys(cfg, EXTENSION_KEYS, "monomial")  # no error


def test_var_keyword_label_names_do_not_warn(caplog):
    register_loss("kw", lambda logits, labels, **extra_labels: logits.mean())
    model = _model("decoder_only", loss_type="kw")
    with caplog.at_level(logging.WARNING, logger="calt.models.losses"):
        model(**_batch())
    assert caplog.text == ""


def test_register_loss_needs_a_string_name():
    with pytest.raises(TypeError, match="string name"):
        register_loss(None, CarryLoss)


# --------------------------------------------------------------------------- #
# Extra labels through the IO pipeline (R2)                                    #
# --------------------------------------------------------------------------- #

LEXER = {
    "vocab": {
        "range": {"numbers": ["", 0, 9]},
        "misc": ["+", "="],
        "special_tokens": {},
        "flags": {"include_base_vocab": False, "include_base_special_tokens": True},
    },
    "number_policy": {"attach_sign": False, "digit_group": 1, "allow_float": False},
    "strict": True,
    "include_base_vocab": False,
}


def _carry_targets(left, right):
    carry, targets = 0, []
    for a, b in zip(left, right):
        carry = int(int(a) + int(b) + carry >= 10)
        targets.append(carry)
    return targets


def _format(problem, answer, extra=lambda left, right, target: {}):
    """Zero-padded, lowest digit first (the tutorial's notation)."""
    a, b = map(int, problem)
    width = max(3, len(str(a)), len(str(b)))
    left = f"{a:0{width}d}"[::-1]
    right = f"{b:0{width}d}"[::-1]
    target = f"{int(answer):0{width + 1}d}"[::-1]
    labels = extra(left, right, target)
    text = (" ".join(f"{left}+{right}="), " ".join(target))
    return (*text, labels) if labels is not None else text


def _carry_extra(left, right, target):
    return {"carry_labels": _carry_targets(left, right) + [0]}


def _write_data(tmp_path, n=48, seed=0):
    rng = np.random.default_rng(seed)
    for split, count in (("train", n), ("test", 8)):
        lines = []
        for _ in range(count):
            a, b = (int(x) for x in rng.integers(0, 1000, size=2))
            lines.append(
                json.dumps({"problem": [str(a), str(b)], "answer": str(a + b)})
            )
        (tmp_path / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    lexer_path = tmp_path / "lexer.yaml"
    lexer_path.write_text(OmegaConf.to_yaml(OmegaConf.create(LEXER)))
    return OmegaConf.create(
        {
            "lexer_config": str(lexer_path),
            "train_dataset_jsonl": str(tmp_path / "train.jsonl"),
            "test_dataset_jsonl": str(tmp_path / "test.jsonl"),
            "use_jsonl": True,
        }
    )


def _io(tmp_path, fn, **write_kwargs):
    pipeline = IOPipeline.from_config(_write_data(tmp_path, **write_kwargs))
    pipeline.dataset_load_preprocessor = UserCallableLoadPreprocessor(fn)
    return pipeline.build()


def _collate(io_dict, indices=(0, 1, 2)):
    items = [io_dict["train_dataset"][i] for i in indices]
    return io_dict["data_collator"](items)


def test_extra_labels_are_padded_and_batched(tmp_path):
    def fn(problem, answer):
        return _format(
            problem,
            answer,
            lambda left, right, target: {
                "carry_labels": _carry_targets(left, right) + [0],
                "length_label": len(str(int(answer))),
            },
        )

    batch = _collate(_io(tmp_path, fn))
    extra = batch["extra_targets"]
    labels = batch["labels"]
    assert set(extra) == {"carry_labels", "length_label"}
    assert extra["carry_labels"].shape == labels.shape
    assert extra["carry_labels"].dtype == torch.long
    assert extra["length_label"].shape == (labels.size(0),)
    # T-1 values: calt appends one value at the EOS position, and it is 0
    eos_col = (labels != -100).sum(dim=1) - 1
    rows = torch.arange(labels.size(0))
    assert (extra["carry_labels"][rows, eos_col] == 0).all()
    assert not (extra["carry_labels"] == -100).any()


def test_padding_of_extra_labels_is_zero(tmp_path):
    def fn(problem, answer):
        a, b = map(int, problem)
        # Different answer lengths in one batch
        target = " ".join(str(a + b))
        return "1 + 2 =", target, {"digits": [7] * (len(str(a + b)) + 1)}

    io_dict = _io(tmp_path, fn)
    batch = _collate(io_dict, indices=range(8))
    labels, digits = batch["labels"], batch["extra_targets"]["digits"]
    assert torch.equal(digits[labels == -100], torch.zeros_like(digits[labels == -100]))
    assert (digits[labels != -100] == 7).all()


def test_label_two_short_is_rejected_before_training(tmp_path):
    def fn(problem, answer):
        return _format(
            problem,
            answer,
            lambda left, right, target: {"carry_labels": _carry_targets(left, right)},
        )

    with pytest.raises(ValueError, match="must have 5 or 4 values"):
        _io(tmp_path, fn)


def test_collator_also_checks_lengths(tmp_path):
    io_dict = _io(tmp_path, lambda p, a: _format(p, a, _carry_extra))
    item = io_dict["train_dataset"][0]
    item["extra_targets"] = {"carry_labels": [1]}
    with pytest.raises(ValueError, match="carry_labels"):
        io_dict["data_collator"]([item])


def test_samples_must_return_the_same_names(tmp_path):
    calls = []

    def fn(problem, answer):
        calls.append(1)
        extra = {"carry_labels": [0] * 4} if len(calls) % 2 else {}
        return _format(problem, answer, lambda *args: extra)

    with pytest.raises(ValueError, match="same"):
        _io(tmp_path, fn)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"labels": 1}, "reserved"),
        ({"decoder_x": 1}, "reserved"),
        ({"tag": "x"}, "number"),
        ({"nested": [[1, 2]]}, "flat sequence"),
    ],
)
def test_bad_extra_labels_are_rejected(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        _io(tmp_path, lambda p, a: _format(p, a, lambda *args: extra))


def test_float_and_numpy_labels(tmp_path):
    def fn(problem, answer):
        return _format(
            problem,
            answer,
            lambda left, right, target: {
                "weights": np.full(len(target) + 1, 0.5),
                "flag": np.int64(1),
            },
        )

    batch = _collate(_io(tmp_path, fn))
    assert batch["extra_targets"]["weights"].dtype == torch.float
    assert batch["extra_targets"]["flag"].dtype == torch.long


def test_chain_preprocessor_may_return_extra_labels(tmp_path):
    class Inner:
        def process_sample(self, source):
            return source

    user = UserCallableLoadPreprocessor(lambda p, a: _format(p, a, _carry_extra))
    pipeline = IOPipeline.from_config(_write_data(tmp_path))
    pipeline.dataset_load_preprocessor = ChainLoadPreprocessor(Inner(), user)
    io_dict = pipeline.build()
    assert "carry_labels" in io_dict["train_dataset"][0]["extra_targets"]


def test_two_element_functions_still_work(tmp_path):
    io_dict = _io(tmp_path, lambda p, a: _format(p, a, lambda *args: None))
    item = io_dict["train_dataset"][0]
    assert set(item) == {"input", "target"}
    assert "extra_targets" not in _collate(io_dict)


def test_list_results_still_work():
    assert split_sample(["a", "b"], "x") == ("a", "b", None)
    assert split_sample(["a", "b", {"n": 1}], "x") == ("a", "b", {"n": 1})


@pytest.mark.parametrize("kind", KINDS)
def test_forward_has_no_extra_label_column_for_transformers(kind):
    # transformers treats every forward() argument named "*label*" as a label;
    # a second one would change Trainer.label_names and drop eval_loss.
    assert find_labels(MODELS[kind]) == ["labels"]


def test_preprocessing_cache_rejects_extra_labels():
    assert _texts_only(("a", "b")) == ("a", "b")
    with pytest.raises(ValueError, match="caches cannot store"):
        _texts_only(("a", "b", {"x": 1}))


def test_prepare_inputs_moves_nested_tensors():
    fake = SimpleNamespace(args=SimpleNamespace(device=torch.device("cpu")))
    inputs = {"labels": torch.ones(2), "extra_targets": {"c": torch.zeros(2)}, "n": 3}
    moved = Trainer._prepare_inputs(fake, inputs)
    assert torch.equal(moved["extra_targets"]["c"], torch.zeros(2))
    assert moved["n"] == 3


def test_extra_label_clashing_with_loss_kwargs_fails_before_training():
    register_loss("carry", CarryLoss)
    model = _model("decoder_only", loss_type="carry", loss_kwargs={"weight": 0.5})
    dataset = SimpleNamespace(extra_labels=[{"weight": 2}])
    with pytest.raises(ValueError, match="clash"):
        _check_extra_labels_reach_a_loss(model, dataset)


def test_empty_extra_labels_mean_none(tmp_path):
    io_dict = _io(tmp_path, lambda p, a: _format(p, a, lambda *args: {}))
    assert io_dict["train_dataset"].extra_labels is None
    assert "extra_targets" not in io_dict["train_dataset"][0]


def test_extra_labels_need_a_supported_model():
    dataset = SimpleNamespace(extra_labels=[{"carry_labels": [0]}])
    bart_like = SimpleNamespace(config=SimpleNamespace(model_type="bart"))
    with pytest.raises(ValueError, match="does not pass them to a loss"):
        _check_extra_labels_reach_a_loss(bart_like, dataset)


def test_loss_warns_about_missing_labels(caplog):
    # CarryLoss takes **_, so it would accept any label; the name check still
    # notices that carry_labels never arrives.
    register_loss("carry", CarryLoss)
    model = _model("decoder_only", loss_type="carry")
    batch = _batch()
    batch["extra_targets"] = {"carry_label_typo": torch.zeros(2, 4, dtype=torch.long)}
    with caplog.at_level(logging.WARNING, logger="calt.models.losses"):
        model(**batch)
    assert "takes 'carry_labels'" in caplog.text


def test_loss_warns_about_unused_labels(caplog):
    def strict(logits, labels, carry_labels):
        return logits.mean()

    register_loss("strict", strict)
    model = _model("decoder_only", loss_type="strict")
    batch = _batch()
    batch["extra_targets"] = {
        "carry_labels": torch.zeros(2, 4, dtype=torch.long),
        "length_label": torch.zeros(2, dtype=torch.long),
    }
    with caplog.at_level(logging.WARNING, logger="calt.models.losses"):
        model(**batch)
    assert "'length_label' is not an argument" in caplog.text
    assert "carry_labels" not in caplog.text


# --------------------------------------------------------------------------- #
# Short training on addition (README §8.3)                                      #
# --------------------------------------------------------------------------- #


def _train(tmp_path, loss_type, loss_kwargs=None, fn=None):
    io_dict = _io(tmp_path, fn or (lambda p, a: _format(p, a, _carry_extra)), n=64)
    torch.manual_seed(0)  # model init uses the global RNG, as with set_seed
    model_cfg = _model_cfg(
        "decoder_only",
        num_layers=2,
        num_heads=2,
        ffn_dim=64,
        d_model=32,
        loss_type=loss_type,
        loss_kwargs=loss_kwargs or {},
    )
    model = ModelPipeline.from_io_dict(model_cfg, io_dict).build()
    train_cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "out"),
            "num_train_epochs": 8,
            "batch_size": 16,
            "test_batch_size": 8,
            "learning_rate": 3e-3,
            "lr_scheduler_type": "constant",
            "eval_strategy": "no",
            "save_strategy": "no",
            "logging_steps": 1,
            "report_to": "none",
            "num_workers": 0,
        }
    )
    pipeline = TrainerPipeline.from_io_dict(
        train_cfg, model, io_dict, OmegaConf.create({"no_wandb": True})
    ).build()
    pipeline.train()
    losses = [
        log["loss"] for log in pipeline.trainer.state.log_history if "loss" in log
    ]
    return pipeline, losses


def test_sequence_loss_trains(tmp_path):
    def sequence_loss(logits, labels, strength=1.0, power=2.0):
        token = F.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
        )
        ce = token.sum() / (labels != -100).sum()
        reward = torch.exp(-power * token.sum(dim=1))
        return ce + strength * (1 - reward).mean()

    register_loss("sequence", sequence_loss)
    _, losses = _train(tmp_path, "sequence", {"strength": 0.5})
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3


def test_carry_loss_trains_and_its_head_learns(tmp_path):
    initial = {}

    class RecordingCarryLoss(CarryLoss):
        def __init__(self, d_model, weight=0.1):
            super().__init__(d_model, weight)
            initial["head"] = self.head

    register_loss("carry", RecordingCarryLoss)
    pipeline, losses = _train(tmp_path, "carry", {"weight": 0.5})
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3
    head = pipeline.trainer.model.loss_module.head
    assert head is initial["head"]
    # The head starts from calt's init (zero bias) and is updated by training
    assert head.bias.abs().sum() > 0
    metrics = pipeline.trainer.evaluate()  # auxiliary labels reach eval too
    assert "eval_loss" in metrics
