"""Checkpoint loading: load_inference_bundle, load_run."""

import json
import yaml
from pathlib import Path
from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from calt.io.preprocessor import NumberPolicy, UnifiedLexer
from calt.io.vocabulary.config import VocabConfig

from .manifest import load_manifest, resolve_paths
from .types import RunBundle


# Model types that use seq2seq; rest treated as causal
_SEQ2SEQ_TYPES = {"bart", "t5", "marian", "pegasus", "mbart", "blenderbot", "m2m_100"}


def _detect_model_type(model_path: Path) -> str:
    """Infer 'seq2seq' or 'causal' from config.json in model dir."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return "seq2seq"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mt = (cfg.get("model_type") or "").lower()
    arch = cfg.get("architectures") or []
    arch_str = " ".join(arch).lower()
    if mt in _SEQ2SEQ_TYPES or any(
        a in arch_str for a in ("bart", "t5", "marian", "pegasus", "mbart", "encoderdecoder")
    ):
        return "seq2seq"
    return "causal"


def _load_model(model_path: Path, model_type: str, device: str):
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
    return model.to(device)


def load_inference_bundle(
    output_dir: str | Path,
    *,
    device: str = "cpu",
    model_subdir: str | None = None,
    tokenizer_subdir: str | None = None,
    model_type: str | None = None,
):
    """Load model and tokenizer for minimal inference.

    Assumes inference input is already token_text (space-separated tokens).
    If manifest exists, paths and model_type are taken from it; otherwise
    defaults to model/ and tokenizer/, and model_type is inferred from config.json.

    Args:
        output_dir: Run root (e.g. training output_dir).
        device: Target device ("cpu", "cuda", "cuda:0", etc.).
        model_subdir: Override model subdir (default from manifest or "model").
        tokenizer_subdir: Override tokenizer subdir (default from manifest or "tokenizer").
        model_type: Override "seq2seq" or "causal" (default from manifest or config.json).

    Returns:
        (model, tokenizer)
    """
    root = Path(output_dir)
    manifest = load_manifest(output_dir)

    if model_subdir is not None:
        model_path = root / model_subdir
    elif manifest and "paths" in manifest and manifest["paths"].get("model_dir"):
        model_path = root / manifest["paths"]["model_dir"]
    else:
        model_path = root / "model"

    if tokenizer_subdir is not None:
        tokenizer_path = root / tokenizer_subdir
    elif manifest and "paths" in manifest and manifest["paths"].get("tokenizer_dir"):
        tokenizer_path = root / manifest["paths"]["tokenizer_dir"]
    else:
        tokenizer_path = root / "tokenizer"

    if model_type is None and manifest and manifest.get("model_type"):
        model_type = manifest["model_type"]
    if model_type is None:
        model_type = _detect_model_type(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    model = _load_model(model_path, model_type, device)
    return model, tokenizer


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_lexer_from_config(lexer_config: dict, vocab_config: VocabConfig) -> UnifiedLexer:
    """Build UnifiedLexer from lexer_config (number_policy, strict, etc.) and VocabConfig."""
    np = lexer_config.get("number_policy") or {}
    number_policy = NumberPolicy(
        sign=np.get("sign", "separate"),
        digit_group=np.get("digit_group", 0),
        allow_float=np.get("allow_float", True),
        dot_token=np.get("dot_token", "."),
    )
    lexer_cfg = lexer_config.get("lexer") or {}
    strict = lexer_cfg.get("strict", lexer_config.get("strict", True))
    return UnifiedLexer(
        vocab_config=vocab_config,
        number_policy=number_policy,
        strict=strict,
        include_base_vocab=lexer_config.get("include_base_vocab", True),
    )


def load_run(
    output_dir: str | Path,
    *,
    device: str = "cpu",
    load_lexer: bool = True,
    model_subdir: str | None = None,
    tokenizer_subdir: str | None = None,
    model_type: str | None = None,
) -> RunBundle:
    """Load a full run bundle: model, tokenizer, vocab_config, lexer, and optionally lexer.

    When load_lexer=True, requires calt/vocab.yaml and calt/lexer.yaml.
    When load_lexer=False, vocab_config/lexer_config/vocab/lexer are built from
    available YAMLs if present; otherwise they are left as empty dicts and vocab/lexer
    are None. For simplicity we always require vocab+lexer when load_lexer=True and
    build them when the YAMLs exist; if load_lexer=True and YAMLs are missing we raise.

    Args:
        output_dir: Run root.
        device: Target device.
        load_lexer: If True, require and load vocab.yaml + lexer.yaml and build lexer.
        model_subdir: Override model subdir.
        tokenizer_subdir: Override tokenizer subdir.
        model_type: Override model type.

    Returns:
        RunBundle with model, tokenizer, vocab_config, lexer_config, vocab, lexer, etc.
    """
    root = Path(output_dir)
    manifest = load_manifest(output_dir)

    model_path, tokenizer_path = resolve_paths(output_dir, manifest)
    if model_subdir is not None:
        model_path = root / model_subdir
    if tokenizer_subdir is not None:
        tokenizer_path = root / tokenizer_subdir

    if model_type is None and manifest and manifest.get("model_type"):
        model_type = manifest["model_type"]
    if model_type is None:
        model_type = _detect_model_type(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    model = _load_model(model_path, model_type, device)

    calt_dir = root / "calt"
    train_config: dict | None = None
    vocab_config: dict = {}
    lexer_config: dict = {}
    vocab: VocabConfig | None = None
    lexer: UnifiedLexer | None = None

    train_path = calt_dir / "train.yaml"
    if train_path.exists():
        train_config = _load_yaml(train_path)

    vocab_path = calt_dir / "vocab.yaml"
    lexer_path = calt_dir / "lexer.yaml"

    if load_lexer:
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"load_lexer=True requires {vocab_path}. Not found."
            )
        if not lexer_path.exists():
            raise FileNotFoundError(
                f"load_lexer=True requires {lexer_path}. Not found."
            )
        vocab_config = _load_yaml(vocab_path)
        lexer_config = _load_yaml(lexer_path)
        voc = VocabConfig([], {})
        voc.from_config(vocab_config)
        vocab = voc
        lexer = _build_lexer_from_config(lexer_config, vocab)
    else:
        if vocab_path.exists():
            vocab_config = _load_yaml(vocab_path)
            voc = VocabConfig([], {})
            voc.from_config(vocab_config)
            vocab = voc
        if lexer_path.exists():
            lexer_config = _load_yaml(lexer_path)
            if vocab is not None:
                lexer = _build_lexer_from_config(lexer_config, vocab)

    _vocab = vocab or VocabConfig([], {})
    _lexer = lexer or _build_lexer_from_config(lexer_config or {}, _vocab)

    return RunBundle(
        model=model,
        tokenizer=tokenizer,
        vocab_config=vocab_config,
        lexer_config=lexer_config,
        vocab=_vocab,
        lexer=_lexer,
        train_config=train_config,
        manifest=manifest,
    )
