"""Checkpoint saving: save_run."""

import json
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def _save_config(cfg: Any, path: Path) -> None:
    """Save DictConfig or dict to YAML. Never pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if OmegaConf.is_config(cfg):
        OmegaConf.save(cfg, str(path))
    elif isinstance(cfg, dict):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)
    else:
        raise TypeError(f"Expected DictConfig or dict, got {type(cfg)}")


def save_run(
    output_dir: str | Path,
    *,
    model=None,
    tokenizer=None,
    train_config: DictConfig | dict | None = None,
    vocab_config: DictConfig | dict | None = None,
    lexer_config: DictConfig | dict | None = None,
    task: str | None = None,
    input_format: str = "token_text",
    model_type: str | None = None,
    extra_meta: dict | None = None,
    overwrite: bool = True,
) -> None:
    """Save a training run for later inference or full reproduction.

    Ensures output_dir/calt/ exists. Saves model and tokenizer to model/ and
    tokenizer/ subdirs when provided. Writes train.yaml, vocab.yaml, lexer.yaml
    into calt/ when given. Always writes calt/manifest.yaml.

    Args:
        output_dir: Run root (e.g. Trainer output_dir).
        model: HuggingFace PreTrainedModel. If None, assume already saved.
        tokenizer: HuggingFace tokenizer. If None, assume already saved.
        train_config: Training config (OmegaConf or dict) for calt/train.yaml.
        vocab_config: Vocab config for calt/vocab.yaml.
        lexer_config: Lexer config (NumberPolicy etc.) for calt/lexer.yaml.
        task: Optional task name (e.g. polynomial, arithmetic).
        input_format: "token_text" or "raw_text".
        model_type: Optional "seq2seq" or "causal" for loading.
        extra_meta: Optional dict for calt/meta.json (e.g. git commit, versions).
        overwrite: If True, overwrite existing files.
    """
    root = Path(output_dir)
    calt_dir = root / "calt"
    calt_dir.mkdir(parents=True, exist_ok=True)

    model_dir = "model"
    tokenizer_dir = "tokenizer"

    if model is not None:
        d = root / model_dir
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d)

    if tokenizer is not None:
        d = root / tokenizer_dir
        d.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(d)

    if train_config is not None:
        _save_config(train_config, calt_dir / "train.yaml")

    if vocab_config is not None:
        _save_config(vocab_config, calt_dir / "vocab.yaml")

    if lexer_config is not None:
        _save_config(lexer_config, calt_dir / "lexer.yaml")

    if extra_meta is not None:
        with open(calt_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(extra_meta, f, indent=2, ensure_ascii=False)

    from .manifest import default_manifest, save_manifest

    manifest = default_manifest(
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        input_format=input_format,
        task=task,
        has_train=train_config is not None,
        has_vocab=vocab_config is not None,
        has_lexer=lexer_config is not None,
        model_type=model_type,
    )
    save_manifest(output_dir, manifest)
