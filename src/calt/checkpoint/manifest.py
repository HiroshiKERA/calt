"""Manifest read/write and path resolution."""

import yaml
from pathlib import Path
from typing import Any


def _get_calt_version() -> str:
    try:
        from importlib.metadata import version
        return version("calt-x")
    except Exception:
        return "0.0.0"


def default_manifest(
    *,
    model_dir: str = "model",
    tokenizer_dir: str = "tokenizer",
    input_format: str = "token_text",
    task: str | None = None,
    run_id: str | None = None,
    has_train: bool = False,
    has_vocab: bool = False,
    has_lexer: bool = False,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Build the default manifest dict."""

    return {
        "calt_version": _get_calt_version(),
        "run_id": run_id,
        "task": task,
        "input_format": input_format,
        "model_type": model_type,
        "paths": {
            "model_dir": model_dir,
            "tokenizer_dir": tokenizer_dir,
            "train_config": "calt/train.yaml" if has_train else None,
            "vocab_config": "calt/vocab.yaml" if has_vocab else None,
            "lexer_config": "calt/lexer.yaml" if has_lexer else None,
        },
    }


def load_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    """Load manifest.yaml from output_dir/calt/. Returns None if missing."""

    path = Path(output_dir) / "calt" / "manifest.yaml"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> None:
    """Write manifest.yaml into output_dir/calt/."""

    calt_dir = Path(output_dir) / "calt"
    calt_dir.mkdir(parents=True, exist_ok=True)
    path = calt_dir / "manifest.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, allow_unicode=True)


def resolve_paths(output_dir: str | Path, manifest: dict[str, Any] | None) -> tuple[Path, Path]:
    """Resolve (model_dir, tokenizer_dir) as absolute paths.

    Uses manifest.paths if present; otherwise defaults to model/ and tokenizer/.
    """
    root = Path(output_dir)
    if manifest and "paths" in manifest:
        p = manifest["paths"]
        model_dir = p.get("model_dir", "model")
        tokenizer_dir = p.get("tokenizer_dir", "tokenizer")
    else:
        model_dir = "model"
        tokenizer_dir = "tokenizer"
    return (root / model_dir, root / tokenizer_dir)
