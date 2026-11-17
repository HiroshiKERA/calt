"""Helpers shared by the config mappings (cfg.model -> model config)."""

import json
from typing import Any

from omegaconf import DictConfig, OmegaConf


def plain_dict(model_config: DictConfig, key: str) -> dict[str, Any]:
    """Return ``model_config[key]`` as a plain dict ({} if missing or null).

    Model configs are saved as JSON by ``save_pretrained``, which cannot
    serialize an OmegaConf DictConfig.
    """
    value = getattr(model_config, key, None)
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(
            f"model.{key} must be a mapping, got {type(value).__name__}: {value!r}"
        )
    return dict(value)


def input_embedding_kwargs(model_config: DictConfig) -> dict[str, Any]:
    """Return ``model.input_embedding_kwargs`` as a plain dict, checked.

    The values must survive the JSON round trip of ``save_pretrained``; checking
    here makes a bad value fail when the model is built rather than at the
    first checkpoint, and normalizes tuples to lists as the reloaded config has.
    """
    kwargs = plain_dict(model_config, "input_embedding_kwargs")
    bad_keys = [key for key in kwargs if not isinstance(key, str)]
    if bad_keys:
        raise TypeError(
            f"model.input_embedding_kwargs keys must be strings, got {bad_keys!r}"
        )
    try:
        kwargs = json.loads(json.dumps(kwargs))
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"model.input_embedding_kwargs must be JSON-serializable: {e}"
        ) from e
    clash = sorted({"vocab_size", "d_model"} & set(kwargs))
    if clash:
        raise ValueError(
            f"model.input_embedding_kwargs must not set {clash}; they come from "
            "the rest of the model config."
        )
    return kwargs


def reject_unsupported_keys(
    model_config: DictConfig, keys: tuple[str, ...], model_type: str
) -> None:
    """Raise if ``model_config`` sets any of ``keys``, which ``model_type`` ignores."""
    given = [key for key in keys if getattr(model_config, key, None)]
    if given:
        raise ValueError(
            f"model_type {model_type!r} does not support {', '.join(given)}; "
            "use generic, decoder_only or encoder_classifier."
        )
