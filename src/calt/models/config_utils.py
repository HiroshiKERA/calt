"""Helpers shared by the config mappings (cfg.model -> model config)."""

import json
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .losses import DEFAULT_LOSS, RESERVED_LOSS_ARGS, check_loss_registered


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


def _json_kwargs(
    model_config: DictConfig, key: str, reserved: frozenset[str]
) -> dict[str, Any]:
    """Return ``model_config[key]`` as a plain dict of JSON values, checked.

    The values must survive the JSON round trip of ``save_pretrained``; checking
    here makes a bad value fail when the model is built rather than at the
    first checkpoint, and normalizes tuples to lists as the reloaded config has.
    """
    kwargs = plain_dict(model_config, key)
    bad_keys = [name for name in kwargs if not isinstance(name, str)]
    if bad_keys:
        raise TypeError(f"model.{key} keys must be strings, got {bad_keys!r}")
    try:
        kwargs = json.loads(json.dumps(kwargs))
    except (TypeError, ValueError) as e:
        raise TypeError(f"model.{key} must be JSON-serializable: {e}") from e
    clash = sorted(reserved & set(kwargs))
    if clash:
        raise ValueError(f"model.{key} must not set {clash}; calt passes these itself.")
    return kwargs


def input_embedding_kwargs(model_config: DictConfig) -> dict[str, Any]:
    """Return ``model.input_embedding_kwargs`` as a plain dict, checked."""
    return _json_kwargs(
        model_config, "input_embedding_kwargs", frozenset({"vocab_size", "d_model"})
    )


def loss_settings(model_config: DictConfig) -> dict[str, Any]:
    """Return ``loss_type`` and ``loss_kwargs`` for a model config, checked.

    A new model needs its loss now, so an unregistered ``loss_type`` raises here
    (loading a saved model only warns; see ``calt.models.losses``).
    """
    loss_type = getattr(model_config, "loss_type", None) or DEFAULT_LOSS
    if not isinstance(loss_type, str):
        raise TypeError(f"model.loss_type must be a string, got {loss_type!r}")
    check_loss_registered(loss_type)
    return {
        "loss_type": loss_type,
        "loss_kwargs": _json_kwargs(model_config, "loss_kwargs", RESERVED_LOSS_ARGS),
    }


# Keys only the generic, decoder_only and encoder_classifier models honour.
EXTENSION_KEYS = ("input_embedding_kwargs", "loss_type", "loss_kwargs")


def reject_unsupported_keys(
    model_config: DictConfig, keys: tuple[str, ...], model_type: str
) -> None:
    """Raise if ``model_config`` sets any of ``keys``, which ``model_type`` ignores.

    ``loss_type: cross_entropy`` is the default every model already uses, so
    naming it explicitly is allowed.
    """
    given = []
    for key in keys:
        value = getattr(model_config, key, None)
        if key == "loss_type" and isinstance(value, str):
            # Empty means the default, as in loss_settings
            value = bool(value) and value.lower() != DEFAULT_LOSS
        if value:
            given.append(key)
    if given:
        raise ValueError(
            f"model_type {model_type!r} does not support {', '.join(given)}; "
            "use generic, decoder_only or encoder_classifier."
        )
