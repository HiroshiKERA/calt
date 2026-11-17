"""
Registry for **training losses**, selected by ``model.loss_type``.

The generic, decoder_only and encoder_classifier models compute their training
loss through this registry, the same way their input embedding comes from
``register_input_embedding``. The default ``"cross_entropy"`` is the only loss
built into calt and gives exactly the loss these models computed before.

Registering a loss
------------------
Call ``register_loss`` *before* building or loading the model:

    from calt.models import register_loss

    def sequence_loss(logits, labels, strength=1.0):
        ...
        return loss                       # a scalar tensor

    register_loss("sequence", sequence_loss)

then set ``model.loss_type: sequence`` (and optionally ``model.loss_kwargs``)
in train.yaml.

* A **function** is called on every batch with the data arguments below plus
  ``model.loss_kwargs``.
* A **class** (or a ``functools.partial`` of one, with arguments bound by
  keyword) is built once with ``d_model``, ``vocab_size`` and
  ``model.loss_kwargs``, and becomes the submodule ``model.loss_module``.
  Auxiliary heads inside it are trained and saved with the model. Its
  ``forward`` receives the data arguments. It is built after the model's own
  layers, so the model's initial weights do not depend on it, and its Linear,
  Embedding and LayerNorm layers get the same initialization as the model's.
  Its ``__init__`` may run on the meta device inside ``from_pretrained``, so it
  should only create layers.

Data arguments (all keyword arguments):

* ``logits``: ``(B, T, V)`` for generic/decoder_only, ``(B, V)`` for
  encoder_classifier.
* ``labels``: the answer tokens, shaped like ``logits`` without its last
  dimension: ``(B, T)`` ending with EOS, or ``(B,)`` (the single answer token)
  for encoder_classifier. Padding is -100.
* ``hidden``: the representation the logits were computed from: the hidden
  state that predicts ``labels[:, i]``, ``(B, T, d_model)``, or the pooled
  vector ``(B, d_model)`` for encoder_classifier.
* one keyword argument per extra label returned by the load preprocessor.

A loss receives only the data arguments its signature names, unless it takes
``**kwargs``. ``model.loss_kwargs`` are always passed, so a misspelled key
raises a TypeError.

Loading a saved model whose loss is not registered only warns: the model can
still generate, but computing a loss raises until ``register_loss`` is called
before loading.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_LOSS = "cross_entropy"
DATA_ARGS = ("logits", "labels", "hidden")
# Names model.loss_kwargs and extra labels must not reuse.
RESERVED_LOSS_ARGS = frozenset(DATA_ARGS + ("d_model", "vocab_size"))

# name (lowercased) -> loss function, or class building an nn.Module
LOSS_REGISTRY: dict[str, Callable[..., Any]] = {}


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor, **_) -> torch.Tensor:
    """Default loss: token cross-entropy, ignoring labels set to -100."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
    )


LOSS_REGISTRY[DEFAULT_LOSS] = cross_entropy


def register_loss(name: str, factory: Callable[..., Any]) -> None:
    """Register (or override) a loss under ``name``.

    Call it before building the model, and also before loading a saved model
    that uses it (``from_pretrained``, ``load_from_checkpoint``,
    ``TrainerPipeline.resume_from_checkpoint``).

    Args:
        name: Value of ``model.loss_type`` that selects the loss.
        factory: A function returning a scalar tensor, or an ``nn.Module``
            class (or ``functools.partial`` of one) whose ``forward`` does.

    Raises:
        ValueError: If ``name`` is ``"cross_entropy"``, the built-in default.
        TypeError: If ``name`` is not a string, or ``factory`` is an
            ``nn.Module`` instance or not callable.
    """
    if not isinstance(name, str):
        raise TypeError(f"register_loss needs a string name, got {name!r}")
    key = name.lower()
    if key == DEFAULT_LOSS:
        raise ValueError(f"{DEFAULT_LOSS!r} is the built-in loss; use another name.")
    if isinstance(factory, nn.Module):
        raise TypeError(
            f"register_loss({name!r}, ...) got an nn.Module instance; register "
            "the class instead, so that each model builds its own copy and "
            "trains and saves its parameters."
        )
    if not callable(factory):
        raise TypeError(f"register_loss({name!r}, ...) needs a callable.")
    LOSS_REGISTRY[key] = factory


def check_loss_registered(loss_type: str) -> None:
    """Raise ValueError if ``loss_type`` is not registered."""
    if loss_type.lower() not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown model.loss_type {loss_type!r}. Call register_loss before "
            f"building the model. Registered: {sorted(LOSS_REGISTRY)}"
        )


def _is_class(factory: Any) -> bool:
    if isinstance(factory, functools.partial):
        factory = factory.func
    return inspect.isclass(factory)


def _accepted_names(fn: Callable[..., Any]) -> Optional[frozenset[str]]:
    """Keyword names ``fn`` accepts, or None when it takes ``**kwargs``.

    Callables whose signature cannot be read also get None, i.e. everything.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return frozenset(
        p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    )


def _wanted_labels(fn: Callable[..., Any]) -> frozenset[str]:
    """Parameters of ``fn`` that look like extra labels (``*_label(s)``)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(
        p.name
        for p in params
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and p.name not in DATA_ARGS
        and p.name.endswith(("_label", "_labels"))
    )


class LossSpec:
    """The configured loss of one model (plain object, not a module).

    The module of a class loss lives on the model as ``model.loss_module``, so
    that ``.to()``, DataParallel replicas and checkpoints include it; this object
    only holds what does not change with them.
    """

    def __init__(
        self,
        name: str,
        loss_kwargs: dict[str, Any],
        function: Optional[Callable[..., Any]] = None,
        accepted: Optional[frozenset[str]] = None,
        wanted: frozenset[str] = frozenset(),
        missing: bool = False,
    ):
        self.name = name
        self.loss_kwargs = loss_kwargs
        self.function = function
        self.accepted = accepted
        self.wanted = wanted
        self.missing = missing
        self._warned: set[str] = set()

    @property
    def is_default(self) -> bool:
        return self.name == DEFAULT_LOSS


def attach_loss(model: nn.Module, config: Any) -> None:
    """Set up ``config.loss_type`` on ``model`` (call at the end of ``__init__``).

    Sets ``model.calt_loss`` (a :class:`LossSpec`) and, for a class loss,
    ``model.loss_module``. The default loss adds no module, so the state dict of
    a default model is unchanged. Call it after the model has seeded the RNG
    and initialized its own layers.
    """
    name = (getattr(config, "loss_type", None) or DEFAULT_LOSS).lower()
    loss_kwargs = dict(getattr(config, "loss_kwargs", None) or {})
    factory = LOSS_REGISTRY.get(name)

    if name == DEFAULT_LOSS and loss_kwargs:
        raise ValueError(
            f"The default loss {DEFAULT_LOSS!r} takes no model.loss_kwargs, got "
            f"{sorted(loss_kwargs)}; set model.loss_type to a registered loss."
        )
    if factory is None:
        logger.warning(
            "model.loss_type %r is not registered, so this model can generate but "
            "not compute its training loss. Call register_loss(%r, ...) before "
            "loading the model to train or evaluate it.",
            name,
            name,
        )
        model.calt_loss = LossSpec(name, loss_kwargs, missing=True)
        return

    if not _is_class(factory):
        model.calt_loss = LossSpec(
            name,
            loss_kwargs,
            function=factory,
            accepted=_accepted_names(factory),
            wanted=_wanted_labels(factory),
        )
        return

    init_kwargs = dict(loss_kwargs)
    init_names = _accepted_names(factory)
    for key, value in (("d_model", config.d_model), ("vocab_size", config.vocab_size)):
        if init_names is None or key in init_names:
            init_kwargs[key] = value
    try:
        module = factory(**init_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Loss {name!r} cannot be built with model.loss_kwargs={loss_kwargs}: "
            f"{e}. Bind arguments of a functools.partial by keyword."
        ) from e
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"Loss {name!r} is registered as a class, so it must build an "
            f"nn.Module, got {type(module).__name__}."
        )
    model.loss_module = module
    # Same initialization as the model's own layers; not _initialize_weights,
    # whose "initialized" flag would stop from_pretrained from initializing
    # loss weights missing from a checkpoint.
    module.apply(model._init_weights)
    model.calt_loss = LossSpec(
        name,
        loss_kwargs,
        accepted=_accepted_names(module.forward),
        wanted=_wanted_labels(module.forward),
    )


def compute_loss(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    hidden: torch.Tensor,
    extra_labels: Optional[dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute the configured loss of ``model`` for one batch."""
    spec: LossSpec = model.calt_loss
    extra_labels = extra_labels or {}

    if spec.is_default:
        return cross_entropy(logits, labels)
    if spec.missing:
        raise RuntimeError(
            f"model.loss_type {spec.name!r} is not registered. Call "
            f"register_loss({spec.name!r}, ...) before building or loading the "
            "model."
        )

    clash = sorted(set(extra_labels) & (RESERVED_LOSS_ARGS | set(spec.loss_kwargs)))
    if clash:
        raise ValueError(
            f"Extra labels {clash} clash with the loss arguments or "
            "model.loss_kwargs; rename them."
        )

    data = {"logits": logits, "labels": labels, "hidden": hidden, **extra_labels}
    if spec.accepted is not None:
        for name in extra_labels:
            if name not in spec.accepted and name not in spec._warned:
                spec._warned.add(name)
                logger.warning(
                    "Extra label %r is not an argument of loss %r, so the loss "
                    "does not see it.",
                    name,
                    spec.name,
                )
        data = {key: value for key, value in data.items() if key in spec.accepted}
    for name in spec.wanted - set(extra_labels) - set(spec.loss_kwargs):
        if name not in spec._warned:
            spec._warned.add(name)
            logger.warning(
                "Loss %r takes %r, but the batch has no extra label of that name "
                "(check the load preprocessor).",
                spec.name,
                name,
            )

    if spec.function is not None:
        loss = spec.function(**data, **spec.loss_kwargs)
    else:
        loss = model.loss_module(**data)

    if isinstance(loss, nn.Module):
        raise TypeError(
            f"Loss {spec.name!r} returned an nn.Module; register the class itself "
            "with register_loss instead of a function that builds it."
        )
    if not (isinstance(loss, torch.Tensor) and loss.dim() == 0):
        raise TypeError(
            f"Loss {spec.name!r} must return a scalar tensor, got "
            f"{type(loss).__name__}"
            + (
                f" of shape {tuple(loss.shape)}"
                if isinstance(loss, torch.Tensor)
                else ""
            )
            + "."
        )
    return loss
