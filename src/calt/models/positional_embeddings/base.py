"""
Factory + registry for creating positional embedding instances.

The built-in strategies ("generic"/"learned", "sinusoidal", "rope", "none") are
registered below. Users can plug in their own positional embedding and select it
by config (``model.use_positional_embedding``) — symmetric to input embeddings
and models:

    from calt.models import register_positional_embedding

    register_positional_embedding(
        "my_pe", lambda d_model, max_len, **kw: MyPositional(d_model, max_len)
    )

then set ``model.use_positional_embedding: my_pe`` in train.yaml. A factory
receives at least ``d_model`` and ``max_len`` and returns an ``nn.Module``
(or ``None`` for "no positional embedding") applied to the token embeddings,
mapping ``(batch, seq, d_model) -> (batch, seq, d_model)``.
"""

from typing import Callable, Optional

import torch.nn as nn

# name (lowercased) -> factory(d_model, max_len, **kwargs) -> nn.Module | None
POSITIONAL_EMBEDDING_REGISTRY: dict[str, Callable[..., Optional[nn.Module]]] = {}


def register_positional_embedding(
    name: str, factory: Callable[..., Optional[nn.Module]]
) -> None:
    """Register (or override) a positional-embedding factory under ``name``."""
    POSITIONAL_EMBEDDING_REGISTRY[name.lower()] = factory


def get_positional_embedding(
    pe_type: str,
    d_model: int,
    max_len: int = 512,
    **kwargs,
) -> Optional[nn.Module]:
    """Create a positional embedding instance based on the specified type.

    Args:
        pe_type (str): Type of positional embedding. Built-in types:
            - "generic"/"learned": Learnable embeddings using nn.Embedding
            - "sinusoidal": Fixed sin/cos embeddings (original Transformer)
            - "rope": Rotary Position Embedding (RoFormer)
            - "none": No positional embedding (returns None)
            Plus any type registered via ``register_positional_embedding``.
        d_model (int): The dimension of the model embeddings.
        max_len (int): Maximum sequence length to support.
        **kwargs: Additional arguments passed to the factory (e.g. RoPE ``base``).

    Returns:
        nn.Module | None: Positional embedding module, or None for "none".

    Raises:
        ValueError: If pe_type is not supported.
    """
    key = pe_type.lower() if isinstance(pe_type, str) else pe_type
    if key not in POSITIONAL_EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unsupported positional embedding type: {pe_type}. "
            f"Supported types: {sorted(POSITIONAL_EMBEDDING_REGISTRY)}"
        )
    return POSITIONAL_EMBEDDING_REGISTRY[key](
        d_model=d_model, max_len=max_len, **kwargs
    )


# --------------------------------------------------------------------------- #
# Built-in factories (registered at import).                                   #
# --------------------------------------------------------------------------- #


def _make_generic(d_model, max_len, **kwargs):
    from .generic import GenericPositionalEmbedding

    return GenericPositionalEmbedding(d_model=d_model, max_len=max_len)


def _make_sinusoidal(d_model, max_len, **kwargs):
    from .sinusoidal import SinusoidalPositionalEmbedding

    return SinusoidalPositionalEmbedding(d_model=d_model, max_len=max_len)


def _make_rope(d_model, max_len, base=10000.0, **kwargs):
    from .rope import RotaryPositionalEmbedding

    return RotaryPositionalEmbedding(d_model=d_model, max_len=max_len, base=base)


def _make_none(**kwargs):
    return None


register_positional_embedding("generic", _make_generic)
register_positional_embedding("learned", _make_generic)  # backward-compat alias
register_positional_embedding("sinusoidal", _make_sinusoidal)
register_positional_embedding("rope", _make_rope)
register_positional_embedding("none", _make_none)
