"""
Factory + registry for **input (token) embeddings**.

The model's input embedding used to be a hard-coded ``nn.Embedding``. This module
turns it into a pluggable choice selected by config (``model.input_embedding_type``)
— symmetric to how positional embeddings and models are chosen — so a user can
inject their own input embedding (e.g. frozen/pretrained tables, factorized or
hashed embeddings) without editing the library.

Built-in
--------
``"token"`` (aliases ``"default"``, ``"learned"``): a plain ``nn.Embedding``;
this is the historical behavior, so existing configs are unaffected.

Registering a custom input embedding
------------------------------------
Call this *before* building the model (e.g. at the top of your train script):

    import torch.nn as nn
    from calt.models import register_input_embedding

    class MyEmbedding(nn.Module):
        def __init__(self, vocab_size, d_model):
            super().__init__()
            ...
        def forward(self, input_ids):      # (B, S) long  ->  (B, S, d_model)
            ...

    register_input_embedding(
        "my_emb", lambda vocab_size, d_model, **kw: MyEmbedding(vocab_size, d_model)
    )

then set ``model.input_embedding_type: my_emb`` in train.yaml.

Contract
--------
A factory receives at least ``vocab_size`` and ``d_model`` as keyword arguments
(extra config keys are forwarded as kwargs) and returns an ``nn.Module`` mapping
``input_ids`` of shape ``(batch, seq)`` to embeddings of shape
``(batch, seq, d_model)``.
"""

from typing import Callable

import torch.nn as nn

# name (lowercased) -> factory(vocab_size, d_model, **kwargs) -> nn.Module
INPUT_EMBEDDING_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_input_embedding(name: str, factory: Callable[..., nn.Module]) -> None:
    """Register (or override) an input-embedding factory under ``name``."""
    INPUT_EMBEDDING_REGISTRY[name.lower()] = factory


def get_input_embedding(
    emb_type: str, vocab_size: int, d_model: int, **kwargs
) -> nn.Module:
    """Build the input embedding registered under ``emb_type``.

    Raises ValueError with the list of supported types if ``emb_type`` is unknown.
    """
    key = emb_type.lower() if isinstance(emb_type, str) else emb_type
    if key not in INPUT_EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unsupported input embedding type: {emb_type!r}. "
            f"Supported types: {sorted(INPUT_EMBEDDING_REGISTRY)}"
        )
    return INPUT_EMBEDDING_REGISTRY[key](
        vocab_size=vocab_size, d_model=d_model, **kwargs
    )


def _token_embedding(vocab_size: int, d_model: int, **kwargs) -> nn.Embedding:
    """Default: a plain learnable token-embedding table (historical behavior)."""
    return nn.Embedding(vocab_size, d_model)


# Built-ins (registered at import). "token" is the default; the aliases make
# common config values resolve to the same thing.
register_input_embedding("token", _token_embedding)
register_input_embedding("default", _token_embedding)
register_input_embedding("learned", _token_embedding)
