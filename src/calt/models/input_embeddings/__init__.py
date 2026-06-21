"""Input (token) embedding factory and registry."""

from .base import (
    INPUT_EMBEDDING_REGISTRY,
    get_input_embedding,
    register_input_embedding,
)

__all__ = [
    "INPUT_EMBEDDING_REGISTRY",
    "get_input_embedding",
    "register_input_embedding",
]
