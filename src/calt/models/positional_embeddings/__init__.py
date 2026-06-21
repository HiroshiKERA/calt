"""
Positional embedding modules for transformer models.

This module provides various positional embedding strategies:
- Sinusoidal: Fixed sin/cos embeddings (original Transformer)
- Learned/Generic: Trainable embeddings using nn.Embedding
- RoPE: Rotary Position Embedding (RoFormer)
"""

from .base import (
    POSITIONAL_EMBEDDING_REGISTRY,
    get_positional_embedding,
    register_positional_embedding,
)
from .generic import GenericPositionalEmbedding
from .rope import RotaryPositionalEmbedding
from .sinusoidal import SinusoidalPositionalEmbedding

# Alias for backward compatibility
LearnedPositionalEmbedding = GenericPositionalEmbedding

__all__ = [
    "SinusoidalPositionalEmbedding",
    "GenericPositionalEmbedding",
    "LearnedPositionalEmbedding",
    "RotaryPositionalEmbedding",
    "get_positional_embedding",
    "register_positional_embedding",
    "POSITIONAL_EMBEDDING_REGISTRY",
]
