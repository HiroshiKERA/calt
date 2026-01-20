"""
Factory function for creating positional embedding instances.

This module provides a function for creating different types of positional embeddings.
"""

from typing import Optional

import torch.nn as nn


def get_positional_embedding(
    pe_type: str,
    d_model: int,
    max_len: int = 512,
    **kwargs,
) -> Optional[nn.Module]:
    """Create a positional embedding instance based on the specified type.
    
    Args:
        pe_type (str): Type of positional embedding. Supported types:
            - "learned" or "generic": Learnable embeddings using nn.Embedding
            - "sinusoidal": Fixed sin/cos embeddings (original Transformer)
            - "rope": Rotary Position Embedding (RoFormer)
            - "none": No positional embedding (returns None)
        d_model (int): The dimension of the model embeddings.
        max_len (int): Maximum sequence length to support.
        **kwargs: Additional arguments passed to the positional embedding constructor.
            For RoPE, supports:
            - base (float): Base for computing frequencies. Default: 10000.0.
    
    Returns:
        nn.Module | None: Positional embedding module, or None if pe_type is "none".
    
    Raises:
        ValueError: If pe_type is not supported.
    
    Examples:
        >>> # Create a learned positional embedding
        >>> pe = get_positional_embedding("learned", d_model=128, max_len=512)
        >>> 
        >>> # Create a sinusoidal positional embedding
        >>> pe = get_positional_embedding("sinusoidal", d_model=128, max_len=512)
        >>> 
        >>> # Create a RoPE positional embedding with custom base
        >>> pe = get_positional_embedding("rope", d_model=128, max_len=512, base=10000.0)
        >>> 
        >>> # No positional embedding
        >>> pe = get_positional_embedding("none", d_model=128, max_len=512)
        >>> assert pe is None
    """
    # Import here to avoid circular import
    from .sinusoidal import SinusoidalPositionalEmbedding
    from .generic import GenericPositionalEmbedding
    from .rope import RoPEPositionalEmbedding
    
    pe_type = pe_type.lower() if isinstance(pe_type, str) else pe_type
    
    if pe_type in ["learned", "generic"]:
        return GenericPositionalEmbedding(
            d_model=d_model,
            max_len=max_len,
        )
    elif pe_type == "sinusoidal":
        return SinusoidalPositionalEmbedding(
            d_model=d_model,
            max_len=max_len,
        )
    elif pe_type == "rope":
        base = kwargs.get("base", 10000.0)
        return RoPEPositionalEmbedding(
            d_model=d_model,
            max_len=max_len,
            base=base,
        )
    elif pe_type == "none":
        return None
    else:
        supported_types = ["learned", "generic", "sinusoidal", "rope", "none"]
        raise ValueError(
            f"Unsupported positional embedding type: {pe_type}. "
            f"Supported types: {supported_types}"
        )
