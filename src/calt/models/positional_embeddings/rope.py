"""
Rotary Position Embedding (RoPE) as introduced in "RoFormer: Enhanced Transformer
with Rotary Position Embedding" by Su et al.

RoPE encodes relative positional information by rotating the query and key vectors
in the complex plane. This implementation provides positional embeddings that can
be applied to input embeddings.
"""

import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    RoPE applies rotations to embeddings based on their positions. Unlike
    additive positional embeddings, RoPE is multiplicative and preserves
    the magnitude of vectors while encoding positional information.
    
    Args:
        d_model (int): The dimension of the model embeddings. Must be even.
        max_len (int): Maximum sequence length to support.
        base (float): The base for computing frequencies. Default: 10000.0.
    """
    
    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute cos and sin caches for efficiency
        self._build_cache(max_len)
    
    def _build_cache(self, max_len: int):
        """Pre-compute cos and sin values for positions up to max_len."""
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # Shape: (max_len, d_model // 2)
        
        # Concatenate cos and sin for even and odd dimensions
        emb = torch.cat([freqs, freqs], dim=-1)  # Shape: (max_len, d_model)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))  # (1, max_len, d_model)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))  # (1, max_len, d_model)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, embeddings: torch.Tensor, position_ids: torch.LongTensor | None = None) -> torch.Tensor:
        """Apply rotary positional embeddings to input embeddings.
        
        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).
            position_ids (torch.LongTensor | None): Position IDs of shape (batch_size, seq_len).
                If None, automatically generated from embeddings shape.
        
        Returns:
            torch.Tensor: Rotated embeddings with positional information, shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = embeddings.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=embeddings.device).unsqueeze(0).expand(batch_size, -1)
        
        if seq_len > self.max_len:
            # Dynamically extend cache if needed
            self._build_cache(seq_len)
        
        # Get cos and sin for the given positions
        # position_ids: (batch_size, seq_len)
        # cos_cached: (1, max_len, d_model)
        cos = self.cos_cached[:, :seq_len, :]  # (1, seq_len, d_model)
        sin = self.sin_cached[:, :seq_len, :]  # (1, seq_len, d_model)
        
        # Apply RoPE multiplicatively (rotate the embeddings)
        cos = cos.expand(batch_size, -1, -1)  # (batch_size, seq_len, d_model)
        sin = sin.expand(batch_size, -1, -1)  # (batch_size, seq_len, d_model)
        
        # Rotate embeddings: x * cos + rotate_half(x) * sin
        return embeddings * cos + self._rotate_half(embeddings) * sin
