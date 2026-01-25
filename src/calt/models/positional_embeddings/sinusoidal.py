"""
Sinusoidal positional embeddings as used in the original Transformer paper.
Fixed, non-learnable positional embeddings using sin/cos functions.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings from "Attention Is All You Need".

    Creates fixed positional embeddings using sin and cos functions.
    These embeddings are not learnable and are computed on-the-fly.

    Args:
        d_model (int): The dimension of the model embeddings.
        max_len (int): Maximum sequence length to support.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Pre-compute the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, so it won't be trained)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(
        self, embeddings: torch.Tensor, position_ids: torch.LongTensor | None = None
    ) -> torch.Tensor:
        """Apply sinusoidal positional embeddings to input embeddings.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).
            position_ids (torch.LongTensor | None): Position IDs of shape (batch_size, seq_len).
                If None, automatically generated from embeddings shape.

        Returns:
            torch.Tensor: Embeddings with positional information added, shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = embeddings.shape

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=embeddings.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Ensure we don't exceed max_len
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}. "
                f"Increase max_len when initializing SinusoidalPositionalEmbedding."
            )

        # Select the positional encodings for the given positions
        # position_ids: (batch_size, seq_len)
        # self.pe: (1, max_len, d_model)
        # pe_selected: (batch_size, seq_len, d_model)
        pe_selected = self.pe[:, :seq_len, :].expand(batch_size, -1, -1)

        # Add positional embeddings to input embeddings
        return embeddings + pe_selected
