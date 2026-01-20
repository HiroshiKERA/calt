"""
Learnable positional embeddings using nn.Embedding.
These embeddings are trained along with the model.
"""

import torch
import torch.nn as nn


class GenericPositionalEmbedding(nn.Module):
    """Trainable positional embeddings using nn.Embedding.
    
    These embeddings are trainable parameters that are learned during training.
    
    Args:
        d_model (int): The dimension of the model embeddings.
        max_len (int): Maximum sequence length to support.
    """
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable embedding layer
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, embeddings: torch.Tensor, position_ids: torch.LongTensor | None = None) -> torch.Tensor:
        """Apply learnable positional embeddings to input embeddings.
        
        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).
            position_ids (torch.LongTensor | None): Position IDs of shape (batch_size, seq_len).
                If None, automatically generated from embeddings shape.
            
        Returns:
            torch.Tensor: Embeddings with positional information added, shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = embeddings.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=embeddings.device).unsqueeze(0).expand(batch_size, -1)
        
        # Ensure we don't exceed max_len
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}. "
                f"Increase max_len when initializing LearnedPositionalEmbedding."
            )
        
        # Clamp position_ids to valid range
        position_ids = torch.clamp(position_ids, 0, self.max_len - 1)
        
        # Get positional embeddings: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        pe = self.embedding(position_ids)
        
        # Add positional embeddings to input embeddings
        return embeddings + pe
