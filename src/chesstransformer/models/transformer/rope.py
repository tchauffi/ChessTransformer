"""This module implements Rotary Positional Embeddings (RoPE) as described in
the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021).
"""

import torch
import torch.nn as nn

class RoPEEmbedding(nn.Module):
    """Rotary Positional Embedding module.

    This module applies rotary positional embeddings to the input tensor.
    """

    def __init__(self, dim: int, base: int = 10_000, max_position_embeddings: int = 2048):
        """Initialize the RoPE embedding module.

        Args:
            dim (int): Dimension of the model.
            max_position_embeddings (int): Maximum number of position embeddings.
        """
        super(RoPEEmbedding, self).__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        position = torch.arange(max_position_embeddings, dtype=torch.float)
        idx_theta = torch.einsum("i , j -> i j", position, inv_freq)
        
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)

        self.register_buffer("cos_cached", idx_theta2.cos()[:, None, None, :], persistent=False)
        self.register_buffer("sin_cached", idx_theta2.sin()[:, None, None, :], persistent=False)


    def _neg_half(self, x: torch.Tensor):
        """Helper function to rotate the last dimension of the tensor by -90 degrees.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rotated tensor.
        """
        d_2 = self.dim // 2

        x1 = x[..., :d_2]
        x2 = x[..., d_2:]

        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: torch.Tensor):
        """Apply RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Tensor with RoPE applied.
        """
        seq_len = x.shape[0]
        cos = self.cos_cached[:seq_len].to(x.device)
        sin = self.sin_cached[:seq_len].to(x.device)

        x_rotated = self._neg_half(x)

        return x * cos + x_rotated * sin
