"""This module implements Rotary Positional Embeddings (RoPE) as described in
the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021).
"""

import torch
import torch.nn as nn

class RoPEEmbedding(nn.Module):
    """Rotary Positional Embedding module.

    This module applies rotary positional embeddings to the input tensor.
    Works with input shape (batch, num_heads, seq_len, head_dim).
    """

    def __init__(self, dim: int, base: int = 10_000, max_position_embeddings: int = 2048):
        """Initialize the RoPE embedding module.

        Args:
            dim (int): Dimension of each attention head (head_dim).
            base (int): Base for the frequency computation. Lower values = faster rotation.
                        Default 10,000 works well for long sequences. For short sequences
                        like chess (67 tokens), consider lower values like 1,000.
            max_position_embeddings (int): Maximum number of position embeddings.
        """
        super(RoPEEmbedding, self).__init__()
        self.dim = dim
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Compute position encodings for all positions
        position = torch.arange(max_position_embeddings, dtype=torch.float)
        # (max_pos, dim/2)
        idx_theta = torch.einsum("i, j -> i j", position, inv_freq)
        
        # Duplicate for both halves: (max_pos, dim)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)

        # Shape: (1, 1, max_pos, dim) for broadcasting with (batch, heads, seq, dim)
        self.register_buffer("cos_cached", idx_theta2.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", idx_theta2.sin()[None, None, :, :], persistent=False)

    def _neg_half(self, x: torch.Tensor):
        """Rotate tensor by swapping and negating halves.
        
        For RoPE, we need to compute: [-x2, x1] where x = [x1, x2]
        This implements the rotation matrix multiplication efficiently.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

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
            x (torch.Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input.
        """
        # x shape: (batch, num_heads, seq_len, head_dim)
        seq_len = x.shape[2]  # Get seq_len from the correct dimension
        
        # Slice cached values to sequence length: (1, 1, seq_len, dim)
        cos = self.cos_cached[:, :, :seq_len, :].to(x.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(x.device)

        # Apply rotary embedding: x * cos + rotate(x) * sin
        x_rotated = self._neg_half(x)
        return x * cos + x_rotated * sin


if __name__ == "__main__":
    # Test with chess-like dimensions
    batch_size = 2
    num_heads = 8
    seq_len = 67  # 64 squares + 3 game state tokens
    head_dim = 64
    
    rope = RoPEEmbedding(dim=head_dim, base=10_000, max_position_embeddings=128)
    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    output = rope(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify positions are encoded differently
    # Two different positions should produce different outputs
    x_same = torch.ones(1, 1, 10, head_dim)
    out = rope(x_same)
    print(f"\nPosition encoding test (should be different):")
    print(f"  Position 0 norm: {out[0, 0, 0].norm():.4f}")
    print(f"  Position 5 norm: {out[0, 0, 5].norm():.4f}")
    print(f"  Position 0 vs 5 cosine sim: {torch.cosine_similarity(out[0, 0, 0], out[0, 0, 5], dim=0):.4f}")
