"""Utility functions for models."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Implements Layer Normalization"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """
        Forward pass of layer normalization
        Args:
            x: Input tensor of shape (batch_size, seq_length, dim)
        Returns:
            out: Layer-normalized tensor of the same shape as input
        """

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        return out


class GELU(nn.Module):
    """Implements the GELU activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of GELU activation
        Args:
            x: Input tensor
        Returns:
            out: Tensor after applying GELU activation
        """
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
