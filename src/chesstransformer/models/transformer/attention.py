"""This module implements attention mechanisms for LLMs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads, context_length, qkv_bias=False, dropout=0.0,
        mask_future=True
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_out = d_out
        self.d_head = d_out // num_heads
        self.mask_future = mask_future
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Computes multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        bz, nb_tockens, _ = x.shape

        # Linear projections
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        # Reshape for multi-head attention
        keys = keys.view(bz, nb_tockens, self.num_heads, self.d_head).transpose(1, 2)
        values = values.view(bz, nb_tockens, self.num_heads, self.d_head).transpose(
            1, 2
        )
        querys = queries.view(bz, nb_tockens, self.num_heads, self.d_head).transpose(
            1, 2
        )
        # Compute attention scores
        attention_scores = querys @ keys.transpose(
            2, 3
        )  # (batch_size, num_heads, seq_len, seq_len)

        if self.mask_future:
            # Apply mask
            mask_bool = self.mask.bool()[:nb_tockens, :nb_tockens]
            attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)

        # Compute attention weights
        attention_weights = F.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # apply attention weights to values
        context_vec = (attention_weights @ values).transpose(1, 2).contiguous()

        # Merge heads
        context_vec = context_vec.view(bz, nb_tockens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
