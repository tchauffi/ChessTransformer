"""This module implements attention mechanisms for LLMs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chesstransformer.models.transformer.rope import RoPEEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads, context_length, qkv_bias=False, dropout=0.0,
        mask_future=True, apply_rope=False
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
        self.apply_rope = apply_rope

        if self.apply_rope:
            self.rope = RoPEEmbedding(dim=self.d_head, max_position_embeddings=context_length)

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
        values = values.view(bz, nb_tockens, self.num_heads, self.d_head).transpose(1, 2)
        queries = queries.view(bz, nb_tockens, self.num_heads, self.d_head).transpose(1, 2)

        if self.apply_rope:
            keys = self.rope(keys)
            queries = self.rope(values)

        # Compute attention scores
        attention_scores = queries @ keys.transpose(
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
    

class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False, mask_future=True, apply_rope=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.mask_future = mask_future
        self.apply_rope = apply_rope

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.rope = None  # Placeholder for RoPE, if needed
        if self.apply_rope:
            self.rope = RoPEEmbedding(dim=self.head_dim)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        if self.apply_rope and self.rope is not None:
            keys = self.rope(keys)
            queries = self.rope(queries)

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=self.mask_future)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec
    
class GroupedQueryAttention(nn.Module):
    """
    Multi-head self-attention with grouped keys and values.
    This implementation allows multiple attention heads to share the same keys and values,
    reducing the number of parameters and computational cost.

    It works by dividing the attention heads into groups, where each group shares a set of keys and values.

    Example:
        If there are 8 attention heads and 4 key-value groups, then each group will be shared by 2 attention heads.

    Args:
        d_in: Input feature dimension
        d_out: Output feature dimension
        num_heads: Number of attention heads
        num_kv_groups: Number of key-value groups (must divide num_heads)
        dropout: Dropout probability
        qkv_bias: Whether to include bias terms in the linear projections
        mask_future: Whether to apply causal masking to prevent attention to future tokens
        apply_rope: Whether to apply Rotary Positional Embeddings (RoPE)
    Returns:
        Output tensor of shape (batch_size, seq_len, d_out)
    """
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dropout=0.0, qkv_bias=False, mask_future=True, apply_rope=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads is indivisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.q_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias)
        self.group_size = num_heads // num_kv_groups
        self.num_kv_groups = num_kv_groups

        self.proj = nn.Linear(d_out, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.mask_future = mask_future
        self.apply_rope = apply_rope
        if self.apply_rope:
            self.rope = RoPEEmbedding(dim=self.head_dim)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)


        keys = keys_new.repeat_interleave(self.group_size, dim=1)
        values = values_new.repeat_interleave(self.group_size, dim=1)

        if self.apply_rope:
            keys = self.rope(keys)
            queries = self.rope(queries)

        use_dropout = 0. if not self.training else self.dropout.p

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values, 
            attn_mask=None, 
            dropout_p=use_dropout, 
            is_causal=self.mask_future
        )
        
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)

        return context_vec
    
if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 16
    d_in = 64
    d_out = 64
    num_heads = 8
    num_kv_groups = 4

    x = torch.randn(batch_size, seq_len, d_in)

    attn = GroupedQueryAttention(d_in, d_out, num_heads, num_kv_groups, apply_rope=True)
    output = attn(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
