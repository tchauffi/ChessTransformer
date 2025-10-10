"""This module implements GPT model architecture."""

import torch
import torch.nn as nn
from ..transformer.attention import MultiHeadAttention
from ..transformer.utils import LayerNorm, GELU


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            context_length=cfg["context_size"],
            dropout=cfg["dropout"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.feed_forward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embed_dim"])
        self.norm2 = LayerNorm(cfg["embed_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        return x


class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.position_embedding = nn.Embedding(cfg["context_size"], cfg["embed_dim"])
        self.dropout_embed = nn.Dropout(cfg["dropout"])
        self.transfomer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        self.final_norm = LayerNorm(cfg["embed_dim"])
        self.lm_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        bz, sq_len = x.size()

        token_emb = self.token_embedding(x)  # (bz, sq_len, embed_dim)

        positions = torch.arange(sq_len, device=x.device)
        pos_emb = self.position_embedding(positions)  # (sq_len, embed_dim)

        x = token_emb + pos_emb  # (bz, sq_len, embed_dim)
        x = self.dropout_embed(x)

        x = self.transfomer_blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)  # (bz, sq_len, vocab_size)
        return logits
