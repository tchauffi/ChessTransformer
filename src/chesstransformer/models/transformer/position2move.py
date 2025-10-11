import torch 
from torch import nn

from ..transformer.attention import MultiHeadAttention
from ..transformer.utils import LayerNorm, GELU


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        dropout,
        qkv_bias,
        mask_future=False
    ):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_future=mask_future
        )
        self.feed_forward = FeedForward(embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout_shortcut = nn.Dropout(dropout)

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

class Position2MoveModel(nn.Module):
    def __init__(
            self, 
            vocab_size:int=13,
            move_vocab_size:int=1856,
            embed_dim:int=512,
            nb_transformer_layers:int=8,
            num_heads:int=8,
            dropout:float=0.1,
            kvq_bias:bool= False,
            mask_future:bool=False
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(64, embed_dim)  # 64 squares
        self.player_embedding = nn.Embedding(2, embed_dim)  # 2 players: white and black
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_length=64,
                dropout=dropout,
                qkv_bias=kvq_bias,
                mask_future=mask_future
            ) for _ in range(nb_transformer_layers)]
        )
        self.norm = LayerNorm(embed_dim)

        self.lm_head = nn.Linear(embed_dim, move_vocab_size, bias=False)

    def forward(self, x, is_white):
        """
        Args:
            x: Tensor of shape (batch_size, 64) containing token IDs for each square.
        Returns:
            logits: Tensor of shape (batch_size, move_vocab_size) containing
                    the logits for the next move prediction at each position.
        """
        batch_size, seq_len = x.size()
        assert seq_len == 64, "Input sequence length must be 64 (8x8 chess board)."

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(x)  # (batch_size, 64, embed_dim)
        pos_emb = self.position_embedding(positions)  # (batch_size, 64, embed_dim)

        if is_white:
            player_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=x.device)
        else:
            player_ids = torch.ones((batch_size, seq_len), dtype=torch.long, device=x.device)

        player_emb = self.player_embedding(player_ids)  # (batch_size, 64, embed_dim)

        x = token_emb + pos_emb + player_emb# (batch_size, 64, embed_dim)
        x = self.dropout(x)

        x = self.transformer_blocks(x)  # (batch_size, 64, embed_dim)
        x = self.norm(x)  # (batch_size, 64, embed_dim)
        
        x = x[:, -1, :]  # Take the representation of the last position

        logits = self.lm_head(x)  # (batch_size, move_vocab_size)
        return logits

