import torch 
from torch import nn

from chesstransformer.models.transformer.attention import MultiHeadAttention
from chesstransformer.models.transformer.utils import LayerNorm, GELU


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
        mask_future=False,
        apply_rope=True
    ):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_future=mask_future,
            apply_rope=apply_rope
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
            mask_future:bool=False,
            rope:bool=True
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        if not rope:
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
                mask_future=mask_future,
                apply_rope=rope
            ) for _ in range(nb_transformer_layers)]
        )
        self.norm = LayerNorm(embed_dim)

        self.lm_head = nn.Linear(embed_dim, move_vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if hasattr(self, 'position_embedding'):
            torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.player_embedding.weight, mean=0.0, std=0.02)
        
        # init the q, k, v, and output projection layers using xavier initialization
        for layer in self.transformer_blocks:
            if isinstance(layer.att, MultiHeadAttention):
                torch.nn.init.xavier_uniform_(layer.att.W_query.weight)
                torch.nn.init.xavier_uniform_(layer.att.W_key.weight)
                torch.nn.init.xavier_uniform_(layer.att.W_value.weight)
                torch.nn.init.xavier_uniform_(layer.att.out_proj.weight)

        torch.nn.init.xavier_uniform_(self.lm_head.weight)
        

    def forward(self, x, is_white):
        """
        Args:
            x: Tensor of shape (batch_size, 64) containing token IDs for each square.
            is_white: Tensor of shape (batch_size,) indicating if the player to move is white (True) or black (False).
        Returns:
            logits: Tensor of shape (batch_size, move_vocab_size) containing
                    the logits for the next move prediction at each position.
        """
        batch_size, seq_len = x.size()
        assert seq_len == 64, "Input sequence length must be 64 (8x8 chess board)."

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(x)  # (batch_size, 64, embed_dim)
        if hasattr(self, 'position_embedding'):
            pos_emb = self.position_embedding(positions)  # (batch_size, 64, embed_dim)
        else:
            pos_emb = torch.zeros_like(token_emb)  # Fallback if RoPE is used

        player_ids = is_white.long().unsqueeze(1).expand(-1, seq_len)  # (batch_size, 64)

        player_emb = self.player_embedding(player_ids)  # (batch_size, 64, embed_dim)

        x = token_emb + pos_emb + player_emb  # (batch_size, 64, embed_dim)
        x = self.dropout(x)

        x = self.transformer_blocks(x)  # (batch_size, 64, embed_dim)
        x = self.norm(x)  # (batch_size, 64, embed_dim)
        
        # average over the sequence length dimension
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        logits = self.lm_head(x)  # (batch_size, move_vocab_size)
        return logits

if __name__ == "__main__":
    # Example usage
    model = Position2MoveModel()
    dummy_input = torch.randint(0, 13, (2, 64))  # batch_size=2, seq_len=64
    is_white = torch.tensor([1, 0])  # First sample is white to move, second is black
    logits = model(dummy_input, is_white)
    print(logits.shape)  # Should be (2, move_vocab_size)