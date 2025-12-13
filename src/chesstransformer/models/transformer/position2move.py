import torch 
from torch import nn

from chesstransformer.models.transformer.attention import GroupedQueryAttention


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_groups,
        context_length,
        dropout,
        qkv_bias,
        mask_future=False,
        apply_rope=True
    ):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_future=mask_future,
            apply_rope=apply_rope
        )
        self.feed_forward = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
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
            num_kv_groups:int=4,
            dropout:float=0.1,
            kvq_bias:bool= False,
            mask_future:bool=False,
            rope:bool=True
    ):
        super().__init__()
        # Board piece embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        if not rope:
            self.position_embedding = nn.Embedding(64, embed_dim)  # 64 squares
        self.player_embedding = nn.Embedding(2, embed_dim)  # 2 players: white and black
        
        # Additional game state embeddings
        # Castling rights: 4 binary flags (white_king, white_queen, black_king, black_queen)
        self.castling_embedding = nn.Embedding(16, embed_dim)  # 2^4 = 16 combinations
        
        # En passant: 9 values (files a-h = 0-7, none = 8)
        self.en_passant_embedding = nn.Embedding(9, embed_dim)
        
        # Halfmove clock: bucketed into ranges for 50-move rule awareness
        # Buckets: 0, 1-5, 6-10, 11-20, 21-30, 31-40, 41-50, 50+
        self.halfmove_embedding = nn.Embedding(8, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_groups=num_kv_groups,
                context_length=64 + 3,  # 64 squares + 3 game state tokens
                dropout=dropout,
                qkv_bias=kvq_bias,
                mask_future=mask_future,
                apply_rope=rope
            ) for _ in range(nb_transformer_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(embed_dim, move_vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if hasattr(self, 'position_embedding'):
            torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.player_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.castling_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.en_passant_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.halfmove_embedding.weight, mean=0.0, std=0.02)
        
        # init the q, k, v, and output projection layers using xavier initialization
        for layer in self.transformer_blocks:
            if isinstance(layer.att, GroupedQueryAttention):
                torch.nn.init.xavier_uniform_(layer.att.q_proj.weight)
                if layer.att.q_proj.bias is not None:
                    torch.nn.init.zeros_(layer.att.q_proj.bias)
                torch.nn.init.xavier_uniform_(layer.att.k_proj.weight)
                if layer.att.k_proj.bias is not None:
                    torch.nn.init.zeros_(layer.att.k_proj.bias)
                torch.nn.init.xavier_uniform_(layer.att.v_proj.weight)
                if layer.att.v_proj.bias is not None:
                    torch.nn.init.zeros_(layer.att.v_proj.bias)
                torch.nn.init.xavier_uniform_(layer.att.proj.weight)

        torch.nn.init.xavier_uniform_(self.lm_head.weight)

    def _bucket_halfmove(self, halfmove: torch.Tensor) -> torch.Tensor:
        """Convert halfmove clock values to bucket indices."""
        # Buckets: 0, 1-5, 6-10, 11-20, 21-30, 31-40, 41-50, 50+
        buckets = torch.zeros_like(halfmove)
        buckets = torch.where(halfmove == 0, 0, buckets)
        buckets = torch.where((halfmove >= 1) & (halfmove <= 5), 1, buckets)
        buckets = torch.where((halfmove >= 6) & (halfmove <= 10), 2, buckets)
        buckets = torch.where((halfmove >= 11) & (halfmove <= 20), 3, buckets)
        buckets = torch.where((halfmove >= 21) & (halfmove <= 30), 4, buckets)
        buckets = torch.where((halfmove >= 31) & (halfmove <= 40), 5, buckets)
        buckets = torch.where((halfmove >= 41) & (halfmove <= 50), 6, buckets)
        buckets = torch.where(halfmove > 50, 7, buckets)
        return buckets

    def forward(self, x, is_white, castling_rights=None, en_passant_file=None, halfmove_clock=None):
        """
        Args:
            x: Tensor of shape (batch_size, 64) containing token IDs for each square.
            is_white: Tensor of shape (batch_size,) indicating if the player to move is white (True) or black (False).
            castling_rights: Tensor of shape (batch_size,) with values 0-15 encoding 4 binary flags.
                             Bit 0: white kingside, Bit 1: white queenside,
                             Bit 2: black kingside, Bit 3: black queenside.
                             If None, defaults to all castling allowed (15).
            en_passant_file: Tensor of shape (batch_size,) with values 0-7 for files a-h, or 8 for none.
                             If None, defaults to no en passant (8).
            halfmove_clock: Tensor of shape (batch_size,) with halfmove clock values (0-100+).
                            If None, defaults to 0.
        Returns:
            logits: Tensor of shape (batch_size, move_vocab_size) containing
                    the logits for the next move prediction at each position.
        """
        batch_size, seq_len = x.size()
        assert seq_len == 64, "Input sequence length must be 64 (8x8 chess board)."
        
        device = x.device
        
        # Default values if not provided
        if castling_rights is None:
            castling_rights = torch.full((batch_size,), 15, dtype=torch.long, device=device)
        if en_passant_file is None:
            en_passant_file = torch.full((batch_size,), 8, dtype=torch.long, device=device)
        if halfmove_clock is None:
            halfmove_clock = torch.zeros(batch_size, dtype=torch.long, device=device)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(x)  # (batch_size, 64, embed_dim)
        if hasattr(self, 'position_embedding'):
            pos_emb = self.position_embedding(positions)  # (batch_size, 64, embed_dim)
        else:
            pos_emb = torch.zeros_like(token_emb)  # Fallback if RoPE is used

        player_ids = is_white.long().unsqueeze(1).expand(-1, seq_len)  # (batch_size, 64)

        player_emb = self.player_embedding(player_ids)  # (batch_size, 64, embed_dim)

        board_emb = token_emb + pos_emb + player_emb  # (batch_size, 64, embed_dim)
        
        # Create game state tokens
        castling_emb = self.castling_embedding(castling_rights.long())  # (batch_size, embed_dim)
        en_passant_emb = self.en_passant_embedding(en_passant_file.long())  # (batch_size, embed_dim)
        halfmove_bucket = self._bucket_halfmove(halfmove_clock.long())
        halfmove_emb = self.halfmove_embedding(halfmove_bucket)  # (batch_size, embed_dim)
        
        # Stack game state tokens: (batch_size, 3, embed_dim)
        game_state_tokens = torch.stack([castling_emb, en_passant_emb, halfmove_emb], dim=1)
        
        # Concatenate board and game state: (batch_size, 67, embed_dim)
        x = torch.cat([board_emb, game_state_tokens], dim=1)
        x = self.dropout(x)

        x = self.transformer_blocks(x)  # (batch_size, 67, embed_dim)
        x = self.norm(x)  # (batch_size, 67, embed_dim)
        
        # Average over all tokens (board + game state)
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        logits = self.lm_head(x)  # (batch_size, move_vocab_size)
        return logits

if __name__ == "__main__":
    # Example usage
    model = Position2MoveModel()
    batch_size = 2
    dummy_input = torch.randint(0, 13, (batch_size, 64))  # batch_size=2, seq_len=64
    is_white = torch.tensor([1, 0])  # First sample is white to move, second is black
    
    # With game state info
    castling_rights = torch.tensor([15, 3])  # First: all castling, Second: only white can castle
    en_passant_file = torch.tensor([8, 4])  # First: no EP, Second: EP on file e (4)
    halfmove_clock = torch.tensor([0, 25])  # Halfmove counters
    
    # Forward pass with all info
    logits = model(dummy_input, is_white, castling_rights, en_passant_file, halfmove_clock)
    print("Output shape:", logits.shape)  # Should be (2, move_vocab_size)
    
    # Also works without game state (backward compatible)
    logits_simple = model(dummy_input, is_white)
    print("Output shape (simple):", logits_simple.shape)