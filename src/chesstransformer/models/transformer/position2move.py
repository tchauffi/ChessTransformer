import torch 
from torch import nn
import torch.nn.functional as F

from chesstransformer.models.transformer.attention import GroupedQueryAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it doesn't compute mean.
    Used in Llama, Mistral, and other modern transformers.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLUFeedForward(nn.Module):
    """SwiGLU Feed-Forward Network.
    
    Combines Swish activation with Gated Linear Units.
    Better performance than standard GELU FFN.
    Used in Llama 2, PaLM, and other modern transformers.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        # Default hidden_dim is ~2.67x embed_dim (compensates for extra params from gate)
        hidden_dim = hidden_dim or int(embed_dim * 8 / 3)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)  # Up projection
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)  # Down projection
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)  # Gate projection
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# Keep old FeedForward for backward compatibility
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
        apply_rope=True,
        use_swiglu=True,
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
        # Use SwiGLU by default, fallback to standard FFN
        if use_swiglu:
            self.feed_forward = SwiGLUFeedForward(embed_dim, dropout=dropout)
        else:
            self.feed_forward = FeedForward(embed_dim)
        
        # Use RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
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
            rope:bool=True,
            use_swiglu:bool=True,
            use_col_row_emb:bool=False,
            use_value_head:bool=False,
    ):
        super().__init__()
        self.use_value_head = use_value_head

        # Learnable CLS token for classification (instead of mean pooling)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Board piece embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        if not rope:
            self.position_embedding = nn.Embedding(64, embed_dim)  # 64 squares
        self.player_embedding = nn.Embedding(2, embed_dim)  # 2 players: white and black

        self.use_col_row_emb = use_col_row_emb

        if use_col_row_emb:
            # Optional column and row embeddings for board squares
            self.col_embedding = nn.Embedding(8, embed_dim)
            self.row_embedding = nn.Embedding(8, embed_dim)
        
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
                context_length=64 + 3 + 1,  # 64 squares + 3 game state tokens + 1 CLS
                dropout=dropout,
                qkv_bias=kvq_bias,
                mask_future=mask_future,
                apply_rope=rope,
                use_swiglu=use_swiglu,
            ) for _ in range(nb_transformer_layers)]
        )
        # Final RMSNorm
        self.norm = RMSNorm(embed_dim)

        # Policy head (move prediction)
        self.lm_head = nn.Linear(embed_dim, move_vocab_size, bias=False)

        # Value head (position evaluation: outputs scalar in [-1, 1])
        # Predicts expected outcome from the current player's perspective:
        #   +1 = current player wins, 0 = draw, -1 = current player loses
        if use_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
                nn.Tanh(),
            )

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if hasattr(self, 'position_embedding'):
            torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.use_col_row_emb:
            torch.nn.init.normal_(self.col_embedding.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.row_embedding.weight, mean=0.0, std=0.02)

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
            
            # Initialize SwiGLU layers
            if isinstance(layer.feed_forward, SwiGLUFeedForward):
                torch.nn.init.xavier_uniform_(layer.feed_forward.w1.weight)
                torch.nn.init.xavier_uniform_(layer.feed_forward.w2.weight)
                torch.nn.init.xavier_uniform_(layer.feed_forward.w3.weight)

        torch.nn.init.xavier_uniform_(self.lm_head.weight)

        # Initialize value head if present
        if self.use_value_head:
            for module in self.value_head:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

    def _bucket_halfmove(self, halfmove: torch.Tensor) -> torch.Tensor:
        """Convert halfmove clock values to bucket indices.
        
        Buckets: 0, 1-5, 6-10, 11-20, 21-30, 31-40, 41-50, 50+
        Uses searchsorted to avoid torch.where (better for MPS compilation).
        """
        # Define bucket boundaries
        boundaries = torch.tensor([1, 6, 11, 21, 31, 41, 51], device=halfmove.device)
        # searchsorted gives us the bucket index directly
        # Values less than 1 -> bucket 0
        # Values in [1, 6) -> bucket 1
        # Values in [6, 11) -> bucket 2, etc.
        buckets = torch.searchsorted(boundaries, halfmove.float(), right=False)
        buckets = torch.clamp(buckets, 0, 7)
        return buckets.long()

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
            If use_value_head is False:
                logits: Tensor of shape (batch_size, move_vocab_size)
            If use_value_head is True:
                tuple of:
                    logits: Tensor of shape (batch_size, move_vocab_size)
                    value: Tensor of shape (batch_size,) in range [-1, 1]
                           representing expected outcome from current player's perspective.
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

        if self.use_col_row_emb:
            cols = (positions % 8).long()  # (batch_size, 64)
            rows = (positions // 8).long()  # (batch_size, 64)
            col_emb = self.col_embedding(cols)  # (batch_size, 64, embed_dim)
            row_emb = self.row_embedding(rows)  # (batch_size, 64, embed_dim)
            board_emb = board_emb + col_emb + row_emb  # (batch_size, 64, embed_dim)
        
        # Create game state tokens
        castling_emb = self.castling_embedding(castling_rights.long())  # (batch_size, embed_dim)
        en_passant_emb = self.en_passant_embedding(en_passant_file.long())  # (batch_size, embed_dim)
        halfmove_bucket = self._bucket_halfmove(halfmove_clock.long())
        halfmove_emb = self.halfmove_embedding(halfmove_bucket)  # (batch_size, embed_dim)
        
        # Stack game state tokens: (batch_size, 3, embed_dim)
        game_state_tokens = torch.stack([castling_emb, en_passant_emb, halfmove_emb], dim=1)
        
        # Expand CLS token for batch: (batch_size, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate: [CLS] + board + game_state: (batch_size, 68, embed_dim)
        x = torch.cat([cls_tokens, board_emb, game_state_tokens], dim=1)
        x = self.dropout(x)

        x = self.transformer_blocks(x)  # (batch_size, 68, embed_dim)
        x = self.norm(x)  # (batch_size, 68, embed_dim)
        
        # Use CLS token output for classification (instead of mean pooling)
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)

        logits = self.lm_head(cls_output)  # (batch_size, move_vocab_size)

        if self.use_value_head:
            value = self.value_head(cls_output).squeeze(-1)  # (batch_size,)
            return logits, value

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
    
    # Forward pass with all info (policy only)
    logits = model(dummy_input, is_white, castling_rights, en_passant_file, halfmove_clock)
    print("Output shape:", logits.shape)  # Should be (2, move_vocab_size)
    
    # Also works without game state (backward compatible)
    logits_simple = model(dummy_input, is_white)
    print("Output shape (simple):", logits_simple.shape)
    
    # With value head
    model_vh = Position2MoveModel(use_value_head=True)
    logits, value = model_vh(dummy_input, is_white, castling_rights, en_passant_file, halfmove_clock)
    print(f"Policy shape: {logits.shape}, Value shape: {value.shape}")  # (2, move_vocab_size), (2,)
    print(f"Value predictions: {value}")  # Should be in [-1, 1]