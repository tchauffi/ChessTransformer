"""Legacy Position2Move bot for V2 model compatibility.

The V2 model was trained with an older architecture that uses:
- MultiHeadAttention instead of GroupedQueryAttention
- LayerNorm with gamma/beta instead of RMSNorm with weight
- Standard GELU FeedForward instead of SwiGLU
- No CLS token, castling, en passant, or halfmove embeddings
"""

from pathlib import Path
import json
import torch
import torch.nn as nn
import chess
from safetensors import safe_open
from chesstransformer.models.transformer.attention import MultiHeadAttention
from chesstransformer.models.tokenizer import PostionTokenizer, MoveTokenizer


class LayerNorm(nn.Module):
    """Legacy LayerNorm with gamma/beta naming."""
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-6

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class FeedForward(nn.Module):
    """Legacy FeedForward network."""
    def __init__(self, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)


class LegacyTransformerBlock(nn.Module):
    """Legacy transformer block for V2 model."""
    def __init__(
        self,
        embed_dim,
        num_heads,
        context_length,
        dropout,
        qkv_bias,
        mask_future=False,
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


class LegacyPosition2MoveModel(nn.Module):
    """Legacy Position2Move model architecture for V2 compatibility."""
    
    def __init__(
        self,
        vocab_size: int = 13,
        move_vocab_size: int = 1968,
        embed_dim: int = 512,
        nb_transformer_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        kvq_bias: bool = False,
        mask_future: bool = False,
    ):
        super().__init__()
        
        # Board piece embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(64, embed_dim)  # 64 squares
        self.player_embedding = nn.Embedding(2, embed_dim)  # 2 players
        
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(
            *[LegacyTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_length=64,
                dropout=dropout,
                qkv_bias=kvq_bias,
                mask_future=mask_future,
            ) for _ in range(nb_transformer_layers)]
        )
        
        self.norm = LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, move_vocab_size, bias=False)

    def forward(self, x, is_white):
        batch_size, seq_len = x.shape
        
        # Get embeddings
        token_emb = self.token_embedding(x)
        pos_indices = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(pos_indices)
        player_emb = self.player_embedding(is_white.long())
        
        # Combine embeddings
        x = token_emb + pos_emb.unsqueeze(0) + player_emb.unsqueeze(1)
        x = self.dropout(x)
        
        # Transformer blocks
        x = self.transformer_blocks(x)
        x = self.norm(x)
        
        # Mean pooling
        x = x.mean(dim=1)
        
        # Output
        logits = self.lm_head(x)
        return logits


data_folder = Path(__file__).parents[1] / "data"


class LegacyPosition2MoveBot:
    """Bot using the legacy V2 model architecture."""
    
    def __init__(
        self, 
        model_path: str = str((data_folder / "models/V2_model/model.safetensors").resolve()), 
        device: str = "cpu"
    ):
        self.device = device
        self.position_tokenizer = PostionTokenizer()
        self.move_tokenizer = MoveTokenizer()
        
        config_path = Path(model_path).parent / "config.json"

        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load legacy model
        self.model = LegacyPosition2MoveModel(**self.config).to(device)
        
        with safe_open(model_path, framework="pt", device=device) as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
            self.model.load_state_dict(state_dict)

        self.model.eval()

    @torch.no_grad()
    def predict(self, board: chess.Board):
        tokens_ids = self.position_tokenizer.encode(board)
        torch_input = torch.tensor(tokens_ids).unsqueeze(0).long().to(self.device)

        is_white = board.turn
        is_white = torch.tensor([is_white]).bool().to(self.device)

        logits = self.model(torch_input, is_white)

        legal_moves = [m.uci() for m in board.legal_moves]

        mask = torch.full((logits.size(-1),), float('-inf')).to(self.device)
        for move in legal_moves:
            move_id = self.move_tokenizer.encode(move)
            mask[move_id] = 0.0

        masked_logits = logits[0] + mask
        probs = torch.softmax(masked_logits, dim=-1)
        
        # Sample from distribution with temperature
        predicted_move_id = torch.multinomial(probs * 0.2, num_samples=1).item()
        proba = probs[predicted_move_id].item()

        predicted_move = self.move_tokenizer.decode(predicted_move_id)
        return predicted_move, proba


if __name__ == "__main__":
    bot = LegacyPosition2MoveBot()

    board = chess.Board()
    print("Initial board:")
    print(board)

    move, proba = bot.predict(board)
    print(f"Bot suggests move: {move} with probability {proba:.4f}")
