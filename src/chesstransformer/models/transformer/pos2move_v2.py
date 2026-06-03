import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTION_PLANES = 73
BOARD_SQUARES = 64
EXTRA_TOKENS = 3  # castling, en_passant, player
CONTEXT_SIZE = BOARD_SQUARES + EXTRA_TOKENS  # 67


def build_chess_relation_index(context_size: int = CONTEXT_SIZE) -> torch.Tensor:
    """Precompute (T, T) relation index between every pair of tokens.

    Categories (8 total):
      0: same square / same token
      1: same file
      2: same rank
      3: same diagonal
      4: knight-reachable
      5: king-adjacent (Chebyshev <= 1)
      6: nearby (Manhattan <= 3)
      7: far
      Game-state tokens (>= 64) get category 0 toward themselves and
      a dedicated "global" category 7 toward all board squares.
    """
    rel = torch.zeros(context_size, context_size, dtype=torch.long)
    for i in range(context_size):
        for j in range(context_size):
            if i == j:
                rel[i, j] = 0
                continue
            if i >= BOARD_SQUARES or j >= BOARD_SQUARES:
                # Any pair involving a game-state token: "global" bucket
                rel[i, j] = 7
                continue
            fi, ri = i % 8, i // 8
            fj, rj = j % 8, j // 8
            df, dr = abs(fi - fj), abs(ri - rj)
            if df == 0:
                rel[i, j] = 1
            elif dr == 0:
                rel[i, j] = 2
            elif df == dr:
                rel[i, j] = 3
            elif {df, dr} == {1, 2}:
                rel[i, j] = 4
            elif df <= 1 and dr <= 1:
                rel[i, j] = 5
            elif df + dr <= 3:
                rel[i, j] = 6
            else:
                rel[i, j] = 7
    return rel


_RELATION_INDEX: torch.Tensor = build_chess_relation_index()


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class ChessGroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads, num_kv_groups, context_size, dropout=0.0, qkv_bias=False
    ):
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

        self.proj = nn.Linear(d_out, d_out, bias=False)
        self.dropout = dropout

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # Chess-geometry relative bias: 8 relation categories per head
        # Massively fewer params than fully-learned (H, T, T) bias.
        self.num_relations = 8
        self.bias_table = nn.Parameter(torch.zeros(num_heads, self.num_relations))
        self.register_buffer("rel_idx", _RELATION_INDEX, persistent=False)


    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        queries = self.q_norm(queries)
        keys_new = self.k_norm(keys_new)

        keys = keys_new.repeat_interleave(self.group_size, dim=1)
        values = values_new.repeat_interleave(self.group_size, dim=1)

        # Build additive chess-geometry bias from the compact (H, 8) table.
        # Shape: (1, H, T, T) so SDPA broadcasts it across the batch.
        attn_bias = self.bias_table[:, self.rel_idx].unsqueeze(0)

        context_vec = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)

        return context_vec
    
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = ReluSquared()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class ChessTransformerLayer(nn.Module):
    def __init__(self, dim, nb_heads, qkv_bias, context_size, dropout=0.1, layer_drop=0.0):
        super().__init__()
        self.attn = ChessGroupedQueryAttention(
            d_in=dim,
            d_out=dim,
            num_heads=nb_heads,
            num_kv_groups=nb_heads // 2,
            context_size=context_size,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.mlp = MLP(dim, dim * 4, dim, dropout=dropout)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.layer_drop = layer_drop

    def forward(self, x):
        # Stochastic depth: with prob `layer_drop`, skip the entire layer.
        # Applied independently to attn and mlp branches (sub-layer LayerDrop).
        if self.training and self.layer_drop > 0.0:
            # Per-sample keep mask would be more standard, but per-layer scalar matches your old behavior.
            keep_attn = torch.empty(1, device=x.device, dtype=x.dtype).bernoulli_(1.0 - self.layer_drop)
            keep_mlp  = torch.empty(1, device=x.device, dtype=x.dtype).bernoulli_(1.0 - self.layer_drop)
            x = x + keep_attn * self.attn(self.norm1(x))
            x = x + keep_mlp  * self.mlp(self.norm2(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x
    
class Pos2MoveV2(nn.Module):
    def __init__(
        self,
        vocab_size: int = 13,
        embed_dim: int = 512,
        nb_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        kvq_bias: bool = False,
        layer_drop: float = 0.0,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(67, embed_dim)  # 64 squares + 3 extra tokens
        self.castling_embedding = nn.Embedding(16, embed_dim)  # 4-bit bitmask: 0-15
        self.en_passant_embedding = nn.Embedding(9, embed_dim)  # 8 files + no en passant (8)
        self.player_embedding = nn.Embedding(2, embed_dim)  # White or Black

        # Linearly increasing layer drop rate: deeper layers drop more.
        # This is the "Deep Networks with Stochastic Depth" schedule.
        drop_rates = [layer_drop * i / max(1, nb_transformer_layers - 1)
                      for i in range(nb_transformer_layers)]

        self.transformer_layers = nn.ModuleList(
            [
                ChessTransformerLayer(
                    dim=embed_dim,
                    nb_heads=num_heads,
                    qkv_bias=kvq_bias,
                    context_size=67,  # Max sequence length (64 squares + 3 extra tokens)
                    dropout=dropout,
                    layer_drop=drop_rates[i],
                )
                for i in range(nb_transformer_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(embed_dim)

        policy_hidden = embed_dim // 2
        self.move_head = nn.Sequential(
            nn.Linear(embed_dim, policy_hidden),
            nn.GELU(),
            nn.Linear(policy_hidden, NUM_ACTION_PLANES),
        )

        # Deeper value head: D → D//4 → GELU → 1
        value_hidden = embed_dim // 4
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )

        self._init_weights(nb_transformer_layers)

    def _init_weights(self, nb_layers: int):
        """GPT-2 style init with depth-scaled residual projections.

        - All Linear/Embedding: N(0, 0.02)
        - Residual projections (attn out + mlp out): scaled by 1/sqrt(2*N)
          to keep residual stream variance bounded with depth.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale down the projections that write into the residual stream
        scale = (2 * nb_layers) ** -0.5
        for layer in self.transformer_layers:
            with torch.no_grad():
                layer.attn.proj.weight.mul_(scale)
                layer.mlp.fc2.weight.mul_(scale)

    def forward(self, board_tokens, player_token, castling_token, en_passant_token):
        batch_size = board_tokens.size(0)

        token_emb = self.token_embedding(board_tokens)  # (B, 64, D)
        pos_emb = self.position_embedding(torch.arange(67, device=board_tokens.device))  # (67, D)
        castling_emb = self.castling_embedding(castling_token).unsqueeze(1)  # (B, 1, D)
        en_passant_emb = self.en_passant_embedding(en_passant_token).unsqueeze(1)  # (B, 1, D)
        player_emb = self.player_embedding(player_token).unsqueeze(1)  # (B, 1, D)

        x = token_emb + pos_emb[:64]  # Add positional embedding to board tokens
        x = torch.cat([
            x,
            castling_emb + pos_emb[64],
            en_passant_emb + pos_emb[65],
            player_emb + pos_emb[66],
        ], dim=1)  # (B, 67, D)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.final_norm(x)

        state_token = x[:, -3:, :].mean(dim=1)  # Average the last 3 game-state tokens: (B, D)

        board_tokens_out = x[:, :64, :]  # (B, 64, D)
        move_logits = self.move_head(board_tokens_out)  # (B, 64, NUM_ACTION_PLANES)
        value_pred = self.value_head(state_token)  # (B, 1)

        return move_logits, value_pred
