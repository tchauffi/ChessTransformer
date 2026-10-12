import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTION_PLANES = 73
BOARD_SQUARES = 64
EXTRA_TOKENS = 3  # castling, en_passant, player
CONTEXT_SIZE = BOARD_SQUARES + EXTRA_TOKENS  # 67
RULE50_MAX = 100  # halfmove clock values 0..100; 100 = the fifty-move draw can be claimed
REPETITION_MAX = 2  # earlier occurrences of the position: 0, 1, 2+


def build_chess_relation_index(
    context_size: int = CONTEXT_SIZE, state_category: bool = False
) -> torch.Tensor:
    """Precompute (T, T) relation index between every pair of tokens.

    Categories (8 by default):
      0: same square / same token
      1: same file
      2: same rank
      3: same diagonal
      4: knight-reachable
      5: king-adjacent (Chebyshev <= 1)
      6: nearby (Manhattan <= 3)
      7: far

    Any pair involving a game-state token (>= 64) is "global". With
    ``state_category=False`` — the v2.1 behaviour — global reuses bucket **7**,
    so the state tokens' only channel to the board shares one learned bias per
    head with the least informative board relation. Since the value head reads
    those tokens, that bias is load-bearing for every MCTS leaf score.
    ``state_category=True`` gives global its own bucket 8 (9 categories).
    """
    rel = torch.zeros(context_size, context_size, dtype=torch.long)
    global_bucket = 8 if state_category else 7
    for i in range(context_size):
        for j in range(context_size):
            if i == j:
                rel[i, j] = 0
                continue
            if i >= BOARD_SQUARES or j >= BOARD_SQUARES:
                rel[i, j] = global_bucket
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
_RELATION_INDEX_STATE: torch.Tensor = build_chess_relation_index(state_category=True)


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class Smolgen(nn.Module):
    """Position-dependent attention bias, after Leela Chess Zero's "smolgen".

    The relation table gives each head one fixed bias per pair of squares, the same
    in every position. Smolgen reads the whole board and writes a bias per head and
    per pair for *this* position, so a head can follow an open file or a pin rather
    than a geometric pattern. The board is compressed to a few numbers per token,
    flattened, and mapped to a small code per head; one projection shared by every
    layer (``gen_proj``, owned by the model) expands each code to a (T, T) bias.
    Leela reports it plays as if the network were 50% larger, for 10% throughput.
    """

    def __init__(self, dim, num_heads, context_size, compress, hidden, gen):
        super().__init__()
        self.compress = nn.Linear(dim, compress, bias=False)
        self.dense1 = nn.Linear(context_size * compress, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.dense2 = nn.Linear(hidden, num_heads * gen)
        self.norm2 = nn.LayerNorm(num_heads * gen)
        self.num_heads = num_heads
        self.context_size = context_size

    def forward(self, x, gen_proj):  # (B, T, D) -> (B, H, T, T)
        h = self.compress(x).flatten(1)
        h = self.norm1(F.silu(self.dense1(h)))
        h = self.norm2(F.silu(self.dense2(h))).unflatten(-1, (self.num_heads, -1))
        return gen_proj(h).unflatten(-1, (self.context_size, self.context_size))


class ChessGroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads, num_kv_groups, context_size, dropout=0.0, qkv_bias=False,
        state_relation=False, attn_bias="buckets", smolgen=None,
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

        # Chess-geometry relative bias: one learned scalar per (head, relation).
        # Massively fewer params than a fully-learned (H, T, T) bias.
        self.num_relations = 9 if state_relation else 8
        self.register_buffer(
            "rel_idx", _RELATION_INDEX_STATE if state_relation else _RELATION_INDEX,
            persistent=False,
        )
        # attn_bias "full" learns one scalar per (head, token pair) instead. The
        # relation buckets tie together pairs that should differ: two of the eight
        # (king-adjacent, nearby) are shadowed by the file / rank / diagonal / knight
        # rules and match no pair at all, 55% of square pairs share "far", and the
        # table cannot tell a pawn's forward capture from a backward diagonal. A
        # bucket checkpoint loads into it exactly (see _load_from_state_dict), and it
        # costs nothing at inference: attention already takes a full (H, T, T) mask.
        if attn_bias == "buckets":
            self.bias_table = nn.Parameter(torch.zeros(num_heads, self.num_relations))
            self.bias_full = None
        elif attn_bias == "full":
            self.bias_table = None
            self.bias_full = nn.Parameter(torch.zeros(num_heads, context_size, context_size))
        else:
            raise ValueError(f"attn_bias must be 'buckets' or 'full', got {attn_bias!r}")
        self.smolgen = Smolgen(d_in, num_heads, context_size, **smolgen) if smolgen else None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A bucket checkpoint into a full-table model: expand the table to the pairs.
        old, new = prefix + "bias_table", prefix + "bias_full"
        if self.bias_full is not None and old in state_dict and new not in state_dict:
            table = state_dict.pop(old)
            state_dict[new] = table[:, self.rel_idx.to(table.device)]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x, gen_proj=None):
        batch_size, num_tokens, embed_dim = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        queries = self.q_norm(queries)
        keys_new = self.k_norm(keys_new)

        # Expand K/V to full head count. At this tiny sequence length (67) the
        # repeat_interleave is cheaper than SDPA's native enable_gqa path, which
        # falls back to a slower kernel here (measured ~20% slower).
        keys = keys_new.repeat_interleave(self.group_size, dim=1)
        values = values_new.repeat_interleave(self.group_size, dim=1)

        # Build additive chess-geometry bias from the compact (H, 8) table.
        # Shape: (1, H, T, T) so SDPA broadcasts it across the batch.
        if self.bias_full is not None:
            attn_bias = self.bias_full.unsqueeze(0)
        else:
            attn_bias = self.bias_table[:, self.rel_idx].unsqueeze(0)
        if self.smolgen is not None:
            attn_bias = attn_bias + self.smolgen(x, gen_proj)  # (B, H, T, T)

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
    def __init__(self, dim, nb_heads, qkv_bias, context_size, dropout=0.1, layer_drop=0.0,
                 state_relation=False, attn_bias="buckets", smolgen=None):
        super().__init__()
        self.attn = ChessGroupedQueryAttention(
            d_in=dim,
            d_out=dim,
            num_heads=nb_heads,
            num_kv_groups=nb_heads // 2,
            context_size=context_size,
            dropout=dropout,
            qkv_bias=qkv_bias,
            state_relation=state_relation,
            attn_bias=attn_bias,
            smolgen=smolgen,
        )
        self.mlp = MLP(dim, dim * 4, dim, dropout=dropout)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.layer_drop = layer_drop

    def forward(self, x, gen_proj=None):
        # Stochastic depth: with prob `layer_drop`, skip the entire layer.
        # Applied independently to attn and mlp branches (sub-layer LayerDrop).
        if self.training and self.layer_drop > 0.0:
            # Per-sample keep mask would be more standard, but per-layer scalar matches your old behavior.
            keep_attn = torch.empty(1, device=x.device, dtype=x.dtype).bernoulli_(1.0 - self.layer_drop)
            keep_mlp  = torch.empty(1, device=x.device, dtype=x.dtype).bernoulli_(1.0 - self.layer_drop)
            x = x + keep_attn * self.attn(self.norm1(x), gen_proj)
            x = x + keep_mlp  * self.mlp(self.norm2(x))
        else:
            x = x + self.attn(self.norm1(x), gen_proj)
            x = x + self.mlp(self.norm2(x))
        return x
    
class ValueAttentionPool(nn.Module):
    """Pool the whole sequence into one vector with a single learned query.

    The v2.1 value head reads ``x[:, -3:, :].mean(dim=1)`` -- the mean of the
    castling / en-passant / player tokens -- so all 64 squares reach it only by
    having been attended *into* those three slots, while the policy head reads
    every square directly. This pools over all 67 tokens instead.

    ``q`` is initialised near zero, which makes the softmax uniform, so the
    module starts as a mean-pool over the full sequence and learns where to look
    from there.
    """

    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(dim))
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x):  # (B, T, D) -> (B, D)
        attn = (self.k_proj(x) @ self.q) * self.scale  # (B, T)
        w = attn.softmax(dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (w * self.v_proj(x)).sum(dim=1)


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
        value_bounded: bool = False,
        value_clamp: bool = False,
        value_head_type: str = "scalar",
        value_pool: str = "state",
        state_relation: bool = False,
        value_elo_offset: bool = False,
        head_tower_layers: int = 0,
        attn_bias: str = "buckets",
        smolgen: bool = False,
        smolgen_compress: int = 8,
        smolgen_hidden: int = 128,
        smolgen_gen: int = 64,
        rule_state: bool = False,
    ):
        super().__init__()
        if not 0 <= head_tower_layers < nb_transformer_layers:
            raise ValueError(
                f"head_tower_layers must be in [0, {nb_transformer_layers}), got {head_tower_layers}"
            )
        if value_head_type not in ("scalar", "wdl"):
            raise ValueError(f"value_head_type must be 'scalar' or 'wdl', got {value_head_type!r}")
        if value_pool not in ("state", "attn"):
            raise ValueError(f"value_pool must be 'state' or 'attn', got {value_pool!r}")
        if value_head_type == "wdl" and (value_bounded or value_clamp):
            raise ValueError(
                "value_bounded / value_clamp are scalar-head options; a WDL head is already "
                "bounded because its value is p(win) - p(loss)."
            )
        if value_elo_offset and value_head_type != "wdl":
            raise ValueError("value_elo_offset shifts win/loss logits, so it needs a WDL head")
        self.value_head_type = value_head_type
        self.value_pool_kind = value_pool

        # Rating-gap offset for the value *target* (training only). The outcome of a
        # human game depends on who was stronger as well as on the position, and the
        # network has no rating input, so without this it explains the stronger
        # player's results through whatever position features correlate with them --
        # e.g. "1.a4 wins", because strong players choose rim openings against weaker
        # ones (docs/search-scaling-and-value-bias.md). With it, training fits
        #   log p(win)/p(loss) = f(position) + 2 * elo_beta * gap / 400
        # so f only has to carry what the board says. Inference passes no gap, which
        # reads the head as a game between equals. elo_beta starts at ln(10)/2, the
        # Elo model's own slope, and is learned (rating noise attenuates it).
        self.elo_beta = (
            nn.Parameter(torch.tensor(math.log(10.0) / 2.0)) if value_elo_offset else None
        )

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(67, embed_dim)  # 64 squares + 3 extra tokens
        # Static position indices, kept as a buffer so we don't rebuild the
        # arange (and a host->device copy) on every forward pass.
        self.register_buffer("pos_index", torch.arange(67), persistent=False)
        self.castling_embedding = nn.Embedding(16, embed_dim)  # 4-bit bitmask: 0-15
        self.en_passant_embedding = nn.Embedding(9, embed_dim)  # 8 files + no en passant (8)
        self.player_embedding = nn.Embedding(2, embed_dim)  # White or Black

        # Draw rules the board cannot show: the fifty-move counter (halfmove clock) and how
        # often the position already occurred in the game + search path. Without them a
        # position one repetition or a few moves from a forced draw reads the same as a
        # fresh one, and the value head has to average the two. They are added to the
        # player token (no new token, so the attention layout is unchanged) and start at
        # zero, so a converted checkpoint plays exactly as before until trained. Callers
        # that pass neither (the human shards carry no game history) get the old model.
        self.rule_state = rule_state
        if rule_state:
            self.rule50_embedding = nn.Embedding(RULE50_MAX + 1, embed_dim)
            self.repetition_embedding = nn.Embedding(REPETITION_MAX + 1, embed_dim)

        # Linearly increasing layer drop rate: deeper layers drop more.
        # This is the "Deep Networks with Stochastic Depth" schedule.
        drop_rates = [layer_drop * i / max(1, nb_transformer_layers - 1)
                      for i in range(nb_transformer_layers)]

        # Smolgen (see Smolgen): one (gen -> T*T) projection shared by every layer, as in
        # Leela. It starts at zero, so a converted checkpoint plays exactly as before
        # until training moves it.
        smol = (dict(compress=smolgen_compress, hidden=smolgen_hidden, gen=smolgen_gen)
                if smolgen else None)
        self.smolgen_proj = nn.Linear(smolgen_gen, 67 * 67, bias=False) if smolgen else None

        def make_layer(i):
            return ChessTransformerLayer(
                dim=embed_dim,
                nb_heads=num_heads,
                qkv_bias=kvq_bias,
                context_size=67,  # Max sequence length (64 squares + 3 extra tokens)
                dropout=dropout,
                layer_drop=drop_rates[i],
                state_relation=state_relation,
                attn_bias=attn_bias,
                smolgen=smol,
            )

        # Per-head towers: the last `head_tower_layers` layers are duplicated, one copy
        # private to each head, so each head still sees nb_transformer_layers layers but
        # only the first nb - K are shared. The two objectives barely *conflict* --
        # their trunk gradients are near-orthogonal (|cos| < 0.11 in every layer of
        # run_026) -- they *compete*: Muon and AdamW are scale-invariant, so the value
        # loss weight sets nothing but the value objective's share of each update to
        # the parameters both heads train. A tower gives each head layers it does not
        # have to share; the weight then only balances the shared part. 0 = v2.1.
        n_shared = nb_transformer_layers - head_tower_layers
        self.head_tower_layers = head_tower_layers
        self.transformer_layers = nn.ModuleList([make_layer(i) for i in range(n_shared)])
        tower = range(n_shared, nb_transformer_layers)
        self.policy_tower = nn.ModuleList([make_layer(i) for i in tower]) if head_tower_layers else None
        self.value_tower = nn.ModuleList([make_layer(i) for i in tower]) if head_tower_layers else None

        self.final_norm = nn.RMSNorm(embed_dim)
        self.value_norm = nn.RMSNorm(embed_dim) if head_tower_layers else None

        policy_hidden = embed_dim // 2
        self.move_head = nn.Sequential(
            nn.Linear(embed_dim, policy_hidden),
            nn.GELU(),
            nn.Linear(policy_hidden, NUM_ACTION_PLANES),
        )

        # How the value head reads the trunk. "state" is v2.1's three-token mean;
        # "attn" pools all 67 tokens (see ValueAttentionPool).
        self.value_pool = ValueAttentionPool(embed_dim) if value_pool == "attn" else None

        # Deeper value head: D → D//4 → GELU → out
        #
        # value_head_type "wdl" emits 3 logits (win / draw / loss, side-to-move
        # POV) trained with cross-entropy instead of one scalar trained with MSE.
        # MSE against a {-1, 0, +1} outcome converges to a shrunken conditional
        # mean -- v2.1's head has output std 0.42 against a target std of 0.95 --
        # and MCTS maximises over those leaf values. A softmax cannot shrink that
        # way, is bounded by construction, and makes the draw an explicit outcome
        # rather than the midpoint of two others.
        #
        # value_bounded appends a tanh so a *scalar* output cannot leave the
        # target's [-1, 1] range. Kept off by default: the shipped v2.1 weights
        # were trained without it, and squashing them post-hoc changes their
        # scale. The Tanh lands at index 3, so the Linear state-dict keys
        # (0.*, 2.*) are identical either way and checkpoints stay interchangeable.
        value_hidden = embed_dim // 4
        value_layers = [
            nn.Linear(embed_dim, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 3 if value_head_type == "wdl" else 1),
        ]
        if value_bounded:
            value_layers.append(nn.Tanh())
        elif value_clamp:
            # Minimal alternative to tanh for weights already trained unbounded:
            # a hard clamp leaves every in-range output untouched and only edits
            # the semantically impossible ones (a leaf "better than won"). v2.1
            # emits those on 0.76% of positions, reaching 1.039 -- and since MCTS
            # maximises over leaf values, that tail is precisely what it chases.
            # MEASURED 2026-08-19: on v2.1 this is a NO-OP -- the chosen move was
            # identical on 25/25 opening positions at both 4000 and 8000 sims. A
            # 4% excursion is too small to steer the search; it mattered for the
            # cp-trained head only because that one reached 1.211. Don't re-run.
            # Kept for unbounded heads that overshoot more; prefer value_bounded
            # (tanh) when training a head from scratch.
            value_layers.append(nn.Hardtanh(-1.0, 1.0))
        self.value_head = nn.Sequential(*value_layers)

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

        # Scale down the projections that write into the residual stream. A tower layer
        # sits at the same depth of its head's path as the layer it replaces, so it
        # gets the same scale.
        scale = (2 * nb_layers) ** -0.5
        for layer in self.modules():
            if not isinstance(layer, ChessTransformerLayer):
                continue
            with torch.no_grad():
                layer.attn.proj.weight.mul_(scale)
                layer.mlp.fc2.weight.mul_(scale)
        if self.smolgen_proj is not None:
            nn.init.zeros_(self.smolgen_proj.weight)
        if self.rule_state:
            nn.init.zeros_(self.rule50_embedding.weight)
            nn.init.zeros_(self.repetition_embedding.weight)

    def _trunk(self, board_tokens, player_token, castling_token, en_passant_token,
               rule50=None, repetition=None):
        """Shared forward up to the heads.

        Returns ``(move_logits, value_features)``. Split out so training can read
        raw WDL logits and inference can read a scalar without either paying for
        a second pass through the trunk.
        """
        token_emb = self.token_embedding(board_tokens)  # (B, 64, D)
        pos_emb = self.position_embedding(self.pos_index)  # (67, D)
        castling_emb = self.castling_embedding(castling_token).unsqueeze(1)  # (B, 1, D)
        en_passant_emb = self.en_passant_embedding(en_passant_token).unsqueeze(1)  # (B, 1, D)
        player_emb = self.player_embedding(player_token).unsqueeze(1)  # (B, 1, D)
        if self.rule_state and rule50 is not None and repetition is not None:
            player_emb = player_emb + (
                self.rule50_embedding(rule50.long().clamp(0, RULE50_MAX))
                + self.repetition_embedding(repetition.long().clamp(0, REPETITION_MAX))
            ).unsqueeze(1)

        x = token_emb + pos_emb[:64]  # Add positional embedding to board tokens
        x = torch.cat([
            x,
            castling_emb + pos_emb[64],
            en_passant_emb + pos_emb[65],
            player_emb + pos_emb[66],
        ], dim=1)  # (B, 67, D)

        g = self.smolgen_proj
        for layer in self.transformer_layers:
            x = layer(x, g)

        if self.head_tower_layers:
            xv = x
            for layer in self.policy_tower:
                x = layer(x, g)
            for layer in self.value_tower:
                xv = layer(xv, g)
            xv = self.value_norm(xv)
        x = self.final_norm(x)
        if not self.head_tower_layers:
            xv = x

        if self.value_pool is not None:
            value_features = self.value_pool(xv)  # (B, D) over all 67 tokens
        else:
            value_features = xv[:, -3:, :].mean(dim=1)  # v2.1: mean of the 3 state tokens

        move_logits = self.move_head(x[:, :64, :])  # (B, 64, NUM_ACTION_PLANES)
        return move_logits, value_features

    def forward(self, board_tokens, player_token, castling_token, en_passant_token,
                elo_gap=None, rule50=None, repetition=None):
        """``(move_logits, value)`` with ``value`` the **raw** head output.

        (B, 1) for a scalar head -- byte-identical to v2.1 -- and (B, 3) logits
        for a WDL head, which is what cross-entropy needs. Anything that wants
        the single number MCTS backs up calls :meth:`value_scalar` on the result;
        for a scalar head that is the identity, so existing callers are unaffected.

        ``elo_gap`` (B,) is the side to move's rating minus its opponent's, and is
        for training only: with ``value_elo_offset`` it shifts the win/loss logits
        by the rating gap (see ``elo_beta``). Leave it ``None`` at inference.

        ``rule50`` (B,) is the halfmove clock and ``repetition`` (B,) the number of
        earlier occurrences of the position in the game + search path; both only
        reach a ``rule_state`` model, and only when both are given.

        Deliberately the only forward path: DDP registers its gradient-sync hooks
        on ``forward``, and ``torch.compile`` only traces what goes through
        ``__call__``, so a separate training entry point would quietly lose both.
        """
        move_logits, value_features = self._trunk(
            board_tokens, player_token, castling_token, en_passant_token, rule50, repetition
        )
        value = self.value_head(value_features)
        if elo_gap is not None and self.elo_beta is not None:
            k = self.elo_beta * (elo_gap.float() / 400.0)  # (B,)
            value = value + torch.stack([k, torch.zeros_like(k), -k], dim=-1).to(value.dtype)
        return move_logits, value

    def value_scalar(self, value_out):
        """Reduce a raw value output to the scalar in [-1, 1] that search uses.

        Identity for a scalar head; Q = p(win) - p(loss) for a WDL head.
        """
        if value_out.size(-1) == 3:
            p = value_out.softmax(dim=-1)
            return (p[..., 0] - p[..., 2]).unsqueeze(-1)
        return value_out
