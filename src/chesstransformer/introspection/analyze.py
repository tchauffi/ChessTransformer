"""Per-position introspection of a Pos2MoveV2 network.

Everything here is read-only with respect to the model: attention maps are
recomputed from the captured layer inputs using the module's own projections,
so the numbers match what SDPA saw during the real forward pass.

Produced per position:
  * policy over legal moves (probability, raw logit, action plane)
  * value head output, plus the value/policy read off *every* layer
    (logit lens) so you can see at which depth the model makes up its mind
  * full attention tensor (L, H, 67, 67)
  * attention rollout (Abnar & Zuidema 2020) over the layer stack
  * occlusion saliency: remove one piece, measure the swing in value and in
    the probability of the move the model wanted to play
  * 1-ply child values, i.e. what the value head says about the position
    *after* each legal move — the fastest way to spot policy/value disagreement
"""

from __future__ import annotations

import json
import math
import zlib
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
from safetensors import safe_open

from chesstransformer.models.tokenizer import PostionTokenizer
from chesstransformer.models.tokenizer.alphazero_move_encoder import (
    NUM_ACTION_PLANES,
    move_to_action_plane,
)
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2

BOARD_SQUARES = 64
CONTEXT_SIZE = 67
STATE_TOKEN_NAMES = ["castling", "en_passant", "player"]


def load_model(model_dir: str | Path, device: str = "cpu", use_ema: bool = False):
    """Load a Pos2MoveV2 checkpoint in fp32 (no compile, no bf16).

    Introspection wants clean numbers more than it wants speed, and the model
    is small enough that fp32 eager is instant either way.
    """
    model_dir = Path(model_dir)

    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        config_path = model_dir.parent.parent / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No model_config.json found for {model_dir}")
    with open(config_path) as f:
        config = json.load(f)

    model = Pos2MoveV2(**config)

    weights = model_dir / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"No model.safetensors in {model_dir}")
    with safe_open(str(weights), framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    if use_ema:
        ema_path = model_dir / "ema_state.pt"
        if not ema_path.exists():
            raise FileNotFoundError(f"No ema_state.pt in {model_dir}")
        ema_state = torch.load(str(ema_path), map_location="cpu", weights_only=True)
        ema_state = {k.replace("_orig_mod.", ""): v for k, v in ema_state.items()}
        model.load_state_dict(ema_state, strict=False)

    model.eval()
    return model.float().to(device), config


def encode_board(board: chess.Board, tokenizer: PostionTokenizer):
    """Same encoding the bots use: 64 piece tokens + castling/ep/player."""
    tokens = tokenizer.encode(board)
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    ep_file = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
    return tokens, int(board.turn), castling, ep_file


@dataclass
class _Capture:
    """Residual stream and attention input for one layer."""

    attn_input: torch.Tensor  # normed input handed to the attention module
    output: torch.Tensor  # residual stream leaving the layer


class PositionAnalyzer:
    def __init__(
        self,
        model: Pos2MoveV2,
        device: str = "cpu",
        max_moves: int = 64,
        saliency: bool = True,
    ):
        self.model = model
        self.device = device
        self.tokenizer = PostionTokenizer()
        self.max_moves = max_moves
        self.saliency = saliency
        self.n_layers = len(model.transformer_layers)
        self.n_heads = model.transformer_layers[0].attn.num_heads

    # ── raw forward helpers ────────────────────────────────────────────
    @torch.no_grad()
    def _forward_batch(self, boards: list[chess.Board]):
        enc = [encode_board(b, self.tokenizer) for b in boards]
        tokens = torch.tensor([e[0] for e in enc], dtype=torch.long, device=self.device)
        player = torch.tensor([e[1] for e in enc], dtype=torch.long, device=self.device)
        castling = torch.tensor([e[2] for e in enc], dtype=torch.long, device=self.device)
        ep = torch.tensor([e[3] for e in enc], dtype=torch.long, device=self.device)
        return self.model(tokens, player, castling, ep)

    @torch.no_grad()
    def _forward_tokens(self, tokens, player, castling, ep):
        return self.model(tokens, player, castling, ep)

    @torch.no_grad()
    def _forward_with_capture(self, board: chess.Board):
        """One forward pass, keeping every layer's input and output."""
        captures: list[_Capture] = []
        pending: dict[int, torch.Tensor] = {}
        handles = []

        def make_pre_hook(idx):
            def hook(_module, args):
                pending[idx] = args[0].detach()

            return hook

        def make_post_hook(idx):
            def hook(_module, _args, output):
                captures.append(_Capture(attn_input=pending[idx], output=output.detach()))

            return hook

        for i, layer in enumerate(self.model.transformer_layers):
            handles.append(layer.attn.register_forward_pre_hook(make_pre_hook(i)))
            handles.append(layer.register_forward_hook(make_post_hook(i)))

        try:
            tokens, player, castling, ep = encode_board(board, self.tokenizer)
            out = self._forward_tokens(
                torch.tensor([tokens], dtype=torch.long, device=self.device),
                torch.tensor([player], dtype=torch.long, device=self.device),
                torch.tensor([castling], dtype=torch.long, device=self.device),
                torch.tensor([ep], dtype=torch.long, device=self.device),
            )
        finally:
            for h in handles:
                h.remove()

        return out, captures

    @torch.no_grad()
    def _attention_probs(self, layer_idx: int, attn_input: torch.Tensor) -> torch.Tensor:
        """Recompute softmax attention for one layer: (H, 67, 67).

        Mirrors ChessGroupedQueryAttention.forward up to the softmax — SDPA
        adds the chess-relation bias to the *scaled* scores, so we do too.
        """
        attn = self.model.transformer_layers[layer_idx].attn
        x = attn_input
        b, t, _ = x.shape

        q = attn.q_proj(x).view(b, t, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(b, t, attn.num_kv_groups, attn.head_dim).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        k = k.repeat_interleave(attn.group_size, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        scores = scores + attn.bias_table[:, attn.rel_idx].unsqueeze(0)
        return torch.softmax(scores, dim=-1)[0]

    # ── analysis ───────────────────────────────────────────────────────
    def analyze(self, board: chess.Board, include_attention: bool = True) -> dict:
        (move_logits, value), captures = self._forward_with_capture(board)
        logits = move_logits[0].float()  # (64, 73)
        value = float(value[0].item())

        result: dict = {
            "fen": board.fen(),
            "turn": "w" if board.turn else "b",
            "fullmove": board.fullmove_number,
            "check": board.is_check(),
            "pieces": [
                (board.piece_at(sq).symbol() if board.piece_at(sq) else None)
                for sq in range(64)
            ],
            "value": value,
        }

        legal = list(board.legal_moves)
        result["legal_count"] = len(legal)
        if not legal:
            result["policy"] = {"moves": [], "entropy": 0.0, "top1_prob": 0.0}
            result["terminal"] = board.result()
            return result

        result.update(self._policy_block(board, logits, legal))
        result.update(self._depth_block(captures, logits, board, legal))

        if include_attention:
            result["attention"] = self._attention_block(captures)

        if self.saliency:
            result["saliency"] = self._saliency_block(board, value, result)

        return result

    def _policy_block(self, board: chess.Board, logits: torch.Tensor, legal) -> dict:
        planes = [
            move_to_action_plane(m.from_square, m.to_square, m.promotion) for m in legal
        ]
        idx = torch.tensor(
            [m.from_square * NUM_ACTION_PLANES + p for m, p in zip(legal, planes)],
            dtype=torch.long,
        )
        flat = logits.reshape(-1).cpu()
        legal_logits = flat[idx]
        probs = torch.softmax(legal_logits, dim=0)

        # Illegal mass: how much of the raw 64x73 softmax the model spends on
        # moves that do not exist. A cheap sanity signal for a broken head.
        full_probs = torch.softmax(flat, dim=0)
        legal_mass = float(full_probs[idx].sum())

        order = torch.argsort(probs, descending=True)
        moves = []
        for rank, i in enumerate(order.tolist()):
            m = legal[i]
            moves.append(
                {
                    "uci": m.uci(),
                    "san": board.san(m),
                    "from": m.from_square,
                    "to": m.to_square,
                    "plane": planes[i],
                    "prob": float(probs[i]),
                    "logit": float(legal_logits[i]),
                    "rank": rank,
                    "capture": board.is_capture(m),
                    "check": board.gives_check(m),
                }
            )

        p = probs[probs > 0]
        entropy = float(-(p * p.log()).sum())
        return {
            "policy": {
                "moves": moves[: self.max_moves],
                "n_moves": len(moves),
                "entropy": entropy,
                "max_entropy": math.log(len(legal)),
                "top1_prob": moves[0]["prob"],
                "legal_mass": legal_mass,
            }
        }

    @torch.no_grad()
    def _depth_block(self, captures, final_logits, board, legal) -> dict:
        """Logit lens: decode value + policy from every layer's residual."""
        norm = self.model.final_norm
        value_by_layer = []
        top_by_layer = []
        write_norms = []

        planes = [
            move_to_action_plane(m.from_square, m.to_square, m.promotion) for m in legal
        ]
        idx = torch.tensor(
            [m.from_square * NUM_ACTION_PLANES + p for m, p in zip(legal, planes)],
            dtype=torch.long,
            device=self.device,
        )
        final_probs = torch.softmax(final_logits.reshape(-1)[idx.cpu()], dim=0)
        final_top = int(torch.argmax(final_probs))

        prev = None
        for cap in captures:
            x = norm(cap.output)
            state = x[:, -3:, :].mean(dim=1)
            value_by_layer.append(float(self.model.value_head(state)[0].item()))

            lens_logits = self.model.move_head(x[:, :64, :])[0].reshape(-1)[idx]
            lens_probs = torch.softmax(lens_logits.float(), dim=0)
            best = int(torch.argmax(lens_probs))
            top_by_layer.append(
                {
                    "uci": legal[best].uci(),
                    "san": board.san(legal[best]),
                    "prob": float(lens_probs[best]),
                    "final_move_prob": float(lens_probs[final_top]),
                    "agrees": best == final_top,
                }
            )

            out = cap.output
            write_norms.append(
                float((out - prev).norm().item()) if prev is not None else float(out.norm().item())
            )
            prev = out

        return {
            "depth": {
                "value_by_layer": value_by_layer,
                "top_by_layer": top_by_layer,
                "residual_write_norm": write_norms,
            }
        }

    @torch.no_grad()
    def _attention_block(self, captures) -> dict:
        """Full (L, H, 67, 67) attention, u8-quantised per row, deflated."""
        maps = []
        for i, cap in enumerate(captures):
            maps.append(self._attention_probs(i, cap.attn_input).cpu())
        attn = torch.stack(maps)  # (L, H, 67, 67)

        rollout = self._rollout(attn)

        # u8 per row, companded through a square root: attention rows are long
        # tails of small weights, and linear u8 would quantise those to 20-30%
        # relative error. sqrt spreads the error evenly across magnitudes.
        row_max = attn.amax(dim=-1, keepdim=True).clamp(min=1e-9)
        quant = ((attn / row_max).sqrt() * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        payload = zlib.compress(quant.tobytes(), 6)

        return {
            "shape": list(attn.shape),
            "scale": row_max.squeeze(-1).flatten().numpy().astype(np.float32).tolist(),
            "data": b64encode(payload).decode("ascii"),
            "rollout": [round(v, 5) for v in rollout.tolist()],
        }

    @staticmethod
    def _rollout(attn: torch.Tensor) -> torch.Tensor:
        """Attention rollout: how much of each token the output actually sees.

        Head-averaged, residual-corrected (0.5*A + 0.5*I), multiplied through
        the stack, then read from the three game-state token rows — those are
        what the value head consumes.
        """
        a = attn.mean(dim=1)  # (L, T, T)
        eye = torch.eye(a.shape[-1])
        joint = None
        for layer in a:
            m = 0.5 * layer + 0.5 * eye
            m = m / m.sum(dim=-1, keepdim=True)
            joint = m if joint is None else m @ joint
        return joint[-3:, :].mean(dim=0)

    @torch.no_grad()
    def _saliency_block(self, board: chess.Board, base_value: float, result: dict) -> dict:
        """Occlusion: delete one piece at a time, measure what moves.

        Kings are left in place (an occluded king is not a chess position the
        model has ever seen). Everything runs as one batch.
        """
        top = result["policy"]["moves"][0]
        top_idx = top["from"] * NUM_ACTION_PLANES + top["plane"]

        squares = [
            sq
            for sq in range(64)
            if board.piece_at(sq) is not None and board.piece_at(sq).piece_type != chess.KING
        ]
        value_delta = [0.0] * 64
        policy_delta = [0.0] * 64
        if not squares:
            return {"value_delta": value_delta, "policy_delta": policy_delta, "squares": []}

        tokens, player, castling, ep = encode_board(board, self.tokenizer)
        base = torch.tensor(tokens, dtype=torch.long).repeat(len(squares), 1)
        for row, sq in enumerate(squares):
            base[row, sq] = 0
        n = len(squares)
        move_logits, values = self._forward_tokens(
            base.to(self.device),
            torch.full((n,), player, dtype=torch.long, device=self.device),
            torch.full((n,), castling, dtype=torch.long, device=self.device),
            torch.full((n,), ep, dtype=torch.long, device=self.device),
        )

        # Probability of the top move under the *full* 64x73 softmax: occlusion
        # changes the legal move set, so a legal-masked comparison would not be
        # measuring the same quantity across rows.
        flat = move_logits.float().reshape(n, -1).cpu()
        probs = torch.softmax(flat, dim=1)[:, top_idx]

        ref_logits, _ = self._forward_batch([board])
        ref_prob = float(torch.softmax(ref_logits[0].float().reshape(-1), dim=0)[top_idx])

        for row, sq in enumerate(squares):
            value_delta[sq] = float(values[row].item()) - base_value
            policy_delta[sq] = float(probs[row]) - ref_prob

        return {
            "value_delta": [round(v, 5) for v in value_delta],
            "policy_delta": [round(v, 5) for v in policy_delta],
            "squares": squares,
            "top_move": top["uci"],
            "ref_prob": ref_prob,
        }

    @torch.no_grad()
    def child_values(self, board: chess.Board, moves: list[dict]) -> None:
        """Fill in the value of the position *after* each candidate move.

        Stored from the mover's point of view (the raw head is side-to-move
        POV, so it gets negated) — directly comparable to ``value``.
        """
        if not moves:
            return
        boards = []
        terminal = {}
        for i, m in enumerate(moves):
            b = board.copy(stack=False)
            b.push(chess.Move.from_uci(m["uci"]))
            if b.is_checkmate():
                terminal[i] = 1.0
            elif b.is_stalemate() or b.is_insufficient_material():
                terminal[i] = 0.0
            boards.append(b)

        _, values = self._forward_batch(boards)
        for i, m in enumerate(moves):
            m["child_value"] = (
                terminal[i] if i in terminal else -float(values[i].item())
            )
