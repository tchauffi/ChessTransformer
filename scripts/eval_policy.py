"""Compare the policy head of one or more models against Stockfish, on a fixed,
held-out set of positions (apples-to-apples).

Samples positions from Lichess (held-out seed), computes each position's
Stockfish best move once, then for each model reports policy top-1 / top-3
agreement with Stockfish (legal-move masked) and the mean centipawn lost by the
model's top move.

Usage
-----
    uv run python scripts/eval_policy.py \
        --models data/models/pos2move_v2.1 data/models/pos2move_v2.1-distill \
        --positions 300 --depth 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
import chess.engine

from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset
from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2

TOK = PostionTokenizer()


def load_model(base_dir: str, device: str):
    base = Path(base_dir)
    config = json.loads((base / "model_config.json").read_text())
    model = Pos2MoveV2(**config)
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.bfloat16().to(device).eval()


@torch.no_grad()
def model_policy_order(model, board, device):
    """Return legal moves ordered by the model's policy (descending)."""
    tok = torch.tensor(TOK.encode(board), dtype=torch.long, device=device).unsqueeze(0)
    player = torch.tensor([int(board.turn)], dtype=torch.long, device=device)
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE): castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE): castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK): castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK): castling |= 8
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
    castling_t = torch.tensor([castling], dtype=torch.long, device=device)
    ep_t = torch.tensor([ep], dtype=torch.long, device=device)
    logits, _ = model(tok, player, castling_t, ep_t)
    logits = logits[0].float().cpu().numpy()
    moves = list(board.legal_moves)
    scores = [logits[m.from_square, move_to_action_plane(m.from_square, m.to_square, m.promotion)] for m in moves]
    return [m for _, m in sorted(zip(scores, moves), key=lambda x: -x[0])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--h5", default="data/elite_db.h5")
    p.add_argument("--positions", type=int, default=300)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--seed", type=int, default=99999, help="held-out sampling seed")
    p.add_argument("--sf-path", default="/usr/games/stockfish")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = HDF5ChessDataset(args.h5, sample_weighting="uniform")
    rng = np.random.default_rng(args.seed)

    # Build a fixed held-out set: (board, sf_best_move, sf_eval_after_best)
    eng = chess.engine.SimpleEngine.popen_uci(args.sf_path)
    eng.configure({"Threads": 1, "Hash": 64})
    positions = []
    while len(positions) < args.positions:
        idx = int(rng.integers(0, len(ds)))
        moves = ds._get_game_moves(ds.valid_game_indices[idx])
        if len(moves) <= 8:
            continue
        ply = int(rng.integers(6, len(moves) - 1))
        b = chess.Board()
        ok = True
        for i in range(ply):
            try:
                b.push(chess.Move.from_uci(ds._decode_move_token(int(moves[i]))))
            except (ValueError, AssertionError):
                ok = False; break
        if not ok or b.is_game_over() or not any(b.legal_moves):
            continue
        info = eng.analyse(b, chess.engine.Limit(depth=args.depth))
        positions.append((b.fen(), info["pv"][0].uci()))

    def eval_after(board, move, depth=10):
        board.push(move)
        sc = -eng.analyse(board, chess.engine.Limit(depth=depth))["score"].relative.score(mate_score=2000)
        board.pop()
        return sc

    print(f"Held-out positions: {len(positions)} (depth {args.depth})\n")
    print(f"{'model':<40}{'top1':>8}{'top3':>8}{'cp_loss':>9}")
    print("-" * 65)
    for mdir in args.models:
        model = load_model(mdir, device)
        top1 = top3 = 0; cp = []
        for fen, sf_best in positions:
            b = chess.Board(fen)
            order = model_policy_order(model, b, device)
            best = order[0].uci()
            top3set = {m.uci() for m in order[:3]}
            top1 += best == sf_best
            top3 += sf_best in top3set
            cp.append(eval_after(b, chess.Move.from_uci(sf_best)) - eval_after(b, order[0]))
        n = len(positions)
        print(f"{Path(mdir).name:<40}{top1/n:>7.1%}{top3/n:>8.1%}{np.mean(cp):>8.0f}c")
        del model
    eng.quit()


if __name__ == "__main__":
    main()
