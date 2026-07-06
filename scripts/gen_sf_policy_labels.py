"""Generate Stockfish soft-policy distillation labels from Lichess positions.

For each sampled position we ask Stockfish for its top-K moves (multipv) and turn
the centipawn evals into a soft target distribution over the AlphaZero 64x73
action planes (softmax over cp with a temperature). These targets feed the
trainer's existing KL-divergence policy loss to distill a sharper policy into
the v2.1 model.

Stockfish is the bottleneck, so work is split across worker processes (each with
its own engine); shards are merged into a single .npz.

Usage
-----
    uv run python scripts/gen_sf_policy_labels.py \
        --h5 data/elite_db.h5 --out data/distill/sf_policy_poc.npz \
        --num 25000 --depth 10 --multipv 8 --workers 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
import chess.engine

from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset
from chesstransformer.models.tokenizer.alphazero_move_encoder import (
    NUM_ACTION_PLANES,
    move_to_action_plane,
)
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer

# Worker globals (initialized once per process).
_DS: HDF5ChessDataset | None = None
_ENG: chess.engine.SimpleEngine | None = None
_TOK = PostionTokenizer()


def _encode_state(board: chess.Board):
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
    return int(board.turn), castling, ep


def _sample_board(rng: np.random.Generator, min_ply: int) -> chess.Board | None:
    """Replay a random game to a random ply and return the board."""
    idx = int(rng.integers(0, len(_DS)))
    game_idx = _DS.valid_game_indices[idx]
    moves = _DS._get_game_moves(game_idx)
    n = len(moves)
    if n <= min_ply + 1:
        return None
    target_ply = int(rng.integers(min_ply, n - 1))
    board = chess.Board()
    for i in range(target_ply):
        try:
            board.push(chess.Move.from_uci(_DS._decode_move_token(int(moves[i]))))
        except (ValueError, AssertionError):
            return None
    if board.is_game_over() or not any(board.legal_moves):
        return None
    return board


def _worker_init(h5: str, sf_path: str, depth: int):
    global _DS, _ENG
    _DS = HDF5ChessDataset(h5, sample_weighting="uniform")
    _ENG = chess.engine.SimpleEngine.popen_uci(sf_path)
    _ENG.configure({"Threads": 1, "Hash": 32})


def _gen_shard(args_tuple):
    """Generate `count` labelled positions in this worker; return arrays."""
    seed, count, depth, multipv, temp, min_ply = args_tuple
    rng = np.random.default_rng(seed)
    tokens, players, castlings, eps = [], [], [], []
    tgt_idx, tgt_prob = [], []

    made = 0
    attempts = 0
    while made < count and attempts < count * 6:
        attempts += 1
        board = _sample_board(rng, min_ply)
        if board is None:
            continue
        try:
            info = _ENG.analyse(
                board, chess.engine.Limit(depth=depth), multipv=multipv
            )
        except chess.engine.EngineError:
            continue

        idxs, cps = [], []
        for line in info:
            pv = line.get("pv")
            if not pv:
                continue
            mv = pv[0]
            plane = move_to_action_plane(mv.from_square, mv.to_square, mv.promotion)
            idxs.append(mv.from_square * NUM_ACTION_PLANES + plane)
            cps.append(float(line["score"].relative.score(mate_score=2000)))
        if not idxs:
            continue

        # Store RAW centipawns (relative to the best move = 0), so the softmax
        # temperature can be chosen at training time without regenerating.
        cps = np.array(cps, dtype=np.float64)
        cps -= cps.max()  # best move -> 0, others <= 0
        k = multipv
        pad_idx = np.full(k, -1, dtype=np.int16)
        pad_cp = np.full(k, -30000.0, dtype=np.float32)  # pad -> softmax ~0
        m = min(len(idxs), k)
        pad_idx[:m] = np.array(idxs[:m], dtype=np.int16)
        pad_cp[:m] = cps[:m].astype(np.float32)

        player, castling, ep = _encode_state(board)
        tokens.append(np.array(_TOK.encode(board), dtype=np.int8))
        players.append(player)
        castlings.append(castling)
        eps.append(ep)
        tgt_idx.append(pad_idx)
        tgt_prob.append(pad_cp)
        made += 1

    if made == 0:
        return None
    return (
        np.stack(tokens), np.array(players, np.int8), np.array(castlings, np.int8),
        np.array(eps, np.int8), np.stack(tgt_idx), np.stack(tgt_prob),
    )


def main():
    import multiprocessing as mp

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default="data/elite_db.h5")
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=25000)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--multipv", type=int, default=8)
    p.add_argument("--temp", type=float, default=100.0, help="cp softmax temperature")
    p.add_argument("--min-ply", type=int, default=6)
    p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    p.add_argument("--sf-path", default="/usr/games/stockfish")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    per = args.num // args.workers
    tasks = [
        (1000 + i, per, args.depth, args.multipv, args.temp, args.min_ply)
        for i in range(args.workers)
    ]
    print(f"Generating ~{per * args.workers} labels across {args.workers} workers "
          f"(depth={args.depth}, multipv={args.multipv}, temp={args.temp}cp)...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.h5, args.sf_path, args.depth),
    ) as pool:
        shards = pool.map(_gen_shard, tasks)

    shards = [s for s in shards if s is not None]
    tokens = np.concatenate([s[0] for s in shards])
    players = np.concatenate([s[1] for s in shards])
    castlings = np.concatenate([s[2] for s in shards])
    eps = np.concatenate([s[3] for s in shards])
    tgt_idx = np.concatenate([s[4] for s in shards])
    tgt_prob = np.concatenate([s[5] for s in shards])

    np.savez_compressed(
        args.out, tokens=tokens, player=players, castling=castlings, ep=eps,
        tgt_idx=tgt_idx, tgt_cp=tgt_prob,  # tgt_prob holds RAW cp (see _gen_shard)
        meta=np.array([args.depth, args.multipv, args.temp], dtype=np.float32),
    )
    print(f"Saved {len(tokens)} labelled positions to {args.out}")


if __name__ == "__main__":
    main()
