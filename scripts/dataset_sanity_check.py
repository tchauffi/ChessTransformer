"""Sanity check: analyze position distribution in the H5 chess dataset.

Reports:
- Move-number histogram (how often each ply is sampled across games)
- Unique-position count by move-number bucket
- Estimated redundancy ratio (samples per unique position)

Usage:
    python scripts/dataset_sanity_check.py [--data data/elite_db.h5] [--games 5000]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import chess
import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer


def position_key(board: chess.Board) -> str:
    """Compact hashable key for a position (FEN minus halfmove/fullmove counters)."""
    parts = board.fen().split()
    # Keep: piece placement, side to move, castling, ep
    return " ".join(parts[:4])


def analyze(h5_path: str, n_games: int):
    move_tok = MoveTokenizer()

    with h5py.File(h5_path, "r") as f:
        total_games = f["moves"].shape[0]
        n_games = min(n_games, total_games)
        print(f"Analyzing {n_games:,} games (out of {total_games:,} in dataset)")

        all_moves = f["moves"][:n_games]
        num_moves = f["num_moves"][:n_games]

    # Per-move-index sample count and unique-position set
    move_idx_count: Counter[int] = Counter()
    move_idx_unique: dict[int, set[str]] = {}
    pos_count: Counter[str] = Counter()

    skipped_games = 0

    for g_idx in tqdm(range(n_games), desc="Replaying"):
        moves = all_moves[g_idx]
        n = int(num_moves[g_idx])
        board = chess.Board()

        for i in range(n):
            key = position_key(board)
            move_idx_count[i] += 1
            move_idx_unique.setdefault(i, set()).add(key)
            pos_count[key] += 1

            try:
                uci = move_tok.decode(int(moves[i]))
                board.push(chess.Move.from_uci(uci))
            except Exception:
                skipped_games += 1
                break

    total_samples = sum(move_idx_count.values())
    total_unique = len(pos_count)
    print(f"\nTotal positions seen : {total_samples:,}")
    print(f"Unique positions     : {total_unique:,}")
    print(f"Avg samples/unique   : {total_samples / max(1, total_unique):.2f}")
    if skipped_games:
        print(f"Skipped games (errors): {skipped_games}")

    # Bucketed report
    buckets = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 200)]
    print("\n" + "=" * 70)
    print(f"{'Move range':<12} {'Samples':>12} {'Unique':>10} {'Sample%':>8} {'Redund.':>8}")
    print("-" * 70)
    for lo, hi in buckets:
        n_samples = sum(c for i, c in move_idx_count.items() if lo <= i < hi)
        n_unique = len(set().union(*(move_idx_unique[i] for i in move_idx_unique if lo <= i < hi))) if any(lo <= i < hi for i in move_idx_unique) else 0
        pct = 100 * n_samples / max(1, total_samples)
        redund = n_samples / max(1, n_unique)
        print(f"{lo:>3}-{hi:<7} {n_samples:>12,} {n_unique:>10,} {pct:>7.1f}% {redund:>8.2f}")

    # Top redundant positions
    print("\nTop 10 most-repeated positions (likely opening book lines):")
    for key, count in pos_count.most_common(10):
        # Extract just piece placement + side to move for readability
        parts = key.split()
        print(f"  {count:>6}x  {parts[1]} to move  | {parts[0][:40]}...")

    # Dedup gain estimate
    print("\n" + "=" * 70)
    for cap in [1, 2, 3, 5, 10]:
        kept = sum(min(c, cap) for c in pos_count.values())
        ratio = 100 * kept / total_samples
        print(f"Cap K={cap:<3} → keeps {kept:>10,} samples ({ratio:>5.1f}% of original)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/elite_db.h5")
    parser.add_argument("--games", type=int, default=5000, help="Number of games to scan")
    args = parser.parse_args()
    analyze(args.data, args.games)


if __name__ == "__main__":
    main()
