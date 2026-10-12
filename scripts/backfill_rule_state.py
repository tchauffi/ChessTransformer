#!/usr/bin/env python3
"""Backfill the rule state (halfmove clock, repetition count) of self-play shards written
before SelfPlayEngine recorded it.

For every positions_XXXX.npz in each directory, the games are replayed from games.jsonl
(start FEN + moves) and each recorded position gets:

  rule50      the halfmove clock, capped at 100
  repetition  earlier occurrences of the position in the game, capped at 2

-- the same definition as chess_core::encoding::rule_state, with the game history starting
at the game's start position. Every replayed board is checked against the shard's tokens.
The result goes to rule_state_XXXX.npz next to the shard (the shard is not rewritten);
train_expert_iter.py reads it when the shard has no rule columns.

    uv run python scripts/backfill_rule_state.py data/rl/selfplay/gen01 data/rl/selfplay/gen02
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np

TOK = {(chess.PAWN, True): 1, (chess.KNIGHT, True): 2, (chess.BISHOP, True): 3,
       (chess.ROOK, True): 4, (chess.QUEEN, True): 5, (chess.KING, True): 6,
       (chess.PAWN, False): 7, (chess.KNIGHT, False): 8, (chess.BISHOP, False): 9,
       (chess.ROOK, False): 10, (chess.QUEEN, False): 11, (chess.KING, False): 12}

_GAMES: dict[int, dict] = {}


def tokens(board: chess.Board) -> np.ndarray:
    t = np.zeros(64, np.uint8)
    for sq, p in board.piece_map().items():
        t[sq] = TOK[(p.piece_type, p.color)]
    return t


def game_rule_state(fen: str, moves: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tokens, rule50, repetition) of the position before each move."""
    board = chess.Board(fen)
    seen: dict = {}
    toks, r50, rep = [], [], []
    for u in moves:
        key = board._transposition_key()  # board, turn, castling, legal e.p. (zobrist's inputs)
        toks.append(tokens(board))
        r50.append(min(board.halfmove_clock, 100))
        rep.append(min(seen.get(key, 0), 2))
        seen[key] = seen.get(key, 0) + 1
        board.push_uci(u)
    return np.array(toks, np.uint8), np.array(r50, np.uint8), np.array(rep, np.uint8)


def _init(games_path: str):
    for line in open(games_path):
        g = json.loads(line)
        _GAMES[g["id"]] = g


def backfill_shard(shard: str) -> tuple[str, int, str]:
    d = np.load(shard)
    out = Path(shard).with_name(Path(shard).name.replace("positions_", "rule_state_"))
    if "rule50" in d:
        return shard, 0, "has rule columns"
    gid = d["game_id"]
    rule50 = np.empty(len(gid), np.uint8)
    repetition = np.empty(len(gid), np.uint8)
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    ends = np.r_[starts[1:], len(gid)]
    for a, b in zip(starts, ends):
        g = _GAMES[int(gid[a])]
        toks, r50, rep = game_rule_state(g["fen"], g["moves"])
        if len(toks) != b - a or not np.array_equal(toks, d["boards"][a:b]):
            raise ValueError(f"{shard}: game {gid[a]} does not replay to its recorded positions")
        rule50[a:b], repetition[a:b] = r50, rep
    np.savez_compressed(out, rule50=rule50, repetition=repetition)
    return shard, len(gid), out.name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    for d in args.dirs:
        shards = sorted(str(s) for s in d.glob("positions_*.npz"))
        with Pool(args.workers, initializer=_init, initargs=(str(d / "games.jsonl"),)) as pool:
            n = 0
            for shard, k, note in pool.imap_unordered(backfill_shard, shards):
                n += k
        print(f"{d}: {len(shards)} shards, {n:,} positions backfilled", flush=True)


if __name__ == "__main__":
    sys.exit(main())
