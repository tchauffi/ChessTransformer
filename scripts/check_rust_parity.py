"""Parity check: Rust selfplay_core encodings vs the Python reference.

Plays random games with python-chess and verifies, at every position:
  - board tokens / player / castling / ep  vs  PostionTokenizer + bot encoding
  - (from_square, action_plane) for every legal move  vs  move_to_action_plane
  - the legal move *sets* agree between shakmaty and python-chess

Run after any change to the Rust crate:
    uv run python scripts/check_rust_parity.py --games 200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import selfplay_core

from chesstransformer.models.tokenizer.alphazero_move_encoder import move_to_action_plane
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer

TOK = PostionTokenizer()


def py_encode(board: chess.Board):
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
    return TOK.encode(board), int(board.turn), castling, ep


def check_position(board: chess.Board) -> list[str]:
    errors = []
    fen = board.fen()

    rs_tokens, rs_player, rs_castling, rs_ep = selfplay_core.debug_encode(fen)
    py_tokens, py_player, py_castling, py_ep = py_encode(board)
    if list(rs_tokens) != list(py_tokens):
        errors.append(f"tokens mismatch at {fen}")
    if rs_player != py_player:
        errors.append(f"player mismatch at {fen}: rs={rs_player} py={py_player}")
    if rs_castling != py_castling:
        errors.append(f"castling mismatch at {fen}: rs={rs_castling} py={py_castling}")
    if rs_ep != py_ep:
        errors.append(f"ep mismatch at {fen}: rs={rs_ep} py={py_ep}")

    rs_planes = {u: (f, p) for u, f, p in selfplay_core.debug_move_planes(fen)}
    py_planes = {
        m.uci(): (m.from_square, move_to_action_plane(m.from_square, m.to_square, m.promotion))
        for m in board.legal_moves
    }
    if set(rs_planes) != set(py_planes):
        errors.append(
            f"legal move set mismatch at {fen}: "
            f"rs-only={set(rs_planes) - set(py_planes)} py-only={set(py_planes) - set(rs_planes)}"
        )
    for uci in set(rs_planes) & set(py_planes):
        if rs_planes[uci] != py_planes[uci]:
            errors.append(f"plane mismatch at {fen} {uci}: rs={rs_planes[uci]} py={py_planes[uci]}")
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    positions = 0
    failures: list[str] = []
    for g in range(args.games):
        board = chess.Board()
        while not board.is_game_over(claim_draw=True) and board.ply() < 200:
            failures.extend(check_position(board))
            positions += 1
            board.push(rng.choice(list(board.legal_moves)))
        if failures:
            break

    if failures:
        for e in failures[:10]:
            print(f"FAIL: {e}")
        sys.exit(1)
    print(f"OK: {positions} positions across {args.games} games, all encodings match "
          f"(tokens, player, castling, ep, move planes, legal move sets)")


if __name__ == "__main__":
    main()
