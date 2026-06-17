"""Engine-vs-engine match between two ct-bot UCI configs (model and/or node
budget). Both engines are deterministic (move_temp=0), so we vary the start via
an opening book and play each line twice with colours swapped. Reports W/D/L and
an approximate Elo difference (config A relative to config B).
"""

from __future__ import annotations

import math
import sys

import chess
import chess.engine

MAX_PLIES = 200
BASE = "data/models/pos2move_v2.1/model.int8.onnx"  # base (non-EMA) int8

# (label, uci cmd, node budget)
A = ("base@3600", ["rust/target/release/ct-bot", "uci", "--model", BASE], 3600)
B = ("base@1800", ["rust/target/release/ct-bot", "uci", "--model", BASE], 1800)

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6",   # open game
    "e2e4 c7c5",             # Sicilian
    "d2d4 d7d5 c2c4",        # QGD
    "d2d4 g8f6 c2c4 g7g6",   # KID
    "e2e4 e7e6",             # French
    "c2c4 e7e5",             # English
    "g1f3 d7d5 g2g3",        # Reti
    "e2e4 c7c6",             # Caro-Kann
]


def play(white, white_nodes, black, black_nodes, opening: str) -> str:
    board = chess.Board()
    for uci in opening.split():
        board.push(chess.Move.from_uci(uci))
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        eng, nodes = (white, white_nodes) if board.turn == chess.WHITE else (black, black_nodes)
        board.push(eng.play(board, chess.engine.Limit(nodes=nodes)).move)
    return board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"


def main() -> None:
    ea = chess.engine.SimpleEngine.popen_uci(A[1])
    eb = chess.engine.SimpleEngine.popen_uci(B[1])
    wa = wb = d = 0
    a_pts = 0.0
    n = 0
    try:
        for op in OPENINGS:
            for a_white in (True, False):
                if a_white:
                    white, wn, black, bn = ea, A[2], eb, B[2]
                else:
                    white, wn, black, bn = eb, B[2], ea, A[2]
                r = play(white, wn, black, bn, op)
                n += 1
                if r == "1/2-1/2":
                    s, tag, d = 0.5, "=", d + 1
                elif (r == "1-0") == a_white:
                    s, tag, wa = 1.0, "+", wa + 1
                else:
                    s, tag, wb = 0.0, "-", wb + 1
                a_pts += s
                print(f"[{n:2}] {A[0] if a_white else B[0]:10}=W  {op[:11]:11} -> {r:7} ({A[0]} {tag})", flush=True)
    finally:
        ea.quit()
        eb.quit()

    score = a_pts / n
    print(f"\n{A[0]} vs {B[0]}: {wa}W {d}D {wb}L / {n}  |  {A[0]} score {score:.3f}")
    if 0.0 < score < 1.0:
        print(f"{A[0]} - {B[0]} ~ {-400 * math.log10(1 / score - 1):+.0f} Elo")
    else:
        print(f"{A[0]} - {B[0]} = decisive")


if __name__ == "__main__":
    sys.exit(main())
