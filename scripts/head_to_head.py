"""Engine-vs-engine match between two ct-bot UCI configs.

Both engines are deterministic (move_temp=0), so games vary via a 24-position
opening suite (same FENs as engine_match.py), each line played twice with
colours swapped. Two drive modes:

  --nodes N          fixed sims per move (strength A/B at matched budget)
  --tc SECS+INC      wall-clock game with increment (timeman A/B; engines get
                     wtime/btime/winc/binc and losses on time are counted)

Per-engine UCI options go through --a-opt/--b-opt (repeatable KEY=VAL), e.g.:

  uv run python scripts/head_to_head.py --nodes 3600 \\
      --a-opt TotalVisitsCap=4000 --label-a cap4000 --label-b current
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import chess
import chess.engine

BASE = "data/models/pos2move_v2.1/model.int8.onnx"  # base (non-EMA) int8
MAX_PLIES = 240

# Same 24-position suite as scripts/engine_match.py.
OPENINGS: list[str] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",          # start
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",     # Sicilian
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",     # QP
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",     # QGambit
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",      # Indian
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",  # open game
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # KID/Gruen
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # Slav
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # QGD
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Sicilian Nc6
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",     # Nimzo/QID
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",     # Scandi
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", # Ruy Lopez
    "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3",  # QGD/Indian
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",    # KID
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", # Italian
    "rnbqkbnr/1ppppppp/p7/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",       # 1.e4 a6
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",      # Modern
    "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 2",     # 1.e4 f5
    "rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 2",     # 1.d4 c5
    "rnbqkb1r/pppppppp/5n2/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 2",    # 1.d4 Nf6 2.Nf3
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 2",    # closed Sicilian
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 4 4", # 4 knights-ish
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq - 2 3",  # London
]


class Player:
    """One UCI engine + (in --tc mode) its remaining clock."""

    def __init__(self, label: str, cmd: list[str], opts: dict[str, str]):
        self.label = label
        self.engine = chess.engine.SimpleEngine.popen_uci(cmd)
        if opts:
            self.engine.configure(opts)
        self.clock = 0.0

    def quit(self) -> None:
        self.engine.quit()


def play_nodes(white: Player, black: Player, fen: str, nodes_w: int, nodes_b: int) -> str:
    board = chess.Board(fen)
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        p, nodes = (white, nodes_w) if board.turn == chess.WHITE else (black, nodes_b)
        board.push(p.engine.play(board, chess.engine.Limit(nodes=nodes)).move)
    return board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"


def play_clock(white: Player, black: Player, fen: str, base: float, inc: float) -> tuple[str, str | None]:
    """Returns (result, flagged_label). Clocks are tracked here (wall time),
    the engines only see wtime/btime/winc/binc — exactly the lichess deal."""
    board = chess.Board(fen)
    white.clock = black.clock = base
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        p = white if board.turn == chess.WHITE else black
        limit = chess.engine.Limit(
            white_clock=white.clock, black_clock=black.clock,
            white_inc=inc, black_inc=inc,
        )
        t0 = time.monotonic()
        move = p.engine.play(board, limit).move
        p.clock -= time.monotonic() - t0
        if p.clock < 0:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            return result, p.label
        p.clock += inc
        board.push(move)
    r = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    return r, None


def parse_opts(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        if not v:
            raise SystemExit(f"bad option '{p}', expected KEY=VAL")
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", default="rust/target/release/ct-bot", help="ct-bot binary")
    ap.add_argument("--model-a", default=BASE)
    ap.add_argument("--model-b", default=BASE)
    ap.add_argument("--a-opt", action="append", default=[], help="UCI option KEY=VAL for engine A (repeatable)")
    ap.add_argument("--b-opt", action="append", default=[], help="UCI option KEY=VAL for engine B (repeatable)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--nodes", type=int, help="fixed sims per move (both engines)")
    mode.add_argument("--tc", help="wall-clock 'SECS+INC', e.g. 180+2")
    ap.add_argument("--nodes-a", type=int, help="override A's per-move sims (with --nodes)")
    ap.add_argument("--nodes-b", type=int, help="override B's per-move sims (with --nodes)")
    ap.add_argument("--openings", type=int, default=len(OPENINGS))
    ap.add_argument("--rounds", type=int, default=1, help="repetitions of the full suite")
    args = ap.parse_args()

    a = Player(args.label_a, [args.bot, "uci", "--model", args.model_a], parse_opts(args.a_opt))
    b = Player(args.label_b, [args.bot, "uci", "--model", args.model_b], parse_opts(args.b_opt))
    if args.tc:
        base_s, _, inc_s = args.tc.partition("+")
        tc = (float(base_s), float(inc_s or 0))
        print(f"{a.label} vs {b.label} | tc {tc[0]:.0f}+{tc[1]:g}")
    else:
        nodes_a = args.nodes_a or args.nodes
        nodes_b = args.nodes_b or args.nodes
        print(f"{a.label} vs {b.label} | nodes {nodes_a}/{nodes_b}")

    wa = wb = d = 0
    flags = {a.label: 0, b.label: 0}
    a_pts, n = 0.0, 0
    try:
        for _ in range(args.rounds):
            for fen in OPENINGS[: args.openings]:
                for a_white in (True, False):
                    white, black = (a, b) if a_white else (b, a)
                    if args.tc:
                        r, flagged = play_clock(white, black, fen, *tc)
                        if flagged:
                            flags[flagged] += 1
                    else:
                        nw = nodes_a if a_white else nodes_b
                        nb = nodes_b if a_white else nodes_a
                        r, flagged = play_nodes(white, black, fen, nw, nb), None
                    n += 1
                    if r == "1/2-1/2":
                        s, tag = 0.5, "="
                        d += 1
                    elif (r == "1-0") == a_white:
                        s, tag = 1.0, "+"
                        wa += 1
                    else:
                        s, tag = 0.0, "-"
                        wb += 1
                    a_pts += s
                    fl = f" FLAG:{flagged}" if flagged else ""
                    print(f"[{n:3}] {white.label:>10}=W {fen.split()[0][:18]:18} -> {r:7} ({a.label} {tag}){fl}", flush=True)
    finally:
        a.quit()
        b.quit()

    score = a_pts / n
    print(f"\n{a.label} vs {b.label}: {wa}W {d}D {wb}L / {n}  |  {a.label} score {score:.3f}")
    if args.tc:
        print(f"time losses: {a.label}={flags[a.label]} {b.label}={flags[b.label]}")
    if 0.0 < score < 1.0:
        print(f"{a.label} - {b.label} ~ {-400 * math.log10(1 / score - 1):+.0f} Elo")
    else:
        print(f"{a.label} - {b.label} = decisive")


if __name__ == "__main__":
    sys.exit(main())
