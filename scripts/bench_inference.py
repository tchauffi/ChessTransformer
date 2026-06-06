"""Inference microbenchmark + lossless-regression guard for Pos2MoveV2Bot.

Runs the bot over a fixed suite of positions at a *fixed depth with no time
limit* (so the search is fully deterministic) and reports throughput:

  - mean time / move
  - nodes / sec (alpha-beta nodes)
  - forward calls / sec (approximated by unique positions evaluated = tt misses)

It also records the move chosen per position. Use ``--save-golden`` once to
capture a reference, then ``--check golden.json`` after any *lossless* change
(Track A) to assert the engine still picks the exact same moves.

Usage
-----
    # Baseline: capture golden moves + speed
    uv run python scripts/bench_inference.py data/models/pos2move_v2 \
        --depth 3 --save-golden bench_golden.json

    # After a lossless change: confirm identical moves + see speed delta
    uv run python scripts/bench_inference.py data/models/pos2move_v2 \
        --depth 3 --check bench_golden.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.bots.pos2move_v2_bot import Pos2MoveV2Bot

# Fixed suite spanning openings, middlegames, tactics and endgames so the
# benchmark exercises a representative range of branching factors.
POSITIONS: list[str] = [
    # Start position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Open Italian middlegame
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Sicilian Najdorf
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    # Queen's Gambit Declined
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    # Tactical middlegame (many captures available)
    "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 10",
    # Sharp King's Indian-style centre
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    # Rook + pawn endgame
    "8/5pk1/6p1/8/8/1R4P1/5PK1/r7 w - - 0 40",
    # King and pawn endgame
    "8/8/4k3/8/4P3/4K3/8/8 w - - 0 50",
    # Queen endgame
    "8/8/4k3/8/8/3QK3/8/8 w - - 0 60",
    # Complex middlegame with castling rights
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 9",
]


def run_suite(bot: Pos2MoveV2Bot) -> tuple[list[dict], dict]:
    """Run the bot over the fixed suite. Returns (per-position records, totals)."""
    records: list[dict] = []
    total_time = 0.0
    total_nodes = 0
    total_misses = 0  # unique positions actually forwarded through the model

    for fen in POSITIONS:
        board = chess.Board(fen)
        # tt is shared across calls; misses for *this* call = evals - tt_hits.
        # We clear the tt per position so the count reflects a single search.
        bot._tt.clear()
        t0 = time.perf_counter()
        move, value = bot.predict(board)
        elapsed = time.perf_counter() - t0

        misses = len(bot._tt)  # distinct positions evaluated this search
        total_time += elapsed
        total_nodes += bot.nodes_searched
        total_misses += misses

        records.append(
            {
                "fen": fen,
                "move": move,
                "value": round(float(value), 4),
                "completed_depth": bot.completed_depth,
                "nodes": bot.nodes_searched,
                "forward_evals": misses,
                "tt_hits": bot.tt_hits,
                "time_s": round(elapsed, 4),
            }
        )

    n = len(POSITIONS)
    totals = {
        "positions": n,
        "total_time_s": round(total_time, 3),
        "mean_time_per_move_s": round(total_time / n, 4),
        "total_nodes": total_nodes,
        "nodes_per_sec": round(total_nodes / total_time, 1) if total_time else 0.0,
        "total_forward_evals": total_misses,
        "forward_evals_per_sec": round(total_misses / total_time, 1) if total_time else 0.0,
    }
    return records, totals


def print_totals(totals: dict):
    print("\n" + "=" * 56)
    print("INFERENCE BENCHMARK")
    print("=" * 56)
    print(f"  positions:            {totals['positions']}")
    print(f"  total time:           {totals['total_time_s']:.3f} s")
    print(f"  mean time / move:     {totals['mean_time_per_move_s'] * 1000:.1f} ms")
    print(f"  alpha-beta nodes:     {totals['total_nodes']}")
    print(f"  nodes / sec:          {totals['nodes_per_sec']:.0f}")
    print(f"  forward evals:        {totals['total_forward_evals']}")
    print(f"  forward evals / sec:  {totals['forward_evals_per_sec']:.0f}")
    print("=" * 56)


def check_against_golden(records: list[dict], golden_path: Path) -> bool:
    with open(golden_path) as f:
        golden = json.load(f)
    g_records = {r["fen"]: r for r in golden["records"]}

    ok = True
    print("\nLOSSLESS REGRESSION CHECK (moves must match golden)")
    print("-" * 56)
    for r in records:
        g = g_records.get(r["fen"])
        if g is None:
            print(f"  ?  {r['fen'][:30]}…  (no golden entry)")
            ok = False
            continue
        move_match = r["move"] == g["move"]
        # Value drift is expected under compile numerics; report but don't fail.
        vdrift = abs(r["value"] - g["value"])
        status = "OK " if move_match else "MISMATCH"
        if not move_match:
            ok = False
        line = f"  {status}  {g['move']}"
        if not move_match:
            line += f" -> {r['move']}"
        if vdrift > 1e-3:
            line += f"  (value drift {vdrift:.4f})"
        print(line)
    print("-" * 56)
    print("RESULT:", "PASS — moves identical" if ok else "FAIL — move(s) changed")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "data" / "models" / "pos2move_v2"),
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Full suite warmup passes before timing (default 1)")
    parser.add_argument("--save-golden", default=None, help="Write reference JSON here")
    parser.add_argument("--check", default=None, help="Compare moves against this golden JSON")
    args = parser.parse_args()

    kwargs = dict(
        model_dir=args.model_dir,
        depth=args.depth,
        top_p=args.top_p,
        time_limit=0.0,  # no time limit -> deterministic fixed-depth search
        use_ema=args.ema,
    )
    if args.device:
        kwargs["device"] = args.device
    bot = Pos2MoveV2Bot(**kwargs)

    # Warmup: torch.compile / CUDA graphs / cudnn autotune happen on first runs.
    for _ in range(max(0, args.warmup_runs)):
        run_suite(bot)

    records, totals = run_suite(bot)
    print_totals(totals)

    if args.save_golden:
        out = Path(args.save_golden)
        with open(out, "w") as f:
            json.dump({"totals": totals, "records": records}, f, indent=2)
        print(f"\nGolden reference saved to {out}")

    if args.check:
        ok = check_against_golden(records, Path(args.check))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
