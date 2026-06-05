"""Tune search budget vs Stockfish: sweep alpha-beta depth and MCTS sims.

Reuses ``elo_gauntlet.run_gauntlet`` to play each engine config against a range
of Stockfish skill levels and prints an estimated Elo per config so you can pick
the best strength/time point.

Usage
-----
    uv run python scripts/tune_vs_stockfish.py \
        logs/pos2move_v2/run_021_20260604_130336/checkpoints/best_model \
        --games 8 --skills 0 2 4 6 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.bots.pos2move_v2_bot import Pos2MoveV2Bot
from chesstransformer.bots.pos2move_v2_mcts_bot import Pos2MoveV2MctsBot

# elo_gauntlet lives in the same scripts/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo_gauntlet import run_gauntlet  # noqa: E402


def weighted_elo(results: list[dict]) -> float | None:
    """Replicates elo_gauntlet's weighting: matchups nearer 50% count more."""
    ests, ws = [], []
    for r in results:
        if r.get("est_elo") is None:
            continue
        total = r["wins"] + r["draws"] + r["losses"]
        closeness = 1.0 - abs(r["score"] - 0.5) * 2
        ests.append(r["est_elo"])
        ws.append(total * max(closeness, 0.1))
    if not ests:
        return None
    return sum(e * w for e, w in zip(ests, ws)) / sum(ws)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", help="Model dir (e.g. run21 best_model)")
    p.add_argument("--games", type=int, default=8, help="Games per skill level")
    p.add_argument("--skills", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    p.add_argument("--sf-path", default="/usr/games/stockfish")
    p.add_argument("--sf-time", type=float, default=0.1)
    p.add_argument("--ab-depths", type=int, nargs="*", default=[2, 3, 4])
    p.add_argument("--mcts-sims", type=int, nargs="*", default=[200, 400, 800])
    p.add_argument("--sim-batch", type=int, default=16)
    p.add_argument("--ema", action="store_true")
    args = p.parse_args()

    # (label, factory) — built lazily so a failure in one doesn't block others.
    configs: list[tuple[str, callable]] = []
    for d in args.ab_depths:
        configs.append((
            f"alphabeta d={d} q=4",
            lambda d=d: Pos2MoveV2Bot(model_dir=args.model_dir, depth=d, quiescence_depth=4,
                                      time_limit=0.0, use_ema=args.ema),
        ))
    for s in args.mcts_sims:
        configs.append((
            f"mcts sims={s} batch={args.sim_batch}",
            lambda s=s: Pos2MoveV2MctsBot(model_dir=args.model_dir, num_simulations=s,
                                          sim_batch=args.sim_batch, time_limit=0.0, use_ema=args.ema),
        ))

    summary = []
    for label, factory in configs:
        print(f"\n{'#' * 64}\n# CONFIG: {label}\n{'#' * 64}")
        try:
            bot = factory()
            t0 = time.time()
            results = run_gauntlet(
                bot=bot, stockfish_path=args.sf_path, skill_levels=args.skills,
                games_per_level=args.games, sf_time_limit=args.sf_time, pgn_path=None,
            )
            elo = weighted_elo(results)
            secs = time.time() - t0
            summary.append((label, elo, results, secs))
            del bot
        except Exception as e:  # pragma: no cover
            print(f"CONFIG FAILED: {e}")
            summary.append((label, None, [], 0.0))

    print(f"\n{'=' * 64}\nTUNING SUMMARY (model: {args.model_dir})\n{'=' * 64}")
    print(f"{'config':<28}{'est Elo':>9}{'levels':>9}{'time':>9}")
    print("-" * 64)
    for label, elo, results, secs in sorted(summary, key=lambda x: (x[1] or -1), reverse=True):
        elo_s = f"~{round(elo / 25) * 25}" if elo is not None else "N/A"
        print(f"{label:<28}{elo_s:>9}{len(results):>9}{secs:>8.0f}s")


if __name__ == "__main__":
    main()
