"""Measure how Pos2MoveV2 strength scales with MCTS simulations, and plot it.

For each simulation budget we run the Stockfish gauntlet and fit a single MLE Elo
over all games (the same estimator the README uses).

Robustness: each sim level runs in an **isolated subprocess with a timeout**, so a
wedged Stockfish UCI process (which can block ``engine.play`` indefinitely) only
kills that one level instead of freezing the whole sweep.

Outputs:
  - docs/strength_vs_sims.json  (raw points, reproducible)
  - docs/strength_vs_sims.png   (Elo vs sims curve for the README)

CPU-friendly: forces CUDA_VISIBLE_DEVICES="" so a training GPU is untouched.

Usage
-----
    uv run python scripts/sims_scaling.py --sims 25 100 200 400 800 --skills 4 6 8 10 12 --games 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # elo_gauntlet, tune_vs_stockfish
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run_one_level(args) -> None:
    """Single sim level: run the gauntlet, print ``RESULT_ELO=<n>``."""
    from elo_gauntlet import run_gauntlet
    from tune_vs_stockfish import mle_elo

    if args.engine_cmd:
        import shlex

        from uci_bot_adapter import UciBotAdapter

        bot = UciBotAdapter(shlex.split(args.engine_cmd), sims=args.sims[0])
    else:
        from chesstransformer.bots.pos2move_v2_mcts_bot import Pos2MoveV2MctsBot

        use_compile = args.device == "cuda"
        bot = Pos2MoveV2MctsBot(model_dir=args.model, num_simulations=args.sims[0],
                                c_puct=args.cpuct, fpu=args.fpu, device=args.device,
                                compile=use_compile, time_limit=0.0, move_temp=0.0)
    try:
        results = run_gauntlet(bot, args.sf_path, args.skills, args.games, args.sf_time, None)
    finally:
        if args.engine_cmd:
            bot.close()
    elo = mle_elo(results)
    print(f"RESULT_ELO={elo if elo is not None else ''}", flush=True)


def plot(points, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    xs = [p["sims"] for p in points if p["elo"] is not None]
    ys = [p["elo"] for p in points if p["elo"] is not None]

    fig, ax = plt.subplots(figsize=(7, 4.3), dpi=130)
    ax.plot(xs, ys, "-o", color="#2b6cb0", lw=2, ms=6, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{round(y)}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#2b6cb0")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("MCTS simulations per move")
    ax.set_ylabel("Estimated Elo (vs Stockfish, MLE)")
    ax.set_title("ChessTransformer v2.1 — strength scales with search")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.out_png)
    print(f"Saved plot: {args.out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/models/pos2move_v2.1")
    p.add_argument("--sims", type=int, nargs="+", default=[25, 100, 200, 400, 800])
    p.add_argument("--skills", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    p.add_argument("--games", type=int, default=5, help="Games per skill level")
    p.add_argument("--cpuct", type=float, default=1.0)
    p.add_argument("--fpu", type=float, default=0.2)
    p.add_argument("--sf-path", default="/usr/games/stockfish")
    p.add_argument("--sf-time", type=float, default=0.1)
    p.add_argument("--level-timeout", type=int, default=600, help="Per-sim-level wall-clock cap (s)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="cuda uses torch.compile for fast, clean measurement (run when GPU is free)")
    p.add_argument("--engine-cmd", default=None,
                   help="UCI engine command to test instead of the Python bot, e.g. "
                        "'rust/target/release/ct-bot uci --model data/models/pos2move_v2.1/model.int8.onnx'. "
                        "Raise --level-timeout for high sim counts.")
    p.add_argument("--out-json", default="docs/strength_vs_sims.json")
    p.add_argument("--out-png", default="docs/strength_vs_sims.png")
    p.add_argument("--_one", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._one:
        run_one_level(args)
        return

    # On CPU we hide the GPU so torch stays on CPU; on cuda we leave it visible.
    env = dict(os.environ)
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    points = []
    for s in sorted(args.sims):
        cmd = [sys.executable, "-u", __file__, "--_one", "--sims", str(s),
               "--skills", *map(str, args.skills), "--games", str(args.games),
               "--model", args.model, "--cpuct", str(args.cpuct), "--fpu", str(args.fpu),
               "--sf-path", args.sf_path, "--sf-time", str(args.sf_time),
               "--device", args.device]
        if args.engine_cmd:
            cmd += ["--engine-cmd", args.engine_cmd]
        print(f"\n########## MCTS sims = {s} (timeout {args.level_timeout}s) ##########", flush=True)
        elo = None
        try:
            r = subprocess.run(cmd, env=env, timeout=args.level_timeout,
                               capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.startswith("RESULT_ELO="):
                    v = line.split("=", 1)[1].strip()
                    elo = float(v) if v else None
            if elo is None:
                print(f"  (no Elo parsed; stderr tail: {r.stderr.strip()[-200:]})", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  !! sims={s} timed out after {args.level_timeout}s — skipping", flush=True)
        print(f">>> sims={s}: MLE Elo ~{round(elo) if elo else 'N/A'}", flush=True)
        points.append({"sims": s, "elo": elo})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "skills": args.skills, "games_per_level": args.games,
        "config": {"c_puct": args.cpuct, "fpu": args.fpu, "model": args.model},
        "points": points,
    }, indent=2))
    print(f"\nSaved data: {args.out_json}")
    plot(points, args)


if __name__ == "__main__":
    main()
