#!/usr/bin/env python3
"""Stage 1 of MCTS parameter tuning: a deterministic screen, not a match.

Playing matches to rank search parameters is statistically hopeless at this budget --
the last 128-game match drew 83 times and still landed a +/-32 Elo interval, so picking
the best of K noisy configs mostly selects noise (winner's curse). This instead runs the
*real* Rust search at a fixed node budget over a fixed position set and scores how often
it finds Stockfish's move, plus how much it loses when it does not. Every position is
deterministic given the parameters, so there is no draw noise and no game-length variance:
far more signal per GPU-minute, at the cost of being a proxy rather than Elo.

Use it to rank configs, then confirm the top one with a real head-to-head. Never quote a
number from here as an Elo gain.

    scripts/h2h_gpu.sh is the env recipe; this script needs the same one. Run it via
    scripts/tune_mcts_gpu.sh so the CUDA libs are on LD_LIBRARY_PATH -- without them ORT
    silently falls back to CPU and every config is scored 20x slower (see ort-cuda notes).

    uv run python scripts/tune_mcts_params.py \
        --models run_023=logs/.../best_model/model.onnx \
                 v2.1=data/models/pos2move_v2.1/model.onnx \
        --c-puct 0.75 1.0 1.5 2.0 --fpu 0.1 0.2 0.4 --positions 200
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_position_set(path: Path, n: int, depth: int, seed: int, sf_path: str, h5: str):
    """Fixed (fen, sf_best) set, cached on disk so every config is scored on the same one."""
    if path.exists():
        cached = json.loads(path.read_text())
        if len(cached["positions"]) >= n and cached["depth"] == depth and cached["seed"] == seed:
            print(f"position set: reusing {path} ({n} of {len(cached['positions'])})")
            return [tuple(p) for p in cached["positions"][:n]]
    import numpy as np
    from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset
    from eval_search_coverage import build_positions
    print(f"position set: building {n} positions at sf depth {depth} (one-off, cached)")
    ds = HDF5ChessDataset(h5, sample_weighting="uniform")
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 1, "Hash": 64})
    pos = build_positions(ds, np.random.default_rng(seed), eng, n, depth)
    eng.quit()
    path.write_text(json.dumps({"depth": depth, "seed": seed, "positions": pos}))
    return pos


class ScoreCache:
    """Stockfish score after a move, shared across configs.

    Configs overwhelmingly agree on the move, so without this the same (fen, move) would be
    re-analysed dozens of times -- the cache is what keeps the Stockfish side off the
    critical path.
    """

    def __init__(self, sf_path: str, depth: int):
        self.depth = depth
        self.lock = threading.Lock()
        self.data: dict[str, float] = {}
        self.eng = chess.engine.SimpleEngine.popen_uci(sf_path)
        self.eng.configure({"Threads": 1, "Hash": 64})

    def score(self, board: chess.Board, move: chess.Move) -> float:
        key = f"{board.fen()}|{move.uci()}"
        with self.lock:
            if key in self.data:
                return self.data[key]
        b = board.copy()
        b.push(move)
        with self.lock:
            info = self.eng.analyse(b, chess.engine.Limit(depth=self.depth))
            val = -info["score"].relative.score(mate_score=2000)
            self.data[key] = val
        return val

    def close(self):
        self.eng.quit()


def run_config(bot, model, nodes, c_puct, fpu, prior_temp, positions, cache, threads):
    cmd = [bot, "uci", "--model", model, "--device", "cuda", "--threads", str(threads),
           "--c-puct", str(c_puct), "--fpu", str(fpu), "--prior-temp", str(prior_temp)]
    eng = chess.engine.SimpleEngine.popen_uci(cmd)
    agree, cps = 0, []
    try:
        for fen, sf_best in positions:
            b = chess.Board(fen)
            res = eng.play(b, chess.engine.Limit(nodes=nodes))
            mv = res.move
            if mv is None:
                continue
            if mv.uci() == sf_best:
                agree += 1
                cps.append(0.0)
            else:
                cps.append(cache.score(b, chess.Move.from_uci(sf_best)) - cache.score(b, mv))
    finally:
        eng.quit()
    n = len(cps) or 1
    return dict(c_puct=c_puct, fpu=fpu, prior_temp=prior_temp,
                agree=agree / n, cp_loss=sum(cps) / n, n=n)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="label=path/to/model.onnx")
    p.add_argument("--bot", default="rust/target-cuda/release/ct-bot")
    p.add_argument("--nodes", type=int, default=1800)
    p.add_argument("--c-puct", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0])
    p.add_argument("--fpu", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    p.add_argument("--prior-temp", type=float, nargs="+", default=[1.0])
    p.add_argument("--positions", type=int, default=200)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--eval-depth", type=int, default=10)
    p.add_argument("--seed", type=int, default=99999)
    p.add_argument("--sf-path", default="/usr/games/stockfish")
    p.add_argument("--h5", default="data/elite_db.h5")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--workers", type=int, default=3, help="configs scored concurrently")
    p.add_argument("--cache", default="data/eval/tune_positions.json")
    p.add_argument("--out", default="data/eval/tune_mcts_results.json")
    args = p.parse_args()

    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    positions = build_position_set(Path(args.cache), args.positions, args.depth,
                                   args.seed, args.sf_path, args.h5)
    grid = [(c, f, t) for c in args.c_puct for f in args.fpu for t in args.prior_temp]
    models = [m.split("=", 1) for m in args.models]
    print(f"{len(positions)} positions x {len(grid)} configs x {len(models)} models "
          f"= {len(positions)*len(grid)*len(models):,} searches @ {args.nodes} nodes\n", flush=True)

    cache = ScoreCache(args.sf_path, args.eval_depth)
    all_rows = {}
    try:
        for label, path in models:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_config, args.bot, path, args.nodes, c, f, t,
                                  positions, cache, args.threads): (c, f, t) for c, f, t in grid}
                rows = []
                for fut in futs:
                    rows.append(fut.result())
                    print(f"  [{label}] {len(rows)}/{len(grid)} configs done", flush=True)
            rows.sort(key=lambda r: (-r["agree"], r["cp_loss"]))
            all_rows[label] = rows
            print(f"\n=== {label} ===")
            print(f"{'c_puct':>7}{'fpu':>6}{'p_temp':>8}{'SF agree':>10}{'cp_loss':>9}")
            for r in rows:
                print(f"{r['c_puct']:>7.2f}{r['fpu']:>6.2f}{r['prior_temp']:>8.2f}"
                      f"{r['agree']:>9.1%}{r['cp_loss']:>8.0f}c")
            base = next((r for r in rows if r["c_puct"] == 1.0 and r["fpu"] == 0.2
                         and r["prior_temp"] == 1.0), None)
            if base:
                best = rows[0]
                print(f"  default (1.0/0.2/1.0): agree {base['agree']:.1%}, cp {base['cp_loss']:.0f}c"
                      f"  ->  best: agree {best['agree']:.1%}, cp {best['cp_loss']:.0f}c")
    finally:
        cache.close()
    Path(args.out).write_text(json.dumps(all_rows, indent=1))
    print(f"\nwrote {args.out}")
    print("PROXY ONLY -- confirm the winner with a real head-to-head before quoting Elo.")


if __name__ == "__main__":
    main()
