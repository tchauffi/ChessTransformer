"""CPU cost of the MCTS search: the Python search against the Rust one.

The same PUCT search runs in both languages: c_puct 1, FPU 0.2, one leaf per
simulation, tree reuse and a cache of evaluated positions. Python is the bot
our first self-play ran on (Pos2MoveV2MctsBot, on python-chess); Rust is the
self-play engine (selfplay_core.SelfPlayEngine). Both play games from the start
position on ONE CPU thread, and the network is replaced by a stub that hands
back pre-drawn random policy logits and values. What is timed is the search
alone: move generation, selection, encoding each leaf for the network,
expansion and backup.

The stub costs almost nothing, so both numbers are upper bounds on how many
simulations per second one CPU core can run. The Rust figure includes the
Python glue that hands its batches to the stub; the Python figure leaves out the
copy of each position to the GPU.

The comparison is per simulation, not per evaluation: the Python game often
falls into repetitions whose positions are already cached, and a cached leaf
skips the encoding, so Python's evaluations per simulation run lower and its
simulations cheaper. That favours Python.

Usage:
    .venv/bin/python scripts/bench_search_cpu.py [--sims 800] [--seconds 20]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import chess  # noqa: E402
import chess.polyglot  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.bots.pos2move_v2_mcts_bot import Pos2MoveV2MctsBot  # noqa: E402
from chesstransformer.models.tokenizer import PostionTokenizer  # noqa: E402

START = chess.STARTING_FEN
BANK = 1024  # pre-drawn stub outputs, used in turn
MAX_PLIES = 300


class StubMctsBot(Pos2MoveV2MctsBot):
    """The Python MCTS bot with its network replaced by the stub. Only the search's
    own state is set up: no model, no device buffers."""

    def __init__(self, sims: int, rng: np.random.Generator):
        self.num_simulations = sims
        self.c_puct, self.prior_temp, self.fpu = 1.0, 1.0, 0.2
        self.sim_batch, self.tree_reuse = 1, True
        self.move_temp, self.move_temp_plies = 0.0, 0
        self._reuse_root, self._reuse_moves = None, None
        self.position_tokenizer = PostionTokenizer()
        self._tt_size, self._tt = 500_000, {}
        self.nodes_searched = self.tt_hits = self.completed_depth = 0
        self.evals = 0
        self._logits = rng.standard_normal((BANK, 64, 73)).astype(np.float32)
        self._values = rng.uniform(-0.5, 0.5, BANK).astype(np.float32)

    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """Pos2MoveV2Bot._evaluate with the forward pass replaced by the stub."""
        key = chess.polyglot.zobrist_hash(board)
        cached = self._tt.get(key)
        if cached is not None:
            self.tt_hits += 1
            return cached
        self._encode_position(board)  # the network's input: tokens, player, castling, ep
        i = self.evals % BANK
        self.evals += 1
        out = (self._logits[i], float(self._values[i]))
        if len(self._tt) >= self._tt_size:
            del self._tt[next(iter(self._tt))]
        self._tt[key] = out
        return out


def bench_python(sims: int, seconds: float, seed: int) -> dict:
    bot = StubMctsBot(sims, np.random.default_rng(seed))
    moves = 0
    board = chess.Board()
    t0 = time.process_time()
    while time.process_time() - t0 < seconds:
        if bot._terminal_value(board) is not None or len(board.move_stack) >= MAX_PLIES:
            board = chess.Board()
        mv, _ = bot.predict(board)
        board.push_uci(mv)
        moves += 1
    cpu = time.process_time() - t0
    return {"moves": moves, "sims": moves * sims, "evals": bot.evals, "cpu_s": cpu}


def bench_rust(sims: int, seconds: float, seed: int, games: int, threads: int) -> dict:
    import selfplay_core

    cfg = {"num_sims": sims, "num_parallel_games": games, "sim_batch": 1, "threads": threads,
           "c_puct": 1.0, "fpu": 0.2, "dirichlet_eps": 0.0, "max_plies": MAX_PLIES}
    eng = selfplay_core.SelfPlayEngine(json.dumps(cfg), [START], seed, 10**9)
    rng = np.random.default_rng(seed)
    banks: dict[int, np.ndarray] = {}  # width -> (BANK, width) contiguous logits
    values = rng.uniform(-0.5, 0.5, BANK).astype(np.float32)

    def totals() -> tuple[int, int, int]:
        st = eng.search_stats()
        return (sum(d["moves"] for d in st), sum(d["sims"] for d in st),
                sum(d["nn_evals"] for d in st))

    def step():
        _, player, _, _, legal = eng.collect()
        eng.drain_finished()
        n, width = legal.shape
        if width not in banks:
            banks[width] = rng.standard_normal((BANK, width)).astype(np.float32)
        eng.apply_legal(banks[width][:n], values[:n])

    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < 2.0:  # games spread out over their plies
        step()
    m0, s0, e0 = totals()
    c0, w0 = time.process_time(), time.perf_counter()
    while time.perf_counter() - w0 < seconds:
        step()
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    m1, s1, e1 = totals()
    return {"moves": m1 - m0, "sims": s1 - s0, "evals": e1 - e0, "cpu_s": cpu, "wall_s": wall}


def report(name: str, r: dict, clock: str) -> float:
    t = r[clock]
    print(f"{name:<22} {r['sims'] / t:>10,.0f} sims/s {1e6 * t / r['sims']:>8.1f} us/sim "
          f"{r['evals'] / r['sims']:>6.2f} evals/sim  ({r['moves']} moves, {t:.1f} s {clock})")
    return r["sims"] / t


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sims", type=int, default=800, help="simulations per move")
    p.add_argument("--seconds", type=float, default=20.0, help="measured time per engine")
    p.add_argument("--games", type=int, default=256, help="Rust: games in flight")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 6],
                   help="Rust: search threads to measure (Python always uses one)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(1)

    print(f"{args.sims} simulations per move, network stubbed out\n")
    py = report("Python, 1 thread", bench_python(args.sims, args.seconds, args.seed), "cpu_s")
    for th in args.threads:
        r = bench_rust(args.sims, args.seconds, args.seed, args.games, th)
        rs = report(f"Rust, {th} thread{'s' * (th > 1)}", r, "cpu_s" if th == 1 else "wall_s")
        print(f"{'':<22} x{rs / py:.0f} the simulations per second of Python")


if __name__ == "__main__":
    main()
