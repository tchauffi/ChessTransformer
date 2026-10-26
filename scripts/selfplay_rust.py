"""Self-play generation driven by the Rust core (rust/selfplay-core).

Writes npz shards + games.jsonl, each position carrying the eventual game
outcome plus **MCTS visit distributions** in CSR form (visit_idx/visit_cnt/
visit_ptr, indices = from_square * 73 + action_plane) — the value and policy
targets needed for expert-iteration training.

The Rust engine runs N games concurrently and batches leaf evaluations across
all of them; Python only runs the batched NN forward. Games are searched with
the deployed bot's own search (``SelfPlayEngine`` on ``chess_core::search``:
deferred TT hits, the tree reused from move to move), so the teacher is the
search the network is judged and deployed with -- but one leaf per wave by
default where the bot sends 16: the GPU batch comes from the games instead, and
the plain sequential search beat the 16-leaf waves by +49 Elo at 800 sims for
the same evaluations (2026-10-25). ``--legacy-engine`` selects the older
self-play search (``Engine``).

One process drives one GPU from a single Python thread, which a datacenter GPU outruns:
``--workers N`` starts N copies of this script on the same output directory (spread over
``--gpus`` round-robin, or all on the visible GPU; with several per GPU, run NVIDIA MPS so
their kernels overlap) and merges their progress into the usual lines. ``--games`` and
``--parallel-games`` are totals, split between the workers. Every worker plays its own
seed, appends to the shared games.jsonl with interleaved game ids (worker k of N owns the
ids = k mod N above the directory's existing ones) and writes its own
positions_w<k>_NNNN.npz shards; one worker writes positions_NNNN.npz as before. A rerun
resumes, as a single process does. Build the core with:

    uvx maturin build --release -m rust/selfplay-core/Cargo.toml -o dist
    uv pip install --force-reinstall dist/selfplay_core-*.whl

Usage
-----
    uv run python scripts/selfplay_rust.py \
        --model data/models/pos2move_v2.1 --out data/selfplay/v2.1-rust \
        --games 2000 --sims 128
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import selfplay_core
from engine_match import OPENINGS

from grpo_puzzles import load_model

_BUCKETS = (32, 64, 128, 256, 512, 1024)

# torch.compile mode on CUDA. Both capture CUDA graphs; "max-autotune" also benchmarks Triton
# GEMM templates per shape and fuses their epilogues: +16% forward throughput on the RTX 5070
# Ti (champion, 128 rows, 2026-10-25), for a longer first compile (cached by inductor).
COMPILE_MODES = ("max-autotune", "reduce-overhead")


class Forward:
    """Batched bf16 forward with bucketed static shapes (compile-friendly)."""

    def __init__(self, model_dir: str, device: str, use_ema: bool, compile_model: bool,
                 buckets: tuple[int, ...] = _BUCKETS, compile_mode: str = COMPILE_MODES[0]):
        self.device = device
        # Any CUDA device ("cuda", "cuda:2"): several Forwards can share one process, each
        # on its own card (gpu_match --gpus).
        self.cuda = torch.device(device).type == "cuda"
        self.buckets = buckets
        self.model = load_model(Path(model_dir), device).bfloat16().eval()
        # Search backs up a scalar; a WDL head is reduced to p(win) - p(loss).
        self.value_scalar = self.model.value_scalar
        if use_ema:
            ema = torch.load(Path(model_dir) / "ema_state.pt", map_location="cpu", weights_only=True)
            ema = {k.replace("_orig_mod.", ""): v for k, v in ema.items()}
            self.model.load_state_dict(ema, strict=False)
            self.model = self.model.bfloat16()
            print(f"EMA weights loaded ({len(ema)} tensors)")
        if compile_model and self.cuda:
            self.model = torch.compile(self.model, mode=compile_mode)
        self.bufs = {}
        self.slots = {}  # async staging buffers, one set per in-flight batch
        # Warm up every bucket so CUDA-graph capture is up front -- under no_grad, as every
        # later call runs: in grad mode the warm-up kept every activation (7.5 GB on CPU at
        # batch 256) and compiled graphs that the no_grad calls then compiled again.
        with torch.no_grad(), self._on():
            for b in buckets:
                self._buffers(b)
                bt, pl, ca, ep = self.bufs[b]
                for _ in range(3):
                    self.model(bt, pl, ca, ep)
        if self.cuda:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()  # the eager warm-up runs' blocks, not the graphs' pool

    def _on(self):
        """This Forward's card as the current device: CUDA graphs, streams and events
        belong to the current device."""
        return torch.cuda.device(self.device) if self.cuda else contextlib.nullcontext()

    def _buffers(self, bucket: int):
        if bucket not in self.bufs:
            self.bufs[bucket] = (
                torch.zeros(bucket, 64, dtype=torch.long, device=self.device),
                torch.zeros(bucket, dtype=torch.long, device=self.device),
                torch.zeros(bucket, dtype=torch.long, device=self.device),
                torch.zeros(bucket, dtype=torch.long, device=self.device),
            )
        return self.bufs[bucket]

    @torch.no_grad()
    def __call__(self, boards: np.ndarray, player: np.ndarray, castling: np.ndarray,
                 ep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(player)
        bucket = next((b for b in self.buckets if n <= b), self.buckets[-1])
        if n > self.buckets[-1]:  # split oversized batches
            l1, v1 = self(boards[:bucket], player[:bucket], castling[:bucket], ep[:bucket])
            l2, v2 = self(boards[bucket:], player[bucket:], castling[bucket:], ep[bucket:])
            return np.concatenate([l1, l2]), np.concatenate([v1, v2])
        with self._on():
            return self._call(n, bucket, boards, player, castling, ep)

    def _call(self, n, bucket, boards, player, castling, ep):
        bt, pl, ca, e = self._buffers(bucket)
        bt.zero_(); pl.zero_(); ca.zero_(); e.zero_()
        bt[:n] = torch.from_numpy(boards.astype(np.int64)).to(self.device)
        pl[:n] = torch.from_numpy(player.astype(np.int64)).to(self.device)
        ca[:n] = torch.from_numpy(castling.astype(np.int64)).to(self.device)
        e[:n] = torch.from_numpy(ep.astype(np.int64)).to(self.device)
        logits, value = self.model(bt, pl, ca, e)
        return (
            logits[:n].float().cpu().numpy(),
            self.value_scalar(value[:n].float()).squeeze(-1).cpu().numpy(),
        )

    # ── Asynchronous path: legal-move logits only, CPU/GPU overlap ────────────
    #
    # launch() queues a batch and returns at once: inputs go to the GPU through
    # pinned buffers, the legal moves' logits are gathered on the GPU (a few
    # dozen floats per position instead of 4672) and come back through pinned
    # buffers behind an event. wait() blocks on that event only. With one slot
    # per engine, the CPU searches one engine's games while the GPU evaluates
    # the other's. Every `verify_every`-th batch is recomputed synchronously
    # and compared, so a stream-ordering mistake fails loudly instead of
    # feeding wrong evaluations to the search.

    verify_every = 50
    launched = 0

    def _slot(self, slot: int, n: int, width: int) -> dict:
        st = self.slots.get(slot)
        if st is None or st["cap"] < n or st["wcap"] < width:
            cap = max(n, st["cap"] if st else 0, 64)
            wcap = max(width, st["wcap"] if st else 0, 64)
            pin = self.cuda

            def host(size, dtype):
                return torch.zeros(size, dtype=dtype, pin_memory=pin)

            def dev(size, dtype):
                return torch.zeros(size, dtype=dtype, device=self.device)

            st = {"cap": cap, "wcap": wcap,
                  "event": torch.cuda.Event() if self.cuda else None,
                  "h_in": [host((cap, 64), torch.uint8)] + [host(cap, torch.uint8) for _ in range(3)],
                  "d_in": [dev((cap, 64), torch.uint8)] + [dev(cap, torch.uint8) for _ in range(3)],
                  "h_legal": host(cap * wcap, torch.int16), "d_legal": dev(cap * wcap, torch.int16),
                  "h_logits": host(cap * wcap, torch.float32), "h_values": host(cap, torch.float32)}
            self.slots[slot] = st
        return st

    @torch.no_grad()
    def launch(self, slot: int, boards: np.ndarray, player: np.ndarray, castling: np.ndarray,
               ep: np.ndarray, legal: np.ndarray):
        """Queue a forward; `wait(handle)` returns (legal logits (n, width), values (n,)).
        `legal` holds flat logit indices padded with 4672 (a zero column is appended)."""
        with self._on():
            return self._launch(slot, boards, player, castling, ep, legal)

    def _launch(self, slot, boards, player, castling, ep, legal):
        n, width = len(player), legal.shape[1]
        st = self._slot(slot, n, width)
        for h, d, a in zip(st["h_in"], st["d_in"], (boards, player, castling, ep)):
            h[:n].numpy()[...] = a
            d[:n].copy_(h[:n], non_blocking=True)
        h_legal = st["h_legal"][: n * width].view(n, width)
        h_legal.numpy()[...] = legal
        d_legal = st["d_legal"][: n * width].view(n, width)
        d_legal.copy_(h_legal, non_blocking=True)
        out = st["h_logits"][: n * width].view(n, width)
        out_v = st["h_values"][:n]
        top = self.buckets[-1]
        for s0 in range(0, n, top):
            m = min(top, n - s0)
            bucket = next(b for b in self.buckets if m <= b)
            bufs = self._buffers(bucket)
            for buf, d in zip(bufs, st["d_in"]):
                buf[:m].copy_(d[s0:s0 + m])
                buf[m:].zero_()
            logits, value = self.model(*bufs)
            flat = torch.cat([logits[:m].reshape(m, -1), logits.new_zeros(m, 1)], dim=1)
            out[s0:s0 + m].copy_(flat.gather(1, d_legal[s0:s0 + m].long()).float(), non_blocking=True)
            out_v[s0:s0 + m].copy_(self.value_scalar(value[:m].float()).squeeze(-1), non_blocking=True)
        if st["event"] is not None:
            st["event"].record()
        return slot, n, width, (boards, player, castling, ep, legal)

    def wait(self, handle) -> tuple[np.ndarray, np.ndarray]:
        """The batch's results, valid until the slot's next launch."""
        slot, n, width, inputs = handle
        st = self.slots[slot]
        if st["event"] is not None:
            st["event"].synchronize()
        logits = st["h_logits"][: n * width].view(n, width).numpy()
        values = st["h_values"][:n].numpy()
        self.launched += 1
        if self.verify_every and self.launched % self.verify_every == 0:
            self._verify(inputs, logits, values)
        return logits, values

    def _verify(self, inputs, logits: np.ndarray, values: np.ndarray):
        boards, player, castling, ep, legal = inputs
        full, ref_v = self(boards, player, castling, ep)
        full = np.concatenate([full.reshape(len(full), 64 * 73), np.zeros((len(full), 1), np.float32)], 1)
        ref = np.take_along_axis(full, legal.astype(np.int64), axis=1)
        if not (np.allclose(logits, ref, rtol=0, atol=1e-2) and np.allclose(values, ref_v, rtol=0, atol=1e-3)):
            raise RuntimeError(
                f"async forward check failed: max |dlogit| {np.abs(logits - ref).max():.4f}, "
                f"max |dvalue| {np.abs(values - ref_v).max():.4f}")


class ShardWriter:
    """npz shards with the selfplay_value_games.py columns + CSR visit dists."""

    SCALAR_COLS = ("player", "castling", "ep", "halfmove", "z", "root_v", "full")
    # The root's search over every legal move (SelfPlayEngine only), for completed-Q
    # (Gumbel) policy targets: CSR like the visits, plus the network's raw root value.
    # A shard keeps them when every game has them.
    LEGAL_COLS = {"legal_idx": np.uint16, "legal_prior": np.float16, "legal_n": np.uint16,
                  "legal_q": np.float16}

    def __init__(self, out_dir: Path, flush_positions: int = 10_000, prefix: str = "positions_"):
        self.out_dir = out_dir
        self.flush_positions = flush_positions
        self.prefix = prefix  # "positions_w<k>_" for a worker of a --workers run
        self.shard_idx = len(list(out_dir.glob(f"{prefix}[0-9]*.npz")))
        self._reset()

    def _reset(self):
        self.cols = {c: [] for c in self.SCALAR_COLS}
        self.boards, self.game_id = [], []
        self.visit_idx, self.visit_cnt = [], []
        self.visit_lens = []
        self.legal = {c: [] for c in self.LEGAL_COLS}
        self.legal_lens, self.root_net_v = [], []
        self.n = 0

    def add_game(self, g: dict, game_id: int):
        p = len(g["z"])
        self.boards.append(np.asarray(g["boards"], dtype=np.uint8))
        for c in self.SCALAR_COLS:
            self.cols[c].append(np.asarray(g[c]))
        self.game_id.append(np.full(p, game_id, dtype=np.uint32))
        self.visit_idx.append(np.asarray(g["visit_idx"], dtype=np.uint16))
        self.visit_cnt.append(np.asarray(g["visit_cnt"], dtype=np.uint16))
        ptr = np.asarray(g["visit_ptr"], dtype=np.int64)
        self.visit_lens.append(np.diff(ptr))
        if "legal_ptr" in g:
            for c, t in self.LEGAL_COLS.items():
                self.legal[c].append(np.asarray(g[c]).astype(t))
            self.legal_lens.append(np.diff(np.asarray(g["legal_ptr"], dtype=np.int64)))
            self.root_net_v.append(np.asarray(g["root_net_v"], dtype=np.float32))
        self.n += p
        if self.n >= self.flush_positions:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        lens = np.concatenate(self.visit_lens)
        ptr = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=ptr[1:])
        arrays = {
            "boards": np.concatenate(self.boards),
            "game_id": np.concatenate(self.game_id),
            "visit_idx": np.concatenate(self.visit_idx),
            "visit_cnt": np.concatenate(self.visit_cnt),
            "visit_ptr": ptr,
        }
        casts = {"player": np.uint8, "castling": np.uint8, "ep": np.uint8,
                 "halfmove": np.uint16, "z": np.float32, "root_v": np.float32, "full": np.uint8}
        for c in self.SCALAR_COLS:
            arrays[c] = np.concatenate(self.cols[c]).astype(casts[c])
        if len(self.legal_lens) == len(self.game_id):
            for c in self.LEGAL_COLS:
                arrays[c] = np.concatenate(self.legal[c])
            legal_lens = np.concatenate(self.legal_lens)
            arrays["legal_ptr"] = np.zeros(len(legal_lens) + 1, dtype=np.int64)
            np.cumsum(legal_lens, out=arrays["legal_ptr"][1:])
            arrays["root_net_v"] = np.concatenate(self.root_net_v)
        path = self.out_dir / f"{self.prefix}{self.shard_idx:04d}.npz"
        np.savez_compressed(path, **arrays)
        print(f"  wrote {path.name} ({self.n} positions)", flush=True)
        self.shard_idx += 1
        self._reset()


def append_line(fd: int, line: str) -> None:
    """One write() per line: with O_APPEND, workers sharing games.jsonl keep lines whole."""
    data = line.encode()
    while data:
        data = data[os.write(fd, data):]


PROGRESS = re.compile(r"^(\d+)/\d+ games \| (\d+) pos \| (\d+) games/h \| tt=(\d+)")
DONE = re.compile(r"^Done: (\d+) games, (\d+) positions \| W (\d+) / D (\d+) / B (\d+) \| "
                  r"([\d.]+) min \((\d+) games/h\)")


def run_workers(args: argparse.Namespace, id_offset: int) -> None:
    """Run args.workers copies of this script on args.out and print their progress as one
    run does: a combined games / positions / rate line on every worker update, other lines
    tagged [w<k>]. A failing worker stops the others, so a rerun resumes them together."""
    gpus = [g for g in args.gpus.split(",") if g] or [None]
    argv = sys.argv[1:] + ["--id-offset", str(id_offset)]
    procs = []
    for k in range(args.workers):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if gpus[k % len(gpus)] is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpus[k % len(gpus)]
        procs.append(subprocess.Popen([sys.executable, __file__, *argv, "--worker", str(k)], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    lines: queue.Queue = queue.Queue()

    def pump(k: int, proc: subprocess.Popen):
        for line in proc.stdout:
            lines.put((k, line.rstrip("\n")))
        lines.put((k, None))

    for k, proc in enumerate(procs):
        threading.Thread(target=pump, args=(k, proc), daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    progress = [(0, 0, 0, 0)] * args.workers
    done = [None] * args.workers
    running = args.workers
    try:
        while running:
            k, line = lines.get()
            if line is None:
                running -= 1
                if procs[k].wait() != 0:
                    print(f"[w{k}] exited with code {procs[k].returncode}; stopping the other workers",
                          flush=True)
                    sys.exit(1)
            elif m := PROGRESS.match(line):
                progress[k] = tuple(map(int, m.groups()))
                g, pos, rate, tt = (sum(c) for c in zip(*progress))
                print(f"{g}/{args.games} games | {pos} pos | {rate} games/h | tt={tt}", flush=True)
            elif m := DONE.match(line):
                done[k] = m.groups()
            elif line.strip():
                print(f"[w{k}] {line}", flush=True)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            proc.wait()
    done = [d for d in done if d]
    g, pos, w, d, b = (sum(int(x[i]) for x in done) for i in range(5))
    minutes = max((float(x[5]) for x in done), default=0.0)
    print(f"\nDone: {g} games, {pos} positions | W {w} / D {d} / B {b} | "
          f"{minutes:.1f} min ({sum(int(x[6]) for x in done)} games/h)")


def selfplay_openings(args) -> list[str]:
    """The start positions games draw from, uniformly: the shallow built-in openings (the
    start position first), and with --book that book's openings past --book-skip, the
    shallow list repeated so that --book-share of the games start from the book."""
    shallow = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] + list(OPENINGS)
    if args.book is None or args.book_share == 0.0:
        return shallow
    book = [o["fen"] for o in json.loads(Path(args.book).read_text())["openings"]][args.book_skip:]
    if not book:
        raise SystemExit(f"{args.book}: no openings past the first {args.book_skip}")
    if args.book_share == 1.0:
        return book
    reps = max(1, round((1.0 - args.book_share) / args.book_share * len(book) / len(shallow)))
    if args.worker in (None, 0):
        print(f"Openings: {len(book)} from {args.book} (#{args.book_skip} on) on "
              f"{len(book) / (len(book) + reps * len(shallow)):.0%} of games, "
              f"{len(shallow)} shallow ones on the rest", flush=True)
    return shallow * reps + book


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="data/models/pos2move_v2.1")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=128,
                   help="search budget of a full move (the policy targets)")
    p.add_argument("--fast-sims", type=int, default=0,
                   help="playout cap randomization: budget of a fast move (0 = every move is full)")
    p.add_argument("--full-frac", type=float, default=1.0,
                   help="probability that a move is a full search when --fast-sims > 0")
    p.add_argument("--cap-total", choices=("none", "fast", "all"), default="none",
                   help="budgets that count the reused tree's warm visits (the root's total "
                        "visits) instead of fresh sims: none, fast moves only, or all moves; "
                        "not --legacy-engine, which has no tree reuse")
    p.add_argument("--forced-playouts", type=float, default=0.0,
                   help="KataGo forced playouts + policy target pruning on full searches: k "
                        "(2 in KataGo; 0 = off). Not --legacy-engine")
    p.add_argument("--fresh-full", action="store_true",
                   help="full searches start from a fresh tree instead of the reused one, whose "
                        "warm visits got no root noise or forced playouts. Not --legacy-engine")
    p.add_argument("--twofold-history", action="store_true",
                   help="Lc0's two-fold rule over the game history too: a search position that "
                        "repeats any earlier one of the game is a draw. Not --legacy-engine")
    p.add_argument("--draw-contempt", type=float, default=0.2,
                   help="contempt for repetition and 50-move draws: the side the network calls "
                        "ahead at the root (value v) scores them -C * min(|v| / --contempt-ramp, 1) "
                        "wherever it moves in the search, so it plays on rather than repeat; the "
                        "other side's stay 0. Recorded values are unchanged. On gen14, 0.2 cut the "
                        "threefolds from 43%% to 36%% of games at the same Stockfish move quality. "
                        "0 = off. Not --legacy-engine")
    p.add_argument("--contempt-ramp", type=float, default=0.3,
                   help="root value at which --draw-contempt applies in full")
    p.add_argument("--depth-stats", action="store_true",
                   help="with --report-every, also report the trees' depth per move (a walk of the "
                        "tree after every search; benchmarks). Not --legacy-engine")
    p.add_argument("--book", default=None,
                   help="also start games from this book's openings (gpu_match's format, e.g. "
                        "data/openings/book2k.json), mixed with the shallow built-in ones")
    p.add_argument("--book-skip", type=int, default=500,
                   help="leave the book's first N openings out: the gates and anchor matches "
                        "play openings 0..N-1, which self-play should not train on")
    p.add_argument("--book-share", type=float, default=0.75,
                   help="share of games started from the book; the rest start from the shallow "
                        "openings (the start position included), so the first moves keep data")
    p.add_argument("--dirichlet-alpha", type=float, default=0.0,
                   help="root noise on full searches (AlphaZero chess: 0.3); 0 = off")
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="stop after this running time, frozen time excluded (games in flight "
                        "are dropped); 0 = no limit")
    p.add_argument("--parallel-games", type=int, default=256,
                   help="games in flight, all workers: a forward carries about parallel_games / "
                        "workers / 2 * sim_batch positions -- size it to where the GPU's evals/s "
                        "peak (make bench-forward; 128 rows on the RTX 5070 Ti)")
    p.add_argument("--sim-batch", type=int, default=1,
                   help="leaves per search wave: 1 = sequential search (+49 Elo over 16 at 800 "
                        "sims); the bot sends 16; above 16 needs --wide-batch")
    p.add_argument("--wide-batch", action="store_true",
                   help="allow --sim-batch above 16, which weakens the search: at 64 leaves, 46%% "
                        "of the sims revisit a pending leaf and the best move matches sequential "
                        "search 83%% of the time (16: 79%% new evals, 85%%); feed the GPU with "
                        "more games instead")
    p.add_argument("--threads", type=int, default=6, help="CPU threads for the searches")
    p.add_argument("--legacy-engine", action="store_true",
                   help="the older self-play search (immediate TT backups, no tree reuse)")
    p.add_argument("--engines", type=int, default=2,
                   help="engine halves taking turns on the GPU (2 = the CPU searches one while "
                        "the GPU evaluates the other); 1 = no overlap")
    p.add_argument("--queued", action="store_true",
                   help="no lockstep halves: games queue their positions as their evaluations come "
                        "back, and batches of --batch-rows leave the queue with up to "
                        "--batches-in-flight on the GPU at once. Same games as lockstep; full "
                        "buckets once enough games play. Not --legacy-engine")
    p.add_argument("--batch-rows", type=int, default=0,
                   help="--queued: positions per batch (0 = the largest CUDA-graph bucket)")
    p.add_argument("--batches-in-flight", type=int, default=3,
                   help="--queued: batches queued on the GPU while the CPU searches")
    p.add_argument("--report-every", type=float, default=0.0,
                   help="every this many seconds, print the positions searched so far, games in "
                        "flight included: a throughput that needs no game to finish (benchmarks; "
                        "0 = off). Not --legacy-engine")
    p.add_argument("--verify-every", type=int, default=50,
                   help="recompute every N-th async batch synchronously and compare (0 = never)")
    p.add_argument("--leaves-per-wave", type=int, default=4, help="--legacy-engine only")
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--fpu", type=float, default=0.2)
    p.add_argument("--move-temp", type=float, default=1.0)
    p.add_argument("--temp-plies", type=int, default=30)
    p.add_argument("--temp-cutoff", type=float, default=0.0,
                   help="sample only the moves whose Q is within this of the most-visited move's "
                        "(Lc0's temp-value-cutoff; 0 = every move). In gen14, sampling over every "
                        "move gave away ~0.3 Q a game before ply 30; 0.05 removes it and keeps "
                        "2/3 of the moves that differ from the best")
    p.add_argument("--resign", type=float, default=0.93)
    p.add_argument("--resign-plies", type=int, default=6)
    p.add_argument("--no-resign-frac", type=float, default=0.1)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--tt-cap", type=int, default=2_000_000)
    p.add_argument("--flush-positions", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", choices=COMPILE_MODES, default=COMPILE_MODES[0],
                   help="torch.compile mode of the forward (CUDA)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=1,
                   help="processes on the same --out, each feeding the GPU from its own thread")
    p.add_argument("--gpus", default="",
                   help="comma-separated GPU ids the workers take round-robin (default: the visible one)")
    p.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)  # set by run_workers
    p.add_argument("--id-offset", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.sim_batch > 16 and not args.wide_batch:
        p.error("--sim-batch above 16 weakens the search (see --wide-batch); raise "
                "--parallel-games or --workers to give the GPU bigger batches")
    if not 0.0 <= args.book_share <= 1.0:
        p.error("--book-share must be in [0, 1]")
    if args.queued and args.legacy_engine:
        p.error("--queued needs the default engine")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    games_log = out_dir / "games.jsonl"
    if args.id_offset is None:
        # Ids of a run are interleaved across engines and workers, so resume after the largest.
        id_offset = (max((json.loads(line)["id"] for line in open(games_log)), default=-1) + 1
                     if games_log.exists() else 0)
        if id_offset:
            print(f"Resuming after {id_offset} existing games in {out_dir}")
        (out_dir / "generation_config.json").write_text(json.dumps(vars(args), indent=2))
    else:
        id_offset = args.id_offset
    if args.workers > 1 and args.worker is None:
        run_workers(args, id_offset)
        return
    # This process's share: worker w of n_w plays its own seed and the ids = w (mod n_w).
    n_w, w = (args.workers, args.worker) if args.worker is not None else (1, 0)
    seed = args.seed + 104_729 * w
    n_games = args.games // n_w + (w < args.games % n_w)
    parallel = max(1, args.parallel_games // n_w)

    cfg = {
        "num_sims": args.sims,
        "num_parallel_games": parallel,
        "leaves_per_wave": args.leaves_per_wave,
        "sim_batch": args.sim_batch,
        "threads": args.threads,
        "c_puct": args.c_puct,
        "fpu": args.fpu,
        "move_temp": args.move_temp,
        "temp_plies": args.temp_plies,
        "temp_cutoff": args.temp_cutoff,
        "resign_threshold": args.resign,
        "resign_plies": args.resign_plies,
        "no_resign_frac": args.no_resign_frac,
        "max_plies": args.max_plies,
        "tt_cap": args.tt_cap,
        "fast_sims": args.fast_sims,
        "full_frac": args.full_frac,
        "cap_total_full": args.cap_total == "all",
        "cap_total_fast": args.cap_total in ("fast", "all"),
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_eps": args.dirichlet_eps,
        "forced_playouts": args.forced_playouts,
        "fresh_full": args.fresh_full,
        "twofold_history": args.twofold_history,
        "draw_contempt": args.draw_contempt,
        "contempt_ramp": args.contempt_ramp,
        "depth_stats": args.depth_stats,
    }
    openings = selfplay_openings(args)
    forward = Forward(args.model, args.device, args.ema, not args.no_compile,
                      compile_mode=args.compile_mode)
    forward.verify_every = args.verify_every
    writer = ShardWriter(out_dir, flush_positions=args.flush_positions,
                         prefix="positions_" if n_w == 1 else f"positions_w{w}_")

    counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    done_games, n_positions = 0, 0
    log = os.open(games_log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    if args.legacy_engine:
        if args.forced_playouts > 0:
            print("WARNING: --legacy-engine has no forced playouts; searching without them", flush=True)
        if args.fresh_full:
            print("WARNING: --legacy-engine has no tree reuse; --fresh-full changes nothing", flush=True)
        if args.twofold_history:
            print("WARNING: --legacy-engine has no two-fold rule; --twofold-history changes nothing",
                  flush=True)
        if args.draw_contempt:
            print("WARNING: --legacy-engine has no draw contempt; --draw-contempt changes nothing",
                  flush=True)
        if args.temp_cutoff:
            print("WARNING: --legacy-engine samples every move; --temp-cutoff changes nothing",
                  flush=True)
        engines = [selfplay_core.Engine(json.dumps(cfg), openings, seed + id_offset, n_games)]
    else:
        cfg["tt_cap"] = min(args.tt_cap, 200_000)  # per game here, one shared TT in Engine
        del cfg["leaves_per_wave"]  # the legacy engine's batching; this one rejects unknown keys
        k_eng = 1 if args.queued else max(1, args.engines)
        cfg["num_parallel_games"] = max(1, parallel // k_eng)
        engines = [selfplay_core.SelfPlayEngine(
            json.dumps(cfg), openings, seed + id_offset + 1_000_003 * k,
            n_games // k_eng + (k < n_games % k_eng)) for k in range(k_eng)]

    def record(games: list, k: int):
        nonlocal done_games, n_positions
        for g in games:
            gid = id_offset + (g["id"] * len(engines) + k) * n_w + w
            z = g["z_white"]
            result = "1-0" if z > 0 else "0-1" if z < 0 else "1/2-1/2"
            counts[result] += 1
            n_positions += len(g["z"])
            writer.add_game(g, gid)
            append_line(log, json.dumps({
                "id": gid, "fen": g["fen"], "moves": g["moves"], "result": result,
                "end": g["end"], "plies": len(g["moves"]),
                "resign_allowed": g["resign_allowed"],
            }) + "\n")
            done_games += 1
            if done_games % 20 == 0:
                el = max(active, 1e-9)
                tt = sum(e.stats()[3] for e in engines)
                print(f"{done_games}/{args.games} games | {n_positions} pos | "
                      f"{done_games / el * 3600:.0f} games/h | tt={tt}", flush=True)

    # The budget counts running time only. A wave never takes 10 minutes, so a longer gap
    # between two waves is the process being frozen (the RL loop's night pause).
    budget = args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    active, last = 0.0, time.monotonic()
    inflight = [None] * len(engines)
    out = collections.deque()  # --queued: (batch id, forward handle), oldest first
    slot = 0
    batch_rows = args.batch_rows or forward.buckets[-1]
    next_report = args.report_every
    launches, rows = 0, 0
    while not (all(e.done() for e in engines) and all(h is None for h in inflight) and not out):
        now = time.monotonic()
        if now - last < 600:
            active += now - last
        last = now
        if active >= budget:
            print(f"time budget of {args.max_hours} h reached", flush=True)
            break
        if 0 < next_report <= active and not args.legacy_engine:
            searched = sum(e.stats()[4] for e in engines)
            print(f"searched {searched} positions in {active:.0f} s | "
                  f"{rows / max(launches, 1):.0f} positions per forward | "
                  f"{search_report(engines, args.depth_stats)}", flush=True)
            next_report += args.report_every
        if args.queued:
            # Collect and launch until --batches-in-flight batches are out, then take back the
            # oldest: its games queue again at the next collect, behind those still waiting.
            eng = engines[0]
            if not eng.done():
                bid, boards, player, castling, ep, legal = eng.collect_queued(batch_rows)
                record(eng.drain_finished(), 0)
                if len(player):
                    out.append((bid, forward.launch(slot, boards, player, castling, ep, legal)))
                    slot = (slot + 1) % args.batches_in_flight
                    launches, rows = launches + 1, rows + len(player)
                    if len(out) < args.batches_in_flight:
                        continue
                elif not out and not eng.done():
                    raise RuntimeError("engine returned an empty batch with games left")
            if out:
                bid, handle = out.popleft()
                eng.apply_queued(bid, *forward.wait(handle))
            continue
        for k, eng in enumerate(engines):
            if args.legacy_engine:  # synchronous, full logits
                boards, player, castling, ep = eng.collect()
                if len(player) > 0:
                    logits, values = forward(boards, player, castling, ep)
                else:
                    logits = np.zeros((0, 64, 73), dtype=np.float32)
                    values = np.zeros(0, dtype=np.float32)
                eng.apply(logits, values)
                record(eng.drain_finished(), k)
                continue
            if inflight[k] is not None:
                eng.apply_legal(*forward.wait(inflight[k]))
                inflight[k] = None
            if eng.done():
                continue
            boards, player, castling, ep, legal = eng.collect()
            record(eng.drain_finished(), k)
            if len(player):
                inflight[k] = forward.launch(k, boards, player, castling, ep, legal)
                launches, rows = launches + 1, rows + len(player)
            elif not eng.done():
                raise RuntimeError("engine returned an empty batch with games left")

    writer.flush()
    os.close(log)
    el = max(active, 1e-9)
    print(f"\nDone: {done_games} games, {n_positions} positions | "
          f"W {counts['1-0']} / D {counts['1/2-1/2']} / B {counts['0-1']} | "
          f"{el / 60:.1f} min ({done_games / el * 3600:.0f} games/h)")
    if not args.legacy_engine:
        print(f"search per move: {search_report(engines, args.depth_stats)}")


def search_report(engines: list, depth: bool) -> str:
    """Per move, by search kind: sims, GPU evals, TT hits and (with --depth-stats) the
    tree's mean / max simulation depth and main-line length, all engines summed."""
    parts = []
    for kind, i in (("full", 1), ("fast", 0)):
        t = {key: sum(e.search_stats()[i][key] for e in engines)
             for key in ("moves", "sims", "nn_evals", "tt_hits", "depth_mean", "depth_max", "main_line")}
        m = t["moves"]
        if not m:
            continue
        s = (f"{kind} {m} moves: {t['sims'] / m:.0f} sims, {t['nn_evals'] / m:.0f} evals, "
             f"{t['tt_hits'] / m:.0f} tt")
        if depth:
            s += (f", depth {t['depth_mean'] / m:.2f} mean / {t['depth_max'] / m:.1f} max / "
                  f"{t['main_line'] / m:.1f} line")
        parts.append(s)
    return " | ".join(parts)


if __name__ == "__main__":
    main()
