"""Forward throughput of the self-play network on this GPU: how many evals/s the search can get.

Two measurements per batch size:
  model   the compiled bf16 forward alone (CUDA-graph bucket = the batch), timed with events;
  launch  self-play's real path, selfplay_rust.Forward.launch/wait: pinned copies, H2D, the
          legal-move gather, D2H -- with the buckets capped at --top (larger batches split).
Run from the repo root:

    uv run python scripts/bench_forward.py --model data/rl/champion
    uv run python scripts/bench_forward.py --batches 1024,4096,8192 --top 2048   # bigger buckets

Self-play's batch per engine half is about parallel_games / workers / 2 * sim_batch leaves, e.g.
1024 games over 2 workers at sim_batch 16 -> ~4000; a batch above a bucket runs in the next one
up, padded. Pick the batch where evals/s peaks (128 rows on the RTX 5070 Ti) and size
--parallel-games to it. Compare evals/s with what `make bench` reports
(games/h * ~106 moves/game * ~110 evals/move / 3600) to see whether the GPU or the CPU is the limit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from selfplay_rust import COMPILE_MODES, Forward, load_model  # noqa: E402


def inputs(n: int, width: int, rng: np.random.Generator):
    boards = rng.integers(0, 13, (n, 64), dtype=np.uint8)
    player = rng.integers(0, 2, n, dtype=np.uint8)
    castling = rng.integers(0, 16, n, dtype=np.uint8)
    ep = np.full(n, 8, dtype=np.uint8)
    legal = rng.integers(0, 64 * 73, (n, width)).astype(np.int16)
    legal[:, width * 3 // 4:] = 64 * 73  # padding, as real rows have
    return boards, player, castling, ep, legal


def bench_model(model_dir: str, batches: list[int], iters: int, compile_model: bool, mode: str) -> None:
    model = load_model(Path(model_dir), "cuda").bfloat16().eval()
    params = sum(p.numel() for p in model.parameters())
    gflop = 2 * params * 64 / 1e9  # ~2 FLOPs per weight per token, 64 tokens; attention extra
    if compile_model:
        model = torch.compile(model, mode=mode)
    print(f"model: {params / 1e6:.1f}M params, ~{gflop:.1f} GFLOP/eval (weights only)")
    print(f"{'batch':>6} {'ms/batch':>9} {'evals/s':>10} {'~TFLOPS':>8} {'peak GB':>8}")
    for b in batches:
        x = [torch.randint(0, 13, (b, 64), device="cuda"), torch.randint(0, 2, (b,), device="cuda"),
             torch.randint(0, 16, (b,), device="cuda"), torch.full((b,), 8, device="cuda")]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(5):
                model(*x)
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters):
                model(*x)
            t1.record()
            torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / iters
        print(f"{b:>6} {ms:>9.2f} {b / ms * 1e3:>10,.0f} {b * gflop / ms:>8.0f} "
              f"{torch.cuda.max_memory_allocated() / 1e9:>8.1f}", flush=True)


def bench_launch(model_dir: str, batches: list[int], iters: int, top: int, width: int,
                 compile_model: bool, mode: str) -> None:
    buckets = tuple(b for b in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192) if b <= top)
    fwd = Forward(model_dir, "cuda", False, compile_model, buckets=buckets, compile_mode=mode)
    fwd.verify_every = 0
    rng = np.random.default_rng(0)
    print(f"\nlaunch/wait path, buckets {buckets}, legal width {width}, two slots alternating")
    print(f"{'batch':>6} {'ms/batch':>9} {'evals/s':>10}")
    for b in batches:
        data = [inputs(b, width, rng) for _ in range(2)]
        for k in range(4):  # warm-up, both slots
            fwd.wait(fwd.launch(k % 2, *data[k % 2]))
        torch.cuda.synchronize()
        t = time.perf_counter()
        pending = None
        for i in range(iters):  # keep one batch queued behind the running one, as self-play does
            h = fwd.launch(i % 2, *data[i % 2])
            if pending is not None:
                fwd.wait(pending)
            pending = h
        fwd.wait(pending)
        ms = (time.perf_counter() - t) / iters * 1e3
        print(f"{b:>6} {ms:>9.2f} {b / ms * 1e3:>10,.0f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="data/rl/champion")
    # Small sizes too: on the RTX 5070 Ti evals/s peaks at 128 rows and falls slightly above.
    p.add_argument("--batches", default="64,128,256,512,1024,2048,4096,8192")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--top", type=int, default=1024, help="largest CUDA-graph bucket of the launch path")
    p.add_argument("--width", type=int, default=64, help="legal-move columns per position")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", choices=COMPILE_MODES, default=COMPILE_MODES[0])
    p.add_argument("--only", choices=("model", "launch"))
    args = p.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    batches = [int(b) for b in args.batches.split(",")]
    if args.only != "launch":
        bench_model(args.model, batches, args.iters, not args.no_compile, args.compile_mode)
    if args.only != "model":
        bench_launch(args.model, batches, args.iters, args.top, args.width, not args.no_compile,
                     args.compile_mode)


if __name__ == "__main__":
    main()
