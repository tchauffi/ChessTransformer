"""Where the self-play forward's GPU time goes, per batch size.

FLOPs per position (FlopCounterMode, eager), then for each batch: ms per forward of the
production compile (--compile-mode, CUDA graphs), achieved TFLOPS, kernel time by kind
(torch.profiler activity tracing: needs no GPU counter permission) and the board's duty cycles
while a 12 s loop runs (nvidia-smi: SM busy, memory controller busy, power, SM clock). The
ceilings it compares to are measured on the same card: an 8192^3 bf16 GEMM and a device copy.
Run from the repo root:

    uv run python scripts/profile_forward.py --model data/rl/champion

On the RTX 5070 Ti (2026-10-25): 2.04 GFLOP per position; ~15k positions/s from 64 to 1024
rows at ~31 TFLOPS of an 87 TFLOPS GEMM ceiling; at 1024 rows matmuls take ~47 % of the time
for 96 % of the FLOPs, attention with the smolgen bias ~17 % and fused norm/activation kernels
~36 %; memory busy 23 % at 64 rows, 80 % at 1024 (activations leave L2).
"""
from __future__ import annotations

import argparse
import collections
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from selfplay_rust import COMPILE_MODES, load_model  # noqa: E402


def inputs(b: int) -> list[torch.Tensor]:
    return [torch.randint(0, 13, (b, 64), device="cuda"), torch.randint(0, 2, (b,), device="cuda"),
            torch.randint(0, 16, (b,), device="cuda"), torch.full((b,), 8, device="cuda")]


def kind(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("fmha", "flash", "attentionkernel")):  # before "cutlass": fmha_cutlass*
        return "attention"
    if "triton" in n:  # inductor's fused kernels, whatever ops their names list
        return "fused elementwise/norm"
    if any(k in n for k in ("gemm", "cutlass", "nvjet", "xmma", "cublas", "matmul")):
        return "matmul"
    if "memcpy" in n or "memset" in n:
        return "copy/set"
    return "other"


def smi_sampler(stop: threading.Event, rows: list) -> None:
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,power.draw,"
                              "clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        try:
            rows.append([float(x) for x in out.strip().split(",")])
        except ValueError:
            pass
        time.sleep(0.5)


def timed(fn, iters: int) -> float:
    """ms per call."""
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def gemm_tflops() -> float:
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    return 2 * 8192**3 / (timed(lambda: a @ b, 20) / 1e3) / 1e12


def copy_gbs() -> float:
    x = torch.empty(512 * 2**20, device="cuda", dtype=torch.uint8)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    return 2 * x.numel() / (timed(lambda: y.copy_(x), 20) / 1e3) / 1e9  # read + write


def ceilings() -> tuple[float, float]:
    tflops, gbs = gemm_tflops(), copy_gbs()
    torch.cuda.empty_cache()
    return tflops, gbs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="data/rl/champion")
    p.add_argument("--batches", default="64,128,256,512,1024")
    p.add_argument("--top", type=int, default=12, help="kernels listed for the largest batch")
    p.add_argument("--compile-mode", choices=COMPILE_MODES, default=COMPILE_MODES[0])
    args = p.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    peak_tf, peak_bw = ceilings()
    print(f"ceilings: bf16 GEMM 8192^3 {peak_tf:.0f} TFLOPS | device copy {peak_bw:.0f} GB/s")

    model = load_model(Path(args.model), "cuda").bfloat16().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(*inputs(64))
    per_row = fc.get_total_flops() / 64
    ops = sorted(fc.get_flop_counts().get("Global", {}).items(), key=lambda kv: -kv[1])
    print(f"model: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M params | "
          f"{per_row / 1e9:.2f} GFLOP per position | " +
          ", ".join(f"{str(k).split('.')[-1]} {v / 64 / 1e9:.2f}" for k, v in ops))

    compiled = torch.compile(model, mode=args.compile_mode)
    batches = [int(b) for b in args.batches.split(",")]
    print(f"\n{'batch':>6} {'ms':>7} {'pos/s':>8} {'TFLOPS':>7} {'%ceil':>6} | kernel time | "
          f"nvidia-smi SM% mem% W MHz")
    for b in batches:
        x = inputs(b)
        with torch.no_grad():
            for _ in range(5):
                compiled(*x)
            torch.cuda.synchronize()
            ms = timed(lambda: compiled(*x), 30)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    compiled(*x)
                torch.cuda.synchronize()
            by, names = collections.Counter(), collections.Counter()
            for e in prof.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    by[kind(e.name)] += e.device_time
                    names[e.name[:90]] += e.device_time
            total = sum(by.values()) or 1
            stop, rows = threading.Event(), []
            th = threading.Thread(target=smi_sampler, args=(stop, rows))
            th.start()
            end = time.time() + 12
            while time.time() < end:
                for _ in range(20):
                    compiled(*x)
                torch.cuda.synchronize()
            stop.set()
            th.join()
        smi = [sum(c) / len(c) for c in zip(*rows[2:])] if len(rows) > 3 else [0.0] * 4
        tf = per_row * b / (ms / 1e3) / 1e12
        share = ", ".join(f"{k} {v / total:.0%}" for k, v in by.most_common())
        print(f"{b:>6} {ms:>7.2f} {b / ms * 1e3:>8,.0f} {tf:>7.1f} {tf / peak_tf:>6.0%} | {share} | "
              f"{smi[0]:.0f}% {smi[1]:.0f}% {smi[2]:.0f} W {smi[3]:.0f} MHz", flush=True)
        if b == batches[-1]:
            print(f"  top kernels at {b}:")
            for name, t in names.most_common(args.top):
                print(f"    {t / total:6.1%}  {kind(name):<22} {name}")


if __name__ == "__main__":
    main()
