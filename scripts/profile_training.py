
#!/usr/bin/env python3
"""Phase 0 baseline: is the pos2move_v2 trainer CPU-bound or GPU-bound?

Measures four things independently:
  A. dataloader-only throughput  (no model at all)
  B. compute-only throughput     (synthetic batch, already resident on GPU)
  C. combined real training step throughput
  D. device peak bf16 matmul TFLOPS  (for an honest MFU denominator)
"""
import argparse
import time
import statistics

import torch
from torch.utils.data import DataLoader, Subset

from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2, NUM_ACTION_PLANES
from chesstransformer.trainers.pos2move_v2_trainer import compute_loss

DEV = "cuda"
# Must mirror data/models/pos2move_v2.1/model_config.json, or the harness measures a
# model that never trains. (layer_drop turns out to cost nothing, but check, don't assume.)
MODEL_CFG = dict(embed_dim=256, nb_transformer_layers=16, num_heads=8, dropout=0.05,
                 kvq_bias=False, layer_drop=0.1)


def build_model(vocab_size):
    m = Pos2MoveV2(vocab_size=vocab_size, **MODEL_CFG).to(DEV)
    return m


def make_optims(model):
    muon, emb, head, other = [], [], [], []
    for name, p in model.named_parameters():
        if "embedding" in name:
            emb.append(p)
        elif "move_head" in name or "value_head" in name:
            head.append(p)
        elif p.ndim == 2:
            muon.append(p)
        else:
            other.append(p)
    o1 = torch.optim.Muon(muon, lr=1e-3, momentum=0.95, weight_decay=0.1)
    o2 = torch.optim.AdamW([
        {"params": emb, "lr": 2e-4, "weight_decay": 0.1},
        {"params": other, "lr": 5e-4, "weight_decay": 0.0},
        {"params": head, "lr": 5e-4, "weight_decay": 0.0},
    ])
    return o1, o2


def synthetic_batch(B, vocab_size, seq_len):
    """Same shapes/dtypes the real collate produces, but resident on GPU."""
    g = torch.Generator(device=DEV).manual_seed(0)
    legal = torch.zeros(B, 64, NUM_ACTION_PLANES, dtype=torch.bool, device=DEV)
    # ~30 legal moves per position, spread over from-squares
    idx_f = torch.randint(0, 64, (B, 30), device=DEV, generator=g)
    idx_p = torch.randint(0, NUM_ACTION_PLANES, (B, 30), device=DEV, generator=g)
    legal[torch.arange(B, device=DEV).unsqueeze(1), idx_f, idx_p] = True
    from_sq = idx_f[:, 0]
    action_plane = idx_p[:, 0]
    legal[torch.arange(B, device=DEV), from_sq, action_plane] = True
    return {
        "position": torch.randint(0, vocab_size, (B, seq_len), device=DEV, generator=g),
        "is_white": torch.randint(0, 2, (B,), device=DEV, generator=g).bool(),
        "castling_rights": torch.randint(0, 16, (B,), device=DEV, generator=g),
        "en_passant_file": torch.randint(0, 9, (B,), device=DEV, generator=g),
        "from_square": from_sq,
        "action_plane": action_plane,
        "legal_moves_planes": legal,
        "result": torch.randint(0, 3, (B,), device=DEV, generator=g),
        "move_number": torch.randint(0, 80, (B,), device=DEV, generator=g),
    }


def train_step(model, batch, o1, o2, precision):
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    with torch.autocast("cuda", dtype=dt, enabled=precision != "fp32"):
        logits, value = model(batch["position"], batch["is_white"].long(),
                              batch["castling_rights"], batch["en_passant_file"])
    loss, _ = compute_loss(logits, value, batch["from_square"], batch["action_plane"],
                           batch["legal_moves_planes"], batch["result"], batch["is_white"],
                           batch["move_number"], 5.0, label_smoothing=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    o1.step(); o2.step()
    o1.zero_grad(set_to_none=True); o2.zero_grad(set_to_none=True)
    return loss


def bench_dataloader(args, dataset):
    """A. Pure input pipeline SUSTAINED throughput.

    Must run long enough to drain the prefetch buffer (workers x prefetch_factor
    batches), otherwise you are timing a queue pop, not the producers.
    """
    n_batches = max(args.dl_batches, args.workers * 4 * 3)
    n = min(len(dataset), args.batch_size * (n_batches + 8))
    sub = Subset(dataset, list(range(n)))
    dl = DataLoader(sub, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    persistent_workers=True, prefetch_factor=4, pin_memory=True)
    it = iter(dl)
    for _ in range(4):                      # let workers spin up
        next(it)
    nbytes, seen, per_batch = 0, 0, []
    t_start = time.perf_counter()
    for i in range(n_batches):
        t0 = time.perf_counter()
        b = next(it)
        per_batch.append(time.perf_counter() - t0)
        seen += args.batch_size
        if i == 0:
            nbytes = sum(v.element_size() * v.nelement()
                         for v in b.values() if torch.is_tensor(v))
    wall = time.perf_counter() - t_start
    del it, dl
    return wall, seen, nbytes, per_batch


def bench_step_breakdown(args, model, o1, o2, batch):
    """Where does the step time actually go? fwd / bwd / clip / Muon / AdamW."""
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    acc = {"fwd": [], "bwd": [], "clip": [], "muon": [], "adamw": []}

    def tick():
        torch.cuda.synchronize()
        return time.perf_counter()

    for _ in range(args.warmup + args.iters):
        t0 = tick()
        with torch.autocast("cuda", dtype=dt, enabled=args.precision != "fp32"):
            logits, value = model(batch["position"], batch["is_white"].long(),
                                  batch["castling_rights"], batch["en_passant_file"])
        loss, _ = compute_loss(logits, value, batch["from_square"], batch["action_plane"],
                               batch["legal_moves_planes"], batch["result"], batch["is_white"],
                               batch["move_number"], 5.0, label_smoothing=0.1)
        t1 = tick()
        loss.backward()
        t2 = tick()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        t3 = tick()
        o1.step()
        t4 = tick()
        o2.step()
        t5 = tick()
        o1.zero_grad(set_to_none=True); o2.zero_grad(set_to_none=True)
        acc["fwd"].append(t1 - t0); acc["bwd"].append(t2 - t1)
        acc["clip"].append(t3 - t2); acc["muon"].append(t4 - t3); acc["adamw"].append(t5 - t4)
    return {k: statistics.median(v[args.warmup:]) for k, v in acc.items()}


def bench_compute(args, model, o1, o2, vocab_size, seq_len):
    """B. Pure compute: synthetic batch already on GPU, nothing to wait for."""
    batch = synthetic_batch(args.batch_size, vocab_size, seq_len)
    for _ in range(args.warmup):
        train_step(model, batch, o1, o2, args.precision)
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_step(model, batch, o1, o2, args.precision)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times, batch


def bench_combined(args, dataset, model, o1, o2):
    """C. The real thing: real dataloader + real step."""
    n = min(len(dataset), args.batch_size * (args.iters + args.warmup) * 2)
    sub = Subset(dataset, list(range(n)))
    dl = DataLoader(sub, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    persistent_workers=True, prefetch_factor=4, pin_memory=True)
    it = iter(dl)
    for _ in range(args.warmup):
        b = next(it)
        b = {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        train_step(model, b, o1, o2, args.precision)
    torch.cuda.synchronize()
    times, utils = [], []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        b = next(it)
        b = {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        train_step(model, b, o1, o2, args.precision)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        try:
            utils.append(torch.cuda.utilization())
        except Exception:
            pass
    del it, dl
    return times, utils


def bench_peak_matmul():
    """D. Achievable bf16 dense TFLOPS on this card -> honest MFU denominator."""
    n = 8192
    a = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = (2 * n ** 3 * reps) / dt / 1e12
    del a, b
    torch.cuda.empty_cache()
    return tflops


def count_flops(model, batch, precision):
    from torch.utils.flop_counter import FlopCounterMode
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.autocast("cuda", dtype=dt, enabled=precision != "fp32"):
            logits, value = model(batch["position"], batch["is_white"].long(),
                                  batch["castling_rights"], batch["en_passant_file"])
        loss, _ = compute_loss(logits, value, batch["from_square"], batch["action_plane"],
                               batch["legal_moves_planes"], batch["result"], batch["is_white"],
                               batch["move_number"], 5.0, label_smoothing=0.1)
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def report_flops(model, sbatch, args, peak, c_med):
    """Print GFLOP/sample and MFU. Destroys the compiled model -- call last."""
    flops = count_flops(model, sbatch, args.precision)
    print(f"\n  {flops/args.batch_size/1e9:.2f} GFLOP/sample (fwd+bwd, measured)")
    achieved = flops / c_med / 1e12
    print(f"  {achieved:,.1f} TFLOPS achieved  ->  MFU = {100*achieved/peak:.1f}%")


def p(label, times, bs):
    med = statistics.median(times)
    print(f"  {label:<34} {med*1000:8.1f} ms/batch   {bs/med:10,.0f} samples/s")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/elite_db.h5")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--sample-weighting", default="uniform")
    ap.add_argument("--dl-batches", type=int, default=150)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--skip-data", action="store_true",
                    help="compute-side only; skips the slow dataloader benches")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"\ndevice: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}")
    print(f"batch={args.batch_size} workers={args.workers} precision={args.precision} "
          f"iters={args.iters}\n")

    dataset = HDF5ChessDataset(args.data, sample_weighting=args.sample_weighting)
    vocab_size = dataset.position_tokenizer.vocab_size

    model = build_model(vocab_size)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"params: {n_params:,}")
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
        print(f"torch.compile enabled (mode={args.compile_mode})")

    o1, o2 = make_optims(model)

    print("\nD. device peak (bf16 dense matmul)")
    peak = bench_peak_matmul()
    print(f"  {peak:,.1f} TFLOPS achievable")

    print("\nB. compute only (synthetic batch, resident on GPU)")
    ct, sbatch = bench_compute(args, model, o1, o2, vocab_size, 64)
    c_med = p("fwd+bwd+opt", ct, args.batch_size)

    # NOTE: count_flops must run LAST, after every timing bench. FlopCounterMode is
    # a TorchDispatchMode; running it against a compiled model forces a fallback
    # that leaks into every later call on that model and silently reports eager
    # timings. Getting this wrong is invisible -- the numbers look plausible, they
    # are just the eager ones. (It cost us a bogus 2,272 samples/s end-to-end in the
    # Phase 0 baseline; see docs/distributed-log.md.)
    print("\nB2. step breakdown")
    bd = bench_step_breakdown(args, model, o1, o2, sbatch)
    tot = sum(bd.values())
    for k, v in bd.items():
        print(f"  {k:<34} {v*1000:8.1f} ms   {100*v/tot:5.1f}%")
    print(f"  {'(sum)':<34} {tot*1000:8.1f} ms")

    if args.skip_data:
        report_flops(model, sbatch, args, peak, c_med)
        print("\n(skipping dataloader benches)")
        return

    print("\nA. dataloader only, SUSTAINED (no model)")
    wall, seen, nbytes, per_batch = bench_dataloader(args, dataset)
    d_med = wall / (seen / args.batch_size)
    print(f"  {len(per_batch)} batches in {wall:.1f}s")
    print(f"  {'sustained':<34} {d_med*1000:8.1f} ms/batch   "
          f"{seen/wall:10,.0f} samples/s")
    print(f"  first-batch {per_batch[0]*1000:.1f} ms  |  last-batch "
          f"{per_batch[-1]*1000:.1f} ms   (rising = prefetch buffer draining)")
    print(f"  {nbytes/1e6:.1f} MB per collated batch")

    print("\nC. combined (real dataloader + real step)")
    tt, utils = bench_combined(args, dataset, model, o1, o2)
    t_med = p("end-to-end", tt, args.batch_size)
    if utils:
        print(f"  GPU util during loop: median {statistics.median(utils):.0f}%  "
              f"max {max(utils):.0f}%")

    # Safe here: nothing is timed after this point.
    report_flops(model, sbatch, args, peak, c_med)

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"  compute-only ceiling : {args.batch_size/c_med:10,.0f} samples/s")
    print(f"  dataloader ceiling   : {args.batch_size/d_med:10,.0f} samples/s "
          f"({args.workers} workers)")
    print(f"  actual end-to-end    : {args.batch_size/t_med:10,.0f} samples/s")
    bound = "DATA-BOUND" if d_med > c_med else "COMPUTE-BOUND"
    print(f"\n  -> {bound}  (dataloader time is {d_med/c_med:.2f}x the compute time)")
    if d_med > c_med:
        print(f"  -> GPU idle ~{100*(1-c_med/t_med):.0f}% of the step")
        print(f"  -> A perfect input pipeline would give {t_med/c_med:.1f}x on THIS GPU.")
    else:
        gpus = d_med / c_med
        print(f"  -> The loader can feed ~{1/gpus:.1f} GPUs of this type before "
              f"becoming the bottleneck.")
        print(f"  -> DDP scaling is safe up to that point; beyond it you need "
              f"Phase 1 shards.")


if __name__ == "__main__":
    main()
