"""Measure what DistributedMuon actually buys, and what it costs.

Two things are being traded and both need to be on the table:

  * saved   -- Newton-Schulz compute (divided by world size) and momentum state (likewise)
  * spent   -- one all_gather of the updated params per shape bucket per step, which is
               traffic DDP was not previously moving

Whether that is a win depends on the interconnect, so this measures rather than asserts.
Run it on the hardware you intend to train on; gloo/CPU numbers say nothing about NCCL.

    uv run scripts/bench_distributed_muon.py --preset large            # 1 rank baseline
    torchrun --nproc_per_node=2 scripts/bench_distributed_muon.py --preset large
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chesstransformer.distributed_muon import DistributedMuon  # noqa: E402
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2  # noqa: E402
from chesstransformer.trainers.pos2move_v2_trainer import PRESETS  # noqa: E402


def state_bytes(opt) -> int:
    return sum(v["momentum_buffer"].numel() * v["momentum_buffer"].element_size()
               for v in opt.state.values() if "momentum_buffer" in v)


def bench(opt, params, steps, warmup=3):
    for _ in range(warmup):
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cpu", action="store_true",
                    help="Force gloo/CPU. Needed to run >1 rank on a single-GPU box, since "
                         "NCCL requires one device per rank. Measures the compute and state "
                         "split; the communication cost is NOT representative of NCCL.")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    distributed = "RANK" in __import__("os").environ
    if distributed:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    rank = dist.get_rank() if distributed else 0
    ws = dist.get_world_size() if distributed else 1
    dev = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(dev)

    cfg = PRESETS[args.preset]
    model = Pos2MoveV2(embed_dim=cfg["embed_dim"], nb_transformer_layers=cfg["num_layers"],
                       num_heads=cfg["num_heads"]).to(dev)
    twod = [p for p in model.parameters() if p.ndim == 2]
    for p in twod:
        p.grad = torch.randn_like(p)          # DDP would have all-reduced these

    dist_opt = DistributedMuon(twod, lr=1e-3, momentum=0.95, weight_decay=0.1)
    d_ms = bench(dist_opt, twod, args.steps)
    d_state = state_bytes(dist_opt)

    ref_ms = ref_state = None
    if ws == 1:
        ref = torch.optim.Muon(twod, lr=1e-3, momentum=0.95, weight_decay=0.1)
        ref_ms = bench(ref, twod, args.steps)
        ref_state = state_bytes(ref)

    if rank == 0:
        n2d = sum(p.numel() for p in twod)
        print(f"  preset {args.preset}: {n2d/1e6:.1f}M params in {len(twod)} 2D tensors "
              f"| world_size={ws} | {dev}")
        print(f"  {dist_opt.sharding_summary().split(' | ')[0]}")
        print(f"  DistributedMuon step : {d_ms:8.2f} ms   momentum state {d_state/2**20:8.1f} MiB/rank")
        if ref_ms is not None:
            print(f"  torch.optim.Muon     : {ref_ms:8.2f} ms   momentum state {ref_state/2**20:8.1f} MiB/rank")
            print("  (world_size=1, so these should match; the win only appears at ws>1)")
        else:
            print("  Run with 1 rank for the torch.optim.Muon baseline to compare against.")
            print(f"  Expect step time ~/{ws} on NS compute, plus one all_gather per shape bucket.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
