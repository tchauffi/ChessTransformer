#!/usr/bin/env python3
"""How does HDF5ChessDataset throughput scale with num_workers?

Determines the per-worker sample rate, which is what sizes the CPU requirement
for a multi-GPU node.
"""
import time
import torch
from torch.utils.data import DataLoader, Subset
from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset

BS = 512
NB = 100

ds = HDF5ChessDataset("data/elite_db.h5", sample_weighting="uniform")
# Sample across the WHOLE dataset, not the first N games -- otherwise the OS
# page cache and the per-worker game_cache make it look faster than reality.
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[: BS * (NB + 8)].tolist()
sub = Subset(ds, idx)

print(f"{'workers':>8} {'samples/s':>12} {'ms/batch':>10} {'per-worker':>12}")
for w in [1, 2, 4, 8, 12, 16]:
    dl = DataLoader(sub, batch_size=BS, shuffle=True, num_workers=w,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    for _ in range(4):
        next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in range(NB):
        next(it)
        n += BS
    wall = time.perf_counter() - t0
    rate = n / wall
    print(f"{w:>8} {rate:>12,.0f} {1000*wall/NB:>10.1f} {rate/w:>12,.0f}")
    del it, dl
