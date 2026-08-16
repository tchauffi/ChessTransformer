"""DDP correctness tests, runnable on CPU/gloo with no GPU.

The point of running these on gloo is that they cost nothing: multi-rank bugs are found
locally and in CI, not at $12/hour on a rented 8-GPU box. Every test here spawns real
processes with ``torchrun``-equivalent env and a real process group.

The load-bearing test is ``test_grad_equivalence_vs_single_rank``. DDP's whole contract is
that N ranks at batch B produce the gradient one rank at batch N*B would: if that holds,
almost everything else is downstream of it, and if it silently does not, the loss curve
still looks fine while the run is quietly wrong.

    uv run pytest tests/test_distributed.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chesstransformer.datasets.flat_shard_dataset import (  # noqa: E402
    FlatShardDataset,
    make_dataloader,
    scatter_legal_planes,
)
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2  # noqa: E402

TINY = dict(embed_dim=32, nb_transformer_layers=2, num_heads=4, dropout=0.0, layer_drop=0.0)


def _run(fn, world_size, *args):
    """Spawn `world_size` gloo ranks running fn(rank, world_size, return_dict, *args)."""
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    out = manager.dict()
    mp.start_processes(
        _entry, args=(world_size, fn, out, args), nprocs=world_size, join=True, start_method="spawn"
    )
    return dict(out)


def _entry(rank, world_size, fn, out, args):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29677",
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, out, *args)
    finally:
        dist.destroy_process_group()


def _batch(ds, idx):
    """Fetch a batch and shape it the way the trainer's unpack_batch does."""
    b = ds[np.asarray(idx)]
    return dict(
        board=b["position"].long(),
        player=b["is_white"].long(),
        castling=b["castling_rights"].long(),
        en_passant=b["en_passant_file"].long(),
        from_sq=b["from_square"].long(),
        action_plane=b["action_plane"].long(),
        is_white=b["is_white"].long(),
        legal=scatter_legal_planes(b["legal_idx"].long()),
    )


def _loss(model, b):
    """Policy CE plus a value term.

    The value term is not decorative: without it `value_head` receives no gradient, its
    `p.grad` stays None, and the gradient comparisons below silently skip part of the
    model (or crash). It also mirrors the real compute_loss, which is two-headed.
    """
    logits, value = model(b["board"], b["player"], b["castling"], b["en_passant"])
    flat = logits.view(logits.size(0), -1).float()
    target = b["from_sq"] * 73 + b["action_plane"]
    policy = torch.nn.functional.cross_entropy(flat, target)
    # Deterministic pseudo-target from the batch itself; the value is what matters here,
    # not that it is the true game outcome.
    value_target = (b["is_white"].float() * 2 - 1).view_as(value) if "is_white" in b else torch.zeros_like(value)
    return policy + ((value.float() - value_target) ** 2).mean()


# ── sampler ──────────────────────────────────────────────────────────────────


def _sampler_coverage(rank, world_size, out, shards):
    ds = FlatShardDataset(shards, "val")
    dl = make_dataloader(ds, batch_size=256, shuffle=True, num_workers=0,
                         drop_last=False, pin_memory=False, seed=7, distributed=True)
    dl.sampler.sampler.set_epoch(0)
    out[rank] = sorted(i for b in dl.sampler for i in b)
    out["n"] = len(ds)


@pytest.mark.parametrize("world_size", [2, 4])
def test_sampler_partitions_dataset(world_size, shards):
    """Union of all ranks covers the dataset; ranks do not overlap.

    This is the test that would have caught the loaders never being told to shard --
    without `distributed=True` every rank returns the identical index list.
    """
    res = _run(_sampler_coverage, world_size, shards)
    n = res.pop("n")
    sets = [set(res[r]) for r in range(world_size)]
    union = set().union(*sets)

    assert union == set(range(n)), "ranks must jointly cover the whole dataset"
    overlap = sum(len(sets[i] & sets[j]) for i in range(world_size) for j in range(i + 1, world_size))
    # DistributedSampler pads to a multiple of world_size by repeating indices, so a few
    # duplicates are expected and documented -- but only a few.
    assert overlap < world_size, f"{overlap} overlapping indices; expected < {world_size} from padding"
    counts = {len(s) for s in sets}
    assert len(counts) == 1, f"ranks got unequal shares: {counts}"


def _set_epoch_order(rank, world_size, out, shards):
    ds = FlatShardDataset(shards, "val")
    dl = make_dataloader(ds, batch_size=256, shuffle=True, num_workers=0,
                         drop_last=False, pin_memory=False, seed=7, distributed=True)
    orders = []
    for ep in (0, 1, 0):
        dl.sampler.sampler.set_epoch(ep)
        orders.append([i for b in dl.sampler for i in b][:32])
    out[rank] = orders


def test_set_epoch_reshuffles(shards):
    """Without set_epoch every pass over the data replays the identical order."""
    res = _run(_set_epoch_order, 2, shards)
    for rank, (ep0, ep1, ep0_again) in res.items():
        assert ep0 != ep1, f"rank {rank}: set_epoch did not change the order"
        assert ep0 == ep0_again, f"rank {rank}: same epoch must be reproducible"


# ── gradients ────────────────────────────────────────────────────────────────


def _grad_sync(rank, world_size, out, shards):
    torch.manual_seed(0)
    model = Pos2MoveV2(**TINY)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    ds = FlatShardDataset(shards, "val")
    idx = list(range(rank * 16, rank * 16 + 16))          # each rank a different slice
    _loss(ddp, _batch(ds, idx)).backward()
    missing = [n for n, p in ddp.module.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    out[rank] = {n: p.grad.detach().clone() for n, p in ddp.module.named_parameters()}


def test_gradients_identical_across_ranks(shards):
    """After backward, DDP must leave every rank holding the same gradient."""
    res = _run(_grad_sync, 2, shards)
    for name, g0 in res[0].items():
        assert torch.allclose(g0, res[1][name], atol=0, rtol=0), f"{name} differs across ranks"


def _grad_vs_single(rank, world_size, out, shards):
    torch.manual_seed(0)
    model = Pos2MoveV2(**TINY)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    ds = FlatShardDataset(shards, "val")
    per_rank = 16
    idx = list(range(rank * per_rank, (rank + 1) * per_rank))
    _loss(ddp, _batch(ds, idx)).backward()
    if rank == 0:
        out["ddp"] = {n: p.grad.detach().clone() for n, p in ddp.module.named_parameters()}
        # Same total data, one rank, one batch.
        torch.manual_seed(0)
        solo = Pos2MoveV2(**TINY)
        solo.zero_grad()
        _loss(solo, _batch(ds, list(range(per_rank * world_size)))).backward()
        out["solo"] = {n: p.grad.detach().clone() for n, p in solo.named_parameters()}


@pytest.mark.parametrize("world_size", [2, 4])
def test_grad_equivalence_vs_single_rank(world_size, shards):
    """N ranks at batch B == 1 rank at batch N*B.

    DDP averages gradients across ranks and cross-entropy averages over the batch, so the
    two are the same computation grouped differently. This is the single test that catches
    most DDP mistakes -- wrong reduction, unsharded data, a rank silently skipping a step.
    """
    res = _run(_grad_vs_single, world_size, shards)
    ddp, solo = res["ddp"], res["solo"]
    worst = max(
        (ddp[n] - solo[n]).abs().max().item() / max(solo[n].abs().max().item(), 1e-12)
        for n in solo
    )
    assert worst < 1e-4, f"largest relative gradient deviation {worst:.2e} exceeds fp tolerance"


# ── EMA ──────────────────────────────────────────────────────────────────────


def _ema_consistency(rank, world_size, out, shards):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from chesstransformer.trainers.pos2move_v2_trainer import create_ema_state, update_ema

    torch.manual_seed(0)
    model = Pos2MoveV2(**TINY)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    ema = create_ema_state(ddp.module)
    ds = FlatShardDataset(shards, "val")
    opt = torch.optim.SGD(ddp.parameters(), lr=0.1)
    for step in range(3):
        opt.zero_grad()
        idx = [(rank * 8 + step * 32 + i) for i in range(8)]
        _loss(ddp, _batch(ds, idx)).backward()
        opt.step()
        update_ema(ddp.module, ema, 0.9)
    out[rank] = {k: v.clone() for k, v in ema.items()}


def test_ema_identical_across_ranks(shards):
    """EMA is computed redundantly on every rank; it must not drift between them.

    Gradients are all-reduced before the step, so params stay identical and so must the
    EMA. If this ever fails, `swap_ema_weights` before validation would have each rank
    evaluating different weights and disagreeing about the best checkpoint.
    """
    res = _run(_ema_consistency, 2, shards)
    for name, v0 in res[0].items():
        assert torch.allclose(v0, res[1][name], atol=0, rtol=0), f"EMA {name} drifted across ranks"


# ── determinism ──────────────────────────────────────────────────────────────


def _loss_trajectory(rank, world_size, out, shards):
    torch.manual_seed(0)
    model = Pos2MoveV2(**TINY)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    ds = FlatShardDataset(shards, "val")
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    losses = []
    for step in range(4):
        opt.zero_grad()
        idx = [(rank * 8 + step * 32 + i) for i in range(8)]
        loss = _loss(ddp, _batch(ds, idx))
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 6))
    out[rank] = losses


def test_same_seed_same_trajectory(shards):
    """Same seed and world size must reproduce the same loss trajectory."""
    a = _run(_loss_trajectory, 2, shards)
    b = _run(_loss_trajectory, 2, shards)
    assert a[0] == b[0], f"rank 0 trajectory not reproducible:\n{a[0]}\n{b[0]}"


# ── precision guard ──────────────────────────────────────────────────────────


def test_precision_falls_back_on_pre_ampere(monkeypatch):
    """bf16 needs compute capability 8.0; Kaggle's T4 is 7.5.

    Without the fallback the default --precision bf16 fails on the exact hardware the free
    testbed runs on.
    """
    from chesstransformer.trainers.pos2move_v2_trainer import resolve_precision

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("ACCELERATE_USE_CPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert resolve_precision("bf16") == "fp16"

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert resolve_precision("bf16") == "bf16"
    # Accelerate spells fp32 as "no" and raises on "fp32".
    assert resolve_precision("fp32") == "no"


def test_synthetic_shards_are_format_valid(synthetic_shards):
    """The CI fixture must produce data the real pipeline would accept.

    Otherwise CI passes against records the loader never sees in production -- in
    particular the scatter would not catch pad-as-zero, the bug the format is shaped to
    avoid.
    """
    ds = FlatShardDataset(synthetic_shards, "train")
    b = ds[np.arange(min(256, len(ds)))]
    planes = scatter_legal_planes(b["legal_idx"].long())
    assert torch.equal(planes.reshape(len(b["legal_cnt"]), -1).sum(1).int(),
                       b["legal_cnt"].int()), "set bits must equal legal_cnt (pad leaked in?)"
    assert b["position"].max() < 13 and b["position"].min() >= 0
    assert b["from_square"].max() < 64 and b["action_plane"].max() < 73
    assert b["en_passant_file"].max() <= 8 and b["castling_rights"].max() < 16
