"""DistributedMuon must be numerically indistinguishable from torch.optim.Muon.

The whole risk of this optimizer is silent wrongness: sharding a matrix the wrong way still
produces a plausible-looking update (measured 52-58% off, no exception raised). So the
load-bearing test compares the *update* against single-rank torch.optim.Muon over several
steps, not just a smoke run.

    uv run pytest tests/test_distributed_muon.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chesstransformer.distributed_muon import (  # noqa: E402
    DistributedMuon,
    muon_momentum,
    muon_weight_decay,
)

# Deliberately mixed: two shapes big enough to be distributed, one below the threshold so
# the replicated path is exercised too, and counts that do not divide evenly by 4 so the
# zero-padding branch runs.
SHAPES = [(256, 512)] * 5 + [(512, 256)] * 3 + [(8, 16)] * 2
LR, MOM, WD = 0.02, 0.95, 0.1


def _make(seed=0):
    g = torch.Generator().manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(*s, generator=g)) for s in SHAPES]
    grads = [[torch.randn(*s, generator=g) for s in SHAPES] for _ in range(4)]
    return ps, grads


def _reference(steps=4):
    """Single-process torch.optim.Muon, the ground truth."""
    ps, grads = _make()
    opt = torch.optim.Muon(ps, lr=LR, momentum=MOM, weight_decay=WD)
    for st in range(steps):
        for p, gr in zip(ps, grads[st]):
            p.grad = gr.clone()
        opt.step()
    return [p.detach().clone() for p in ps]


def _run(fn, world_size, *args):
    ctx = mp.get_context("spawn")
    out = ctx.Manager().dict()
    mp.start_processes(_entry, args=(world_size, fn, out, args), nprocs=world_size,
                       join=True, start_method="spawn")
    return dict(out)


def _entry(rank, world_size, fn, out, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29699",
                      RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, out, *args)
    finally:
        dist.destroy_process_group()


def _distributed_run(rank, world_size, out, steps):
    ps, grads = _make()
    opt = DistributedMuon(ps, lr=LR, momentum=MOM, weight_decay=WD)
    for st in range(steps):
        for p, gr in zip(ps, grads[st]):
            p.grad = gr.clone()          # DDP would have all-reduced these already
        opt.step()
    out[rank] = [p.detach().clone() for p in ps]
    out[f"state_{rank}"] = sum(
        v["momentum_buffer"].numel() for v in opt.state.values() if "momentum_buffer" in v)
    out[f"owned_{rank}"] = (opt.owned_param_count, opt.total_param_count)


@pytest.mark.parametrize("world_size", [2, 4])
def test_matches_single_rank_muon(world_size):
    """N ranks must produce exactly what one rank running torch.optim.Muon produces.

    This is the test that catches sharding a matrix instead of a parameter list: that
    mistake is worth ~50% on the update and raises nothing.
    """
    ref = _reference()
    res = _run(_distributed_run, world_size, 4)
    for r in range(world_size):
        for i, (got, want) in enumerate(zip(res[r], ref)):
            scale = want.abs().max().item()
            err = (got - want).abs().max().item() / max(scale, 1e-12)
            assert err < 1e-5, (
                f"rank {r} param {i} {tuple(want.shape)} deviates {err:.2e} from "
                f"single-rank torch.optim.Muon")


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_ranks_agree(world_size):
    """Every rank must end the step holding identical parameters."""
    res = _run(_distributed_run, world_size, 3)
    for r in range(1, world_size):
        for i, (a, b) in enumerate(zip(res[0], res[r])):
            assert torch.equal(a, b), f"rank 0 and rank {r} disagree on param {i}"


@pytest.mark.parametrize("world_size", [2, 4])
def test_momentum_state_is_sharded(world_size):
    """Momentum must exist only for owned params, or there is no memory win.

    Small params stay replicated by design, so the total across ranks exceeds the
    single-rank total by exactly those; the large buckets must not be replicated.
    """
    res = _run(_distributed_run, world_size, 2)
    single = sum(int(torch.tensor(s).prod()) for s in SHAPES)
    total = sum(res[f"state_{r}"] for r in range(world_size))
    small = sum(int(torch.tensor(s).prod()) for s in SHAPES if int(torch.tensor(s).prod()) < 65_536)
    expected = single - small + small * world_size          # big sharded, small replicated
    assert total == expected, f"state total {total} != expected {expected}"
    assert res["state_0"] < single, "rank 0 stores as much state as a single-rank run"

    owned, tot = res["owned_0"]
    assert owned < tot, f"rank 0 owns all {tot} params; nothing was distributed"


def test_degenerates_without_process_group():
    """With no process group this must be plain Muon — same result, no collectives."""
    ref = _reference(steps=2)
    ps, grads = _make()
    opt = DistributedMuon(ps, lr=LR, momentum=MOM, weight_decay=WD)
    assert opt.world_size == 1
    for st in range(2):
        for p, gr in zip(ps, grads[st]):
            p.grad = gr.clone()
        opt.step()
    for got, want in zip(ps, ref):
        assert torch.allclose(got.detach(), want, atol=0, rtol=0)


def test_rejects_non_2d_params():
    """NS is only defined on matrices; a 1D param must fail loudly, not silently."""
    with pytest.raises(ValueError, match="2D"):
        DistributedMuon([torch.nn.Parameter(torch.randn(16))], lr=LR)


def test_momentum_schedule_shape():
    """Warm up 0.85 -> 0.97, hold, then warm down to 0.90."""
    total = 10_000
    assert muon_momentum(0, total) == pytest.approx(0.85, abs=1e-6)
    assert muon_momentum(400, total) == pytest.approx(0.97, abs=1e-6)
    assert muon_momentum(total // 2, total) == pytest.approx(0.97, abs=1e-6)
    assert muon_momentum(total - 1, total) == pytest.approx(0.90, abs=1e-2)
    # monotone up through warmup, monotone down through warmdown
    up = [muon_momentum(s, total) for s in range(0, 400, 40)]
    assert up == sorted(up)
    down = [muon_momentum(s, total) for s in range(total - 2000, total, 200)]
    assert down == sorted(down, reverse=True)


def test_weight_decay_schedule_decays_to_zero():
    total = 1_000
    assert muon_weight_decay(0, total, 0.1) == pytest.approx(0.1)
    assert muon_weight_decay(total, total, 0.1) == pytest.approx(0.0, abs=1e-9)
    mid = muon_weight_decay(total // 2, total, 0.1)
    assert 0.04 < mid < 0.06, f"cosine midpoint should be ~half, got {mid}"


# ── LR schedule, and its coupling to the momentum schedule ───────────────────

def _lr_curve(schedule, total=10_000, warmup=500, warmdown_ratio=0.2):
    from chesstransformer.trainers.pos2move_v2_trainer import create_lr_scheduler
    p = [torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.SGD(p, lr=1.0)
    sched = create_lr_scheduler(opt, warmup, total, 0.05,
                                schedule=schedule, warmdown_ratio=warmdown_ratio)
    out = []
    for _ in range(total):
        out.append(opt.param_groups[0]["lr"])
        sched.step()
    return out


def test_wsd_holds_peak_then_decays():
    """Warmup, a genuinely flat stable phase, then linear warmdown."""
    total, warmup = 10_000, 500
    lr = _lr_curve("wsd", total, warmup)
    assert lr[0] == pytest.approx(1 / warmup, abs=1e-6)
    assert lr[warmup - 1] == pytest.approx(1.0)
    stable = lr[warmup:total - 2000 + 1]
    assert all(x == pytest.approx(1.0) for x in stable), "stable phase is not flat"
    assert lr[-1] == pytest.approx(0.05, abs=1e-3)
    warmdown = lr[total - 2000:]
    assert warmdown == sorted(warmdown, reverse=True), "warmdown must be monotone"


def test_linear_schedule_still_available():
    """The previous shape stays reachable: decay begins right after warmup."""
    lr = _lr_curve("linear")
    assert lr[500] < 1.0 and lr[5000] < lr[2000] < lr[500]
    assert lr[-1] == pytest.approx(0.05, abs=1e-3)


def test_lr_and_momentum_warmdowns_start_together():
    """The momentum schedule warms down *during LR warmdown* -- so they must align.

    If the LR schedule has no warmdown phase (as with 'linear'), the momentum warmdown is
    keyed to a window that does not exist, which is the incoherence this coupling fixes.
    """
    total, ratio = 10_000, 0.2
    lr = _lr_curve("wsd", total, 500, ratio)
    start = total - round(ratio * total)
    assert lr[start] == pytest.approx(1.0), "LR should still be at peak at warmdown start"
    assert lr[start + 1] < 1.0, "LR must begin decaying right after"
    assert muon_momentum(start, total, warmdown_ratio=ratio) == pytest.approx(0.97, abs=1e-6)
    assert muon_momentum(total - 1, total, warmdown_ratio=ratio) == pytest.approx(0.90, abs=1e-2)


def test_old_checkpoint_config_keeps_linear_shape():
    """A scheduler_config from before this change has no 'schedule' key.

    It is replayed verbatim on resume, so the function default decides what those runs get;
    it must stay 'linear' or an in-flight run silently changes LR shape when it resumes.
    """
    from chesstransformer.trainers.pos2move_v2_trainer import create_lr_scheduler
    legacy = {"warmup_steps": 100, "total_steps": 1000, "final_lr_ratio": 0.05}
    p = [torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.SGD(p, lr=1.0)
    sched = create_lr_scheduler(opt, **legacy)
    lrs = []
    for _ in range(1000):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[500] < 0.99, "legacy config must not get a flat stable phase (that would be wsd)"
