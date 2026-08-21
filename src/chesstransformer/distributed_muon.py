"""Muon with Newton-Schulz distributed across ranks, plus its schedules.

Under plain DDP every rank all-reduces to the *same* gradient and then every rank runs the
*same* Newton-Schulz iteration on it. That is pure redundant compute: at world size N,
N-1 copies of the NS work are thrown away, and every rank stores a full momentum buffer.

The fix is to partition the work — but the partition axis is the whole point. NS is
``X @ X.T`` plus a global spectral-norm division, so it needs the *complete* matrix.
Measured on a 64x32 weight, orthogonalizing per-shard and concatenating gives a **52-58 %
wrong update** at world size 2-4, and nothing raises: a row-slice of a 2D matrix is still a
2D matrix. That is the ZeRO-2 trap, and it is why sharding *inside* a tensor is off limits
here even though it is perfectly safe for elementwise optimizers like AdamW.

So this partitions by **whole parameter**, following nanochat's ``MuonAdamW``
(https://github.com/karpathy/nanochat, ``nanochat/optim.py``): params of a given shape are
stacked, the K of them are divided into contiguous chunks across ranks, each rank runs NS
only on the complete matrices it owns, and one ``all_gather`` per shape bucket republishes
the results. Momentum state exists only for owned params, so it is sharded too.

**Communication note, stated honestly.** nanochat does not use DDP: it ``reduce_scatter``s
gradients inside the optimizer, so grad-sync and state-sharding share one collective and
the total wire cost equals what DDP's all-reduce would have been. This class runs *on top
of* DDP, which has already all-reduced the gradients, so the ``all_gather`` here is
**additional** traffic — bytes DDP did not previously move. It buys NS compute (÷N) and
momentum memory (÷N) at the cost of one param-sized gather per step. On a fast interconnect
that is a clear win; on PCIe it may not be. ``scripts/bench_distributed_muon.py`` measures
it rather than assuming.

**Measured so far (gloo/CPU, preset 46m, 121 2D tensors, 16-core box):**

===========  ===========  ==================
world size   step time    momentum state
===========  ===========  ==================
1            243 ms       177.0 MiB/rank
2            638 ms        89.0 MiB/rank
4            491 ms        45.0 MiB/rank
===========  ===========  ==================

The memory result is the designed one and is exact: state divides by world size. **The step
time is currently a regression**, and the honest reading is that this configuration has not
been shown to be a compute win. Three things confound the CPU numbers -- gloo is far slower
relative to compute than NCCL, N ranks on one 16-core box contend for the same cores so the
"NS compute / N" never materialises, and the benchmark is pure optimizer with no forward or
backward, so the all_gather is 100 % of the measured time instead of a few percent of a real
step. It still should not be reported as a win until multi-GPU says so.

The structural fix is the fused variant, which is why nanochat does not use DDP at all: it
``reduce_scatter``s gradients inside the optimizer, so grad-sync and state-sharding share one
collective and the total wire cost equals DDP's all-reduce -- the ``all_gather`` stops being
*extra*. Doing that here means excluding the Muon params from DDP's reducer
(``DistributedDataParallel._set_params_and_buffers_to_ignore_for_model``) and doing the
reduce-scatter in ``step()``. That is the next step, and it is what turns this from "memory
win, compute unproven" into a pure win.
"""

import math

import torch
import torch.distributed as dist

# Reuse torch's own kernels rather than reimplementing the algorithm: it makes this
# optimizer numerically identical to torch.optim.Muon by construction, which is exactly
# what tests/test_distributed_muon.py asserts. Private, so guarded.
try:
    from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DistributedMuon reuses torch.optim._muon internals; needs a torch with "
        "torch.optim.Muon (2.9+)."
    ) from exc

#: Params smaller than this stay replicated: every rank computes them redundantly and no
#: collective is issued. For Pos2MoveV2-large that is 9 of 13 shape buckets holding 0.2M of
#: 156.4M params — distributing them would cost 9 extra collectives to save nothing.
DEFAULT_MIN_DISTRIBUTED_NUMEL = 65_536


def muon_momentum(step: int, total_steps: int, warmup: int = 400,
                  warmdown_ratio: float = 0.2, lo: float = 0.85,
                  hi: float = 0.97, end: float = 0.90) -> float:
    """Momentum schedule: warm up 0.85 -> 0.97, then down to 0.90 during LR warmdown.

    Muon's momentum controls how much history the orthogonalized direction carries. Early
    on the gradient is changing fast and a long memory averages over a moving target, so it
    starts low; late in training a slightly *shorter* memory helps the run settle, which is
    why it comes back down rather than staying pinned at the peak. Schedule follows
    nanochat's ``get_muon_momentum``.
    """
    if step < warmup:
        frac = step / max(1, warmup)
        return (1 - frac) * lo + frac * hi
    warmdown_iters = max(1, round(warmdown_ratio * total_steps))
    warmdown_start = total_steps - warmdown_iters
    if step >= warmdown_start:
        progress = min(1.0, (step - warmdown_start) / warmdown_iters)
        return hi * (1 - progress) + end * progress
    return hi


def muon_weight_decay(step: int, total_steps: int, base_wd: float) -> float:
    """Cosine-decay weight decay to zero over the run, as nanochat's ``get_weight_decay``."""
    return base_wd * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total_steps))))


class DistributedMuon(torch.optim.Optimizer):
    """Drop-in for ``torch.optim.Muon`` that splits NS work and momentum across ranks.

    Assumes gradients are already synchronized (DDP has all-reduced them), so every rank
    starts from an identical gradient and only the *work* is divided. With no process group
    initialized it degenerates to exactly ``torch.optim.Muon``.
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True,
                 ns_coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7, ns_steps=5,
                 adjust_lr_fn=None, min_distributed_numel=DEFAULT_MIN_DISTRIBUTED_NUMEL):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_coefficients=ns_coefficients, eps=eps,
                        ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.min_distributed_numel = min_distributed_numel
        self._plan = [self._plan_group(g) for g in self.param_groups]

        owned = sum(len(b["owned"]) for p in self._plan for b in p["buckets"])
        total = sum(len(g["params"]) for g in self.param_groups)
        self.owned_param_count, self.total_param_count = owned, total

    def _plan_group(self, group):
        """Bucket a group's params by shape and assign contiguous chunks to ranks.

        Contiguous rather than strided so that ``all_gather_into_tensor`` — which lays rank
        i's contribution at output[i*chunk:(i+1)*chunk] — reassembles the bucket in order
        with no extra permutation.
        """
        by_shape: dict[tuple, list] = {}
        for p in group["params"]:
            if p.ndim != 2:
                raise ValueError(f"DistributedMuon handles 2D params only, got {p.ndim}D")
            by_shape.setdefault(tuple(p.shape), []).append(p)

        buckets = []
        for shape, plist in sorted(by_shape.items(), key=lambda kv: -kv[1][0].numel() * len(kv[1])):
            k = len(plist)
            distributed = (self.world_size > 1
                           and plist[0].numel() >= self.min_distributed_numel)
            if not distributed:
                # Replicated: every rank updates every param, no collective.
                buckets.append(dict(shape=shape, params=plist, owned=plist,
                                    offset=0, chunk=k, distributed=False))
                continue
            chunk = math.ceil(k / self.world_size)
            start = self.rank * chunk
            owned = plist[start:start + chunk]          # may be short on the last rank
            buckets.append(dict(shape=shape, params=plist, owned=owned,
                                offset=start, chunk=chunk, distributed=True))
        return dict(buckets=buckets)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, plan in zip(self.param_groups, self._plan):
            lr, wd = group["lr"], group["weight_decay"]
            momentum, nesterov = group["momentum"], group["nesterov"]
            for bucket in plan["buckets"]:
                for p in bucket["owned"]:
                    if p.grad is None:
                        continue
                    self._update_one(p, lr, wd, momentum, nesterov, group)
            for bucket in plan["buckets"]:
                if bucket["distributed"]:
                    self._publish(bucket)
        return loss

    def _update_one(self, p, lr, wd, momentum, nesterov, group):
        """Exactly torch.optim.Muon's ``_single_tensor_muon`` body, for one param."""
        grad = p.grad
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf
        update = _zeropower_via_newtonschulz(
            update, group["ns_coefficients"], group["ns_steps"], group["eps"])
        adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], p.shape)
        p.mul_(1 - lr * wd)
        p.add_(update, alpha=-adjusted_lr)

    def _publish(self, bucket):
        """One all_gather per shape bucket to give every rank the updated params.

        Zero-padding when K does not divide world_size: the tail slots are gathered and
        discarded, which costs a little bandwidth and keeps the collective's shape uniform.
        """
        plist, chunk, shape = bucket["params"], bucket["chunk"], bucket["shape"]
        k, ws = len(plist), self.world_size
        ref = plist[0]
        local = torch.zeros(chunk, *shape, dtype=ref.dtype, device=ref.device)
        for i, p in enumerate(bucket["owned"]):
            local[i].copy_(p)
        out = torch.empty(chunk * ws, *shape, dtype=ref.dtype, device=ref.device)
        dist.all_gather_into_tensor(out, local)
        for i in range(k):
            plist[i].copy_(out[i])

    def sharding_summary(self) -> str:
        parts = []
        for plan in self._plan:
            for b in plan["buckets"]:
                tag = "dist" if b["distributed"] else "repl"
                parts.append(f"{tuple(b['shape'])}x{len(b['params'])}:{tag}")
        return (f"DistributedMuon rank {self.rank}/{self.world_size}: owns "
                f"{self.owned_param_count}/{self.total_param_count} params | "
                + " ".join(parts))
