"""FP8 (e4m3) trunk matmuls against the bf16 forward self-play runs today: speed and numerics.

The profile (torch.profiler, 5070 Ti, batch 128) puts 81% of a forward in GEMMs, so quantizing
the trunk Linears is the only lever left. This swaps chosen nn.Linear layers for a dynamic
per-tensor FP8 version on torch._scaled_mm (weights quantized once, activations per call), keeps
everything else (norms, attention, smolgen, heads) in bf16, torch.compiles it like self-play does,
and reports evals/s plus the error against fp32 on real self-play positions.

    uv run python scripts/bench_fp8.py --model data/rl/scratch/best --positions data/rl/scratch/selfplay

Run it on an otherwise idle GPU. Variants: which layers are quantized (--variants).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_tensorrt import agreement, positions  # noqa: E402
from selfplay_rust import load_model  # noqa: E402

E4M3_MAX = 448.0
VARIANTS = {
    "bf16": (),
    "fc1+fc2": ("mlp.fc1", "mlp.fc2"),
    "fc1": ("mlp.fc1",),
    "attn": ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"),
    "all": ("mlp.fc1", "mlp.fc2", "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"),
}


class Fp8Linear(nn.Module):
    """y = x @ w.T + b with e4m3 operands and one dynamic scale per activation tensor."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        scale = (w.abs().amax() / E4M3_MAX).clamp(min=1e-12)
        self.register_buffer("wq", (w / scale).to(torch.float8_e4m3fn))
        self.register_buffer("w_scale", scale.reshape(()))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().bfloat16(), requires_grad=False)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        sx = (x2.abs().amax().float() / E4M3_MAX).clamp(min=1e-12)
        xq = (x2.float() / sx).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.wq.t(), scale_a=sx, scale_b=self.w_scale, out_dtype=torch.bfloat16)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], -1)


def quantize(model: nn.Module, suffixes: tuple[str, ...]) -> int:
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(full.endswith(s) for s in suffixes):
                setattr(mod, cname, Fp8Linear(child))
                n += 1
    return n


def evals_per_s(model, inputs, iters: int) -> float:
    for _ in range(10):
        model(*inputs)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        model(*inputs)
    b.record()
    torch.cuda.synchronize()
    return iters * inputs[0].shape[0] / (a.elapsed_time(b) / 1000)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--positions", required=True, help="self-play dir with positions*.npz")
    p.add_argument("--batches", default="128,256")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    model_dir = Path(args.model)

    (bt, pl, ca, ep), legal = positions(args.positions, 512)
    ref = load_model(model_dir, "cuda").float().eval()
    rl, rv = ref(bt, pl, ca, ep)
    rv = ref.value_scalar(rv.float()).squeeze(-1)
    del ref
    print(f"{'variant':9s} {'layers':>6s} | top-1 vs fp32 | logit err mean/max | value err mean/max | "
          + " | ".join(f"evals/s @{b}" for b in batches), flush=True)
    base = {}
    for name in args.variants.split(","):
        model = load_model(model_dir, "cuda").bfloat16().eval()
        n = quantize(model, VARIANTS[name])
        scalar = model.value_scalar
        fwd = torch.compile(model, mode="max-autotune")
        l, v = fwd(bt, pl, ca, ep)
        v = scalar(v.float()).squeeze(-1)
        a, dlm, dlx, dvm, dvx = agreement(l.float().cpu(), v.cpu(), rl.cpu(), rv.cpu(), legal)
        speeds = []
        for b in batches:
            ins = tuple(x[:b].contiguous() for x in (bt, pl, ca, ep))
            e = evals_per_s(fwd, ins, args.iters)
            base.setdefault(b, e)
            speeds.append(f"{e:9,.0f} (x{e / base[b]:.2f})")
        print(f"{name:9s} {n:6d} | {a:12.1%} | {dlm:.3f} / {dlx:.3f}    | {dvm:.4f} / {dvx:.4f}     | "
              + " | ".join(speeds), flush=True)
        del fwd, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
