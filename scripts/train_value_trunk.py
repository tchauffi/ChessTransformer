#!/usr/bin/env python3
"""Retrain the value head *with the trunk unfrozen*, on Stockfish cp targets.

The experiment doc's step 1 froze the trunk and computed the 256-d features
once, then trained only the ~16.5k-parameter head. It returned a null: val MSE
0.2406 -> 0.2369 and a 46.9% gate. The conclusion drawn was that "given the
frozen trunk's features, the pretrained head was already near-optimal -- the
binding constraint is the **trunk representation**, not the value labels."

That conclusion names the experiment that was never run. This is it.

An attempt exists in the record and did not survive: `logs/train_value_trunk.log`
shows the baseline printed (val MSE 0.2405, sign-acc 84.8%) and then a CUDA OOM
at 14.4 GB of 15.46 GB, before a single optimizer step. Unfreezing 16 layers
means storing their activations, and the head-only script's 8192 batch does not
survive that. Hence bf16 autocast, a modest batch, and gradient accumulation.

Why the value head is worth the compute
---------------------------------------
`docs/strength_vs_sims.json` measures 1327 Elo at 25 sims and 2175 at 800 --
about +170 Elo per doubling, and *every* one of those simulations is scored by
the value head at a leaf. Meanwhile the policy axis has now failed five separate
times, most recently at -132 Elo. Leaf evaluation is where the remaining
leverage plausibly is.

Two changes carried over from the sharpness sweep, both measured there:
* `--bounded` appends a tanh. v2.1 emits |v| > 1 on 0.76% of positions (max
  1.039) and MCTS maximises over leaf values, so it actively hunts that tail.
  The pretrained head already outputs roughly [-1, 1], so the tanh is a benign
  init rather than a cold start.
* `--target-sharpness S` reshapes the target in atanh space, leaving mates
  pinned at +-1. S ~ 3 was the sweep's plateau.

Preserving the policy
---------------------
Unfreezing the trunk puts the policy at risk, and a degraded policy cancelled
the value gains in both expert-iteration runs. A forward KL to the frozen base
holds it in place. The KL is taken over the **full 4672-action softmax** rather
than the legal set, because this dataset stores encoded positions without legal
move lists; over-constraining illegal actions is harmless, since they are masked
at inference anyway.

Note the Adam caveat learned in `grpo_selfplay.py`: the coefficient shapes the
update's direction but does not bound its distance, because Adam normalises per
parameter. Step count and learning rate are what bound drift, which is why
checkpoints are saved periodically -- a run is a ladder of candidates, and only
the match harness picks between them.

Usage
-----
    uv run python scripts/train_value_trunk.py \
        --data data/eval/lichess-sf/positions_eval.npz \
        --base data/models/pos2move_v2.1 --out data/models/v2.1-valuetrunk \
        --bounded --target-sharpness 3.0 --steps 6000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo_puzzles import load_model  # noqa: E402


def sharpen_target(value: np.ndarray, s: float) -> np.ndarray:
    """Reshape the cp-derived target in atanh space, leaving mates pinned.

    The target is v = tanh(k*cp/2), so tanh(S*atanh(v)) is exactly what
    re-deriving it with k*S would give. Mates carry no finite cp, so they stay
    at +-1 -- a mate really is certain. Identical to scripts/train_value_head.py
    so the two are comparable.
    """
    if s == 1.0:
        return value
    mate = np.abs(value) >= 1.0
    safe = np.clip(value, -1 + 1e-6, 1 - 1e-6)
    out = np.tanh(s * np.arctanh(safe)).astype(np.float32)
    out[mate] = value[mate]
    return out


def bind_tanh(model) -> None:
    """Append a tanh to the value head, keeping state-dict keys stable.

    The squashing lands at index 3, so the Linear keys (0.*, 2.*) are untouched
    and checkpoints stay interchangeable with unbounded ones.
    """
    if not isinstance(model.value_head[-1], torch.nn.Tanh):
        model.value_head = torch.nn.Sequential(*model.value_head, torch.nn.Tanh())


class Positions:
    """The npz from scripts/prep_eval_value_data.py."""

    def __init__(self, path: Path, sharpness: float):
        d = np.load(path)
        self.boards = torch.from_numpy(d["boards"].astype(np.int64))
        self.player = torch.from_numpy(d["player"].astype(np.int64))
        self.castling = torch.from_numpy(d["castling"].astype(np.int64))
        self.ep = torch.from_numpy(d["ep"].astype(np.int64))
        raw = d["value"].astype(np.float32)
        self.raw = torch.from_numpy(raw)
        self.target = torch.from_numpy(sharpen_target(raw, sharpness))
        self.decisive = torch.from_numpy(d["decisive"])
        self.n = len(self.boards)

    def batch(self, rows: np.ndarray, device: str):
        r = torch.from_numpy(rows)
        return (self.boards[r].to(device), self.player[r].to(device),
                self.castling[r].to(device), self.ep[r].to(device),
                self.target[r].to(device), self.decisive[r].to(device))


@torch.no_grad()
def evaluate(model, ref, data, rows, device, batch_size):
    """Value quality plus a policy-drift check, in eval mode."""
    model.eval()
    se = n = sign_ok = n_dec = sat = agree = 0
    conf_sum = acc_sum = 0.0
    for start in range(0, len(rows), batch_size):
        sl = rows[start:start + batch_size]
        b, pl, ca, ep, tgt, dec = data.batch(sl, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, v = model(b, pl, ca, ep)
            ref_logits, _ = ref(b, pl, ca, ep)
        v = v.float().squeeze(-1)
        se += float(((v - tgt) ** 2).sum())
        n += len(sl)
        sat += int((v.abs() > 0.99).sum())
        if dec.any():
            sign_ok += int((torch.sign(v[dec]) == torch.sign(tgt[dec])).sum())
            n_dec += int(dec.sum())
        # Policy drift: does the net still want the same move as the base?
        agree += int((logits.view(len(sl), -1).argmax(1)
                      == ref_logits.view(len(sl), -1).argmax(1)).sum())
        # Calibration: |v| as a confidence, correctness of its sign.
        if dec.any():
            conf_sum += float(v[dec].abs().sum())
            acc_sum += float((torch.sign(v[dec]) == torch.sign(tgt[dec])).float().sum())
    model.train()
    return {
        "mse": se / max(n, 1),
        "sign_acc": sign_ok / max(n_dec, 1),
        "saturated": sat / max(n, 1),
        "policy_agree": agree / max(n, 1),
        "ece_gap": abs(conf_sum - acc_sum) / max(n_dec, 1),
    }


def save_checkpoint(model, args, step: int, final: bool = False) -> Path:
    out = args.out if final else args.out / f"step_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    cfg = json.loads((Path(args.base) / "model_config.json").read_text())
    # A bounded head only stays bounded if every loader rebuilds the tanh.
    cfg["value_bounded"] = bool(args.bounded)
    (out / "model_config.json").write_text(json.dumps(cfg, indent=2))
    (out / "value_training_meta.json").write_text(json.dumps({
        "base": str(args.base), "data": str(args.data), "step": step,
        "bounded": bool(args.bounded), "target_sharpness": args.target_sharpness,
        "policy_kl_coef": args.policy_kl_coef, "lr": args.lr,
        "trunk": "unfrozen",
    }, indent=1))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path,
                   default=Path("data/eval/lichess-sf/positions_eval.npz"))
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=384,
                   help="kept modest on purpose: unfreezing 16 layers stores "
                        "their activations, and the head-only script's 8192 "
                        "OOMed at 14.4GB before its first step.")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="fine-tuning rate for a pretrained trunk. With Adam this "
                        "and --steps are what actually bound drift.")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--policy-kl-coef", type=float, default=1.0,
                   help="forward KL to the frozen base policy over all 4672 "
                        "actions. A degraded policy cancelled the value gains "
                        "in both expert-iteration runs.")
    p.add_argument("--target-sharpness", type=float, default=3.0, metavar="S")
    p.add_argument("--bounded", action=argparse.BooleanOptionalAction, default=True,
                   help="append tanh to the value head")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = Positions(args.data, args.target_sharpness)
    perm = rng.permutation(data.n)
    n_val = max(1, int(data.n * args.val_frac))
    val_rows, train_rows = perm[:n_val], perm[n_val:]
    print(f"{data.n:,} positions: {len(train_rows):,} train / {len(val_rows):,} val")
    if args.target_sharpness != 1.0:
        print(f"target sharpness {args.target_sharpness}: std "
              f"{data.raw.std():.4f} -> {data.target.std():.4f}")

    model = load_model(args.base, args.device)
    ref = load_model(args.base, args.device).eval()
    for pr in ref.parameters():
        pr.requires_grad_(False)
    if args.bounded:
        bind_tanh(model)
        model.to(args.device)

    base_m = evaluate(model, ref, data, val_rows, args.device, args.batch_size)
    print(f"base{' +tanh' if args.bounded else ''} (val): "
          f"MSE {base_m['mse']:.4f}  sign-acc {base_m['sign_acc']:.1%}  "
          f"|v|>0.99 {base_m['saturated']:.3%}  policy-agree {base_m['policy_agree']:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))

    args.out.mkdir(parents=True, exist_ok=True)
    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        agg_v = agg_kl = 0.0
        for _ in range(args.grad_accum):
            rows = rng.choice(train_rows, size=args.batch_size, replace=False)
            b, pl, ca, ep, tgt, _dec = data.batch(rows, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.device == "cuda"):
                logits, v = model(b, pl, ca, ep)
                with torch.no_grad():
                    ref_logits, _ = ref(b, pl, ca, ep)
            v = v.float().squeeze(-1)
            value_loss = ((v - tgt) ** 2).mean()
            lp = F.log_softmax(logits.view(len(rows), -1).float(), dim=1)
            rlp = F.log_softmax(ref_logits.view(len(rows), -1).float(), dim=1)
            kl = (lp.exp() * (lp - rlp)).sum(1).mean()
            (value_loss + args.policy_kl_coef * kl).div(args.grad_accum).backward()
            agg_v += float(value_loss.detach()) / args.grad_accum
            agg_kl += float(kl.detach()) / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()

        if step % 50 == 0:
            print(f"step {step:5d}  value_mse {agg_v:.4f}  policy_kl {agg_kl:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step % args.eval_steps == 0:
            m = evaluate(model, ref, data, val_rows, args.device, args.batch_size)
            warn = "" if m["policy_agree"] > 0.9 else "   <-- POLICY DRIFTING"
            print(f"  [val @ {step}] MSE {m['mse']:.4f} (base {base_m['mse']:.4f})  "
                  f"sign-acc {m['sign_acc']:.1%} (base {base_m['sign_acc']:.1%})  "
                  f"|v|>0.99 {m['saturated']:.3%}  "
                  f"policy-agree {m['policy_agree']:.3f}{warn}", flush=True)
        if step % args.save_steps == 0:
            print(f"  saved {save_checkpoint(model, args, step)}", flush=True)

    save_checkpoint(model, args, args.steps, final=True)
    m = evaluate(model, ref, data, val_rows, args.device, args.batch_size)
    print(f"\nfinal (val): MSE {m['mse']:.4f} (base {base_m['mse']:.4f})  "
          f"sign-acc {m['sign_acc']:.1%} (base {base_m['sign_acc']:.1%})  "
          f"policy-agree {m['policy_agree']:.3f}")
    print("\nNone of these numbers decide promotion -- on this project they have "
          "anti-correlated with strength five times. Gate with:")
    print(f"  OPENINGS=1000 WORKERS=4 bash scripts/gate_candidate.sh {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
