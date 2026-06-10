"""Trunk-unfrozen value training on self-play outcomes (step 1b of self-play RL).

The head-only experiment (scripts/train_value_head.py) was a null result: given
the frozen trunk's features, the pretrained value head was already near-optimal.
So the trunk representation is the bottleneck. This script unfreezes everything
and lets the value gradient reshape the trunk, while the policy is protected by
a KL anchor to a frozen reference model:

    loss = ramp-weighted value MSE  +  beta_pol * KL(model_policy ‖ ref_policy)

The KL is computed over the full flattened 64×73 action space (fp32), which is
fully batched and anchors the policy *function*, not the parameters — the trunk
may move however it likes as long as policy outputs stay close. Policy drift is
also logged as argmax agreement with the reference so degradation is visible
long before an engine match.

Usage
-----
    uv run python scripts/train_value_trunk.py \
        --data data/selfplay/v2.1-128sims --base data/models/pos2move_v2.1 \
        --out data/models/pos2move_v2.1-spvalue-trunk
"""

from __future__ import annotations

import argparse
import copy
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

from train_value_head import load_model, load_shards


def make_tensors(data: dict) -> dict[str, torch.Tensor]:
    return {
        "boards": torch.from_numpy(data["boards"].astype(np.int64)),
        "player": torch.from_numpy(data["player"].astype(np.int64)),
        "castling": torch.from_numpy(data["castling"].astype(np.int64)),
        "ep": torch.from_numpy(data["ep"].astype(np.int64)),
        "z": torch.from_numpy(data["z"]).float(),
        "root_v": torch.from_numpy(data["root_v"]).float(),
        "halfmove": torch.from_numpy(data["halfmove"].astype(np.float32)),
    }


def batch_forward(model, t: dict, idx: torch.Tensor, device: str):
    b = t["boards"][idx].to(device, non_blocking=True)
    p = t["player"][idx].to(device, non_blocking=True)
    c = t["castling"][idx].to(device, non_blocking=True)
    e = t["ep"][idx].to(device, non_blocking=True)
    return model(b, p, c, e)


@torch.no_grad()
def evaluate(model, ref, t, idx, target, weight, device, batch=4096):
    """Val metrics: weighted value MSE, sign-acc on decisive, policy KL to ref,
    policy argmax agreement with ref."""
    model.eval()
    se, wsum, sign_ok, sign_n, kl_sum, agree, n = 0.0, 0.0, 0, 0, 0.0, 0, 0
    for i in range(0, len(idx), batch):
        sl = idx[i: i + batch]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, value = batch_forward(model, t, sl, device)
            ref_logits, _ = batch_forward(ref, t, sl, device)
        value = value.squeeze(-1).float()
        tt, ww = target[sl].to(device), weight[sl].to(device)
        z = t["z"][sl].to(device)
        se += (ww * (value - tt) ** 2).sum().item()
        wsum += ww.sum().item()
        decisive = z != 0
        sign_ok += ((value[decisive] > 0) == (z[decisive] > 0)).sum().item()
        sign_n += int(decisive.sum().item())
        lp = F.log_softmax(logits.view(len(sl), -1).float(), dim=-1)
        ref_p = F.softmax(ref_logits.view(len(sl), -1).float(), dim=-1)
        kl_sum += (ref_p * (ref_p.clamp_min(1e-12).log() - lp)).sum().item()
        agree += (lp.argmax(-1) == ref_p.argmax(-1)).sum().item()
        n += len(sl)
    model.train()
    return se / max(wsum, 1e-9), sign_ok / max(sign_n, 1), kl_sum / n, agree / n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--beta-pol", type=float, default=1.0,
                   help="weight of the KL(model ‖ ref) policy anchor")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--soft-mix", type=float, default=0.0,
                   help="value target = (1-mix)*z + mix*root_mcts_value")
    p.add_argument("--ramp-halfmoves", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=100, help="steps between val evals")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    base = Path(args.base)
    data = load_shards(Path(args.data))
    t = make_tensors(data)

    target = (1.0 - args.soft_mix) * t["z"] + args.soft_mix * t["root_v"]
    if args.ramp_halfmoves > 0:
        weight = (t["halfmove"] / args.ramp_halfmoves).clamp(max=1.0)
    else:
        weight = torch.ones_like(t["z"])

    games = np.unique(data["game_id"])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(games)
    val_games = set(games[: max(1, int(args.val_frac * len(games)))].tolist())
    val_mask = torch.from_numpy(np.isin(data["game_id"], list(val_games)))
    tr_idx = torch.nonzero(~val_mask).squeeze(1)
    va_idx = torch.nonzero(val_mask).squeeze(1)
    print(f"Train {len(tr_idx)} / val {len(va_idx)} positions "
          f"({len(games) - len(val_games)}/{len(val_games)} games)")

    model = load_model(base, args.device).float()
    ref = load_model(base, args.device).float()
    for pr in ref.parameters():
        pr.requires_grad_(False)
    ref.eval()

    base_mse, base_sign, _, _ = evaluate(model, ref, t, va_idx, target, weight, args.device)
    print(f"Baseline (pretrained model): val MSE {base_mse:.4f}, sign-acc {base_sign:.1%}")
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Seed "best" with the baseline: if training never improves val MSE, we
    # export the pretrained weights unchanged rather than a degraded model.
    best = {"mse": base_mse, "state": copy.deepcopy(model.state_dict()), "step": 0}
    step = 0
    for epoch in range(1, args.epochs + 1):
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[i: i + args.batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                logits, value = batch_forward(model, t, idx, args.device)
                with torch.no_grad():
                    ref_logits, _ = batch_forward(ref, t, idx, args.device)
            value = value.squeeze(-1).float()
            ww = weight[idx].to(args.device)
            v_loss = (ww * (value - target[idx].to(args.device)) ** 2).sum() / ww.sum().clamp_min(1e-9)
            lp = F.log_softmax(logits.view(len(idx), -1).float(), dim=-1)
            ref_p = F.softmax(ref_logits.view(len(idx), -1).float(), dim=-1)
            kl = (ref_p * (ref_p.clamp_min(1e-12).log() - lp)).sum(-1).mean()
            loss = v_loss + args.beta_pol * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % args.eval_every == 0:
                mse, sign, vkl, agree = evaluate(model, ref, t, va_idx, target, weight, args.device)
                marker = ""
                if mse < best["mse"]:
                    best = {"mse": mse, "state": copy.deepcopy(model.state_dict()), "step": step}
                    marker = " *"
                print(f"epoch {epoch} step {step}: train v={v_loss.item():.4f} kl={kl.item():.5f} | "
                      f"val MSE {mse:.4f} sign {sign:.1%} KL {vkl:.5f} agree {agree:.1%}{marker}",
                      flush=True)

    model.load_state_dict(best["state"])
    model.eval()
    mse, sign, vkl, agree = evaluate(model, ref, t, va_idx, target, weight, args.device)
    print(f"\nBest step {best['step']}: val MSE {base_mse:.4f} -> {mse:.4f} "
          f"({(1 - mse / base_mse):+.1%}), sign-acc {base_sign:.1%} -> {sign:.1%}, "
          f"policy KL {vkl:.5f}, agreement {agree:.1%}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    shutil.copy(base / "model_config.json", out / "model_config.json")
    (out / "value_training_meta.json").write_text(json.dumps({
        **vars(args), "positions": len(t["z"]), "games": len(games),
        "baseline_val_mse": base_mse, "final_val_mse": mse,
        "baseline_sign_acc": base_sign, "final_sign_acc": sign,
        "final_policy_kl": vkl, "final_policy_agreement": agree,
        "best_step": best["step"],
    }, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
