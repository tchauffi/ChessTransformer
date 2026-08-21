"""Retrain the value head on Stockfish evals (frozen trunk).

Companion to `scripts/prep_eval_value_data.py`. The transformer trunk stays
frozen, so its 256-dim state features are computed **once** over the whole
dataset (one batched bf16 pass) and only the tiny value-head MLP trains on the
cached features — full-dataset epochs take seconds, so we can afford long
early-stopping.

Target is the side-to-move-POV value in [-1, 1] derived from Stockfish cp/mate
(MSE), unlike the original outcome-based retraining. Positions are independent
(deduped by FEN), so the validation split is a plain random split by position.
The pretrained head's val MSE is reported as the baseline.

NOTE: lower val MSE does NOT imply more playing strength here (val loss has
historically anti-correlated with MCTS strength). Gate the exported model with
scripts/engine_match.py before trusting it.

`--bounded` appends a tanh to the head. The first run of this script (exported
as pos2move_v2.1-sfvalue) fit the target best of any head so far yet gated worst
(33.3%), and the diagnosis was overconfidence: 11.2% of the cp target sits exactly
at +-1 (mates), and an unbounded head fitted with MSE extrapolates past it (2.9% of
outputs beyond +-0.99, reaching +-1.18). Those spurious near-certain leaves are
what MCTS amplifies. tanh makes that failure unrepresentable.

Usage
-----
    uv run python scripts/train_value_head.py \
        --data data/eval/lichess-sf --base data/models/pos2move_v2.1 \
        --out data/models/pos2move_v2.1-sfvalue-bounded --bounded
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2


def load_model(base: Path, device: str) -> Pos2MoveV2:
    cfg = json.loads((base / "model_config.json").read_text())
    m = Pos2MoveV2(**cfg)
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.eval().to(device)


def load_data(data_dir: Path) -> dict[str, np.ndarray]:
    npz = data_dir / "positions_eval.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found (run scripts/prep_eval_value_data.py)")
    with np.load(npz) as f:
        data = {k: f[k] for k in ("boards", "player", "castling", "ep", "value", "decisive")}
    print(f"Loaded {len(data['value']):,} positions | decisive {data['decisive'].mean():.1%}")
    return data


@torch.no_grad()
def extract_features(model: Pos2MoveV2, data: dict, device: str, batch: int) -> torch.Tensor:
    """One frozen-trunk pass over all positions -> (N, D) state features.

    Mirrors Pos2MoveV2.forward up to (but excluding) the heads: embeddings +
    transformer layers + final_norm, then the mean of the 3 game-state tokens.
    """
    n = len(data["value"])
    bt = torch.from_numpy(data["boards"].astype(np.int64))
    pl = torch.from_numpy(data["player"].astype(np.int64))
    ca = torch.from_numpy(data["castling"].astype(np.int64))
    ep = torch.from_numpy(data["ep"].astype(np.int64))

    pos_emb = model.position_embedding(model.pos_index)  # (67, D)
    feats = torch.empty(n, pos_emb.size(-1), dtype=torch.float16)
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        b, p, c, e = (t[sl].to(device) for t in (bt, pl, ca, ep))
        x = model.token_embedding(b) + pos_emb[:64]
        x = torch.cat([
            x,
            (model.castling_embedding(c) + pos_emb[64]).unsqueeze(1),
            (model.en_passant_embedding(e) + pos_emb[65]).unsqueeze(1),
            (model.player_embedding(p) + pos_emb[66]).unsqueeze(1),
        ], dim=1)
        for layer in model.transformer_layers:
            x = layer(x)
        x = model.final_norm(x)
        feats[sl] = x[:, -3:, :].mean(dim=1).half().cpu()
        if (i // batch) % 50 == 0:
            print(f"  features {i}/{n}", end="\r")
    print(f"  features {n}/{n}      ")
    return feats


def sharpen_target(value: np.ndarray, s: float) -> np.ndarray:
    """Reshape the cp-derived target in atanh space, leaving mates pinned.

    The target is v = tanh(k*cp/2), so tanh(S*atanh(v)) is exactly what
    re-deriving it with k*S would give -- letting us vary target confidence
    without re-downloading the eval parquet. Mates carry no finite cp, so they
    stay at +-1 (a mate really is certain).
    """
    if s == 1.0:
        return value
    mate = np.abs(value) >= 1.0
    safe = np.clip(value, -1 + 1e-6, 1 - 1e-6)
    out = np.tanh(s * np.arctanh(safe)).astype(np.float32)
    out[mate] = value[mate]
    return out


@torch.no_grad()
def evaluate(head, feats, target, decisive, device, batch=16384):
    """Val MSE + sign accuracy on decisive positions."""
    se_sum, n_sum, sign_ok, sign_n = 0.0, 0, 0, 0
    for i in range(0, len(target), batch):
        sl = slice(i, i + batch)
        f = feats[sl].to(device).float()
        t, dec = target[sl].to(device), decisive[sl].to(device)
        pred = head(f).squeeze(-1).float()
        se_sum += ((pred - t) ** 2).sum().item()
        n_sum += len(t)
        sign_ok += ((pred[dec] > 0) == (t[dec] > 0)).sum().item()
        sign_n += int(dec.sum().item())
    return se_sum / max(n_sum, 1), sign_ok / max(sign_n, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--feature-batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-sharpness", type=float, default=1.0, metavar="S",
                   help="reshape the cp target in atanh space: v' = tanh(S * atanh(v)). "
                        "S<1 flattens (less confident), S>1 sharpens, S=1 leaves it alone. "
                        "Mates stay pinned at +-1. Exactly equivalent to re-deriving the "
                        "target with cp_k scaled by S, but done from the stored npz -- no "
                        "re-download. NOTE: val MSE is NOT comparable across S values, the "
                        "target itself differs. Gate by games.")
    p.add_argument("--feature-cache", metavar="PATH",
                   help="save/reuse the frozen-trunk features. They depend only on --base "
                        "and --data, so a sweep over --target-sharpness extracts them once.")
    p.add_argument("--bounded", action="store_true",
                   help="append a tanh to the value head so it cannot predict outside "
                        "[-1, 1]. 11.2%% of the cp target sits exactly at +-1 (mates), and an "
                        "unbounded head fitted with MSE learns to extrapolate past it; "
                        "those spurious near-certain leaves are what MCTS amplifies.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    base = Path(args.base)
    data = load_data(Path(args.data))
    model = load_model(base, args.device).bfloat16()

    cache = Path(args.feature_cache) if args.feature_cache else None
    if cache and cache.exists():
        feats = torch.load(cache)
        print(f"Loaded cached features {tuple(feats.shape)} from {cache}")
    else:
        feats = extract_features(model, data, args.device, args.feature_batch)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feats, cache)
            print(f"Cached features -> {cache}")

    raw = data["value"].astype(np.float32)
    value = sharpen_target(raw, args.target_sharpness)
    if args.target_sharpness != 1.0:
        print(f"Target sharpness {args.target_sharpness}: std {raw.std():.4f} -> "
              f"{value.std():.4f}, |v|>0.99 {np.mean(np.abs(raw) > 0.99):.2%} -> "
              f"{np.mean(np.abs(value) > 0.99):.2%}")
    target = torch.from_numpy(value)
    decisive = torch.from_numpy(data["decisive"].astype(bool))

    # Plain random split by position (positions are FEN-deduped, no leakage).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(target))
    n_val = max(1, int(args.val_frac * len(target)))
    va_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])
    print(f"Train {len(tr_idx):,} / val {len(va_idx):,} positions")

    # Value head trains in fp32 on the cached features.
    head = copy.deepcopy(model.value_head).float().to(args.device)
    # Baseline is always the pretrained *unbounded* head, so the number stays
    # comparable across bounded and unbounded runs.
    base_loss, base_sign = evaluate(head, feats[va_idx], target[va_idx], decisive[va_idx], args.device)
    print(f"Baseline (pretrained head): val MSE {base_loss:.4f}, decisive sign-acc {base_sign:.1%}")
    if args.bounded:
        # tanh at index 3 leaves the Linear keys (0.*, 2.*) untouched, and the
        # pretrained head already outputs roughly [-1, 1], so tanh(x) ~ x here:
        # a benign init rather than a cold start.
        head = torch.nn.Sequential(*head, torch.nn.Tanh()).to(args.device)
        bound_loss, bound_sign = evaluate(head, feats[va_idx], target[va_idx], decisive[va_idx], args.device)
        print(f"  + tanh bound, before training: val MSE {bound_loss:.4f}, sign-acc {bound_sign:.1%}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = tr_idx[torch.randperm(len(tr_idx))]
        for i in range(0, len(order), args.batch):
            idx = order[i: i + args.batch]
            f = feats[idx].to(args.device).float()
            pred = head(f).squeeze(-1)
            loss = ((pred - target[idx].to(args.device)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        val_loss, val_sign = evaluate(head, feats[va_idx], target[va_idx], decisive[va_idx], args.device)
        marker = ""
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(head.state_dict())
            bad, marker = 0, " *"
        else:
            bad += 1
        print(f"epoch {epoch:3d}: val MSE {val_loss:.4f}, sign-acc {val_sign:.1%}{marker}")
        if bad >= args.patience:
            print(f"  early stop: no improvement for {args.patience} epochs")
            break

    head.load_state_dict(best_state)
    head.eval()
    final_loss, final_sign = evaluate(head, feats[va_idx], target[va_idx], decisive[va_idx], args.device)
    print(f"\nBest epoch {best_epoch}: val MSE {base_loss:.4f} -> {final_loss:.4f} "
          f"({(1 - final_loss / base_loss):+.1%}), sign-acc {base_sign:.1%} -> {final_sign:.1%}")

    # Export: base weights with only value_head.* swapped, in the bot's layout.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    prefix = "_orig_mod." if any(k.startswith("_orig_mod.") for k in sd) else ""
    for k, v in head.state_dict().items():
        sd[f"{prefix}value_head.{k}"] = v.cpu().to(sd[f"{prefix}value_head.{k}"].dtype)
    save_file(sd, str(out / "model.safetensors"))
    # A bounded head only stays bounded if every loader rebuilds the tanh, so the
    # flag has to travel with the weights.
    cfg = json.loads((base / "model_config.json").read_text())
    cfg["value_bounded"] = bool(args.bounded)
    (out / "model_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "value_training_meta.json").write_text(json.dumps({
        "data": args.data, "base": str(base), "positions": len(target),
        "bounded": bool(args.bounded), "target_sharpness": args.target_sharpness,
        "baseline_val_mse": base_loss, "final_val_mse": final_loss,
        "baseline_sign_acc": base_sign, "final_sign_acc": final_sign,
        "best_epoch": best_epoch, "lr": args.lr,
    }, indent=2))
    print(f"Saved to {out}. Gate vs base with scripts/engine_match.py before promoting.")


if __name__ == "__main__":
    main()
