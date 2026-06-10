"""Retrain the value head on self-play outcomes (step 2 of self-play RL).

The transformer trunk stays frozen, so the trunk's state features are computed
*once* over the whole dataset (one batched bf16 pass) and the tiny value-head
MLP is then trained on the cached features — full-dataset epochs take seconds,
so we can afford proper early stopping on held-out games.

Loss mirrors the original pretraining: MSE against the stm-POV game outcome
z ∈ {-1, 0, +1}, with early plies down-weighted by a halfmove ramp. Optionally
the target is softened toward the root MCTS value recorded at generation time
(KataGo-style mix), which reduces label noise from long games.

Validation is split by *game* (not position) to avoid leakage between
near-duplicate positions of the same game.

Usage
-----
    uv run python scripts/train_value_head.py \
        --data data/selfplay/v2.1 --base data/models/pos2move_v2.1 \
        --out data/models/pos2move_v2.1-spvalue
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


def load_shards(data_dir: Path) -> dict[str, np.ndarray]:
    shards = sorted(data_dir.rglob("positions_*.npz"))  # recursive: parallel workers write subdirs
    if not shards:
        raise FileNotFoundError(f"No positions_*.npz shards in {data_dir}")
    cols = ("boards", "player", "castling", "ep", "halfmove", "z", "root_v", "game_id")
    parts = {c: [] for c in cols}
    # Workers number their games from 0 in their own subdir, so namespace
    # game ids per directory to keep the by-game val split sound.
    dir_offset: dict = {}
    next_free = 0
    for s in shards:
        with np.load(s) as f:
            for c in cols:
                v = f[c]
                if c == "game_id":
                    if s.parent not in dir_offset:
                        dir_offset[s.parent] = next_free
                    v = v.astype(np.int64) + dir_offset[s.parent]
                    next_free = max(next_free, int(v.max()) + 1)
                parts[c].append(v)
    data = {c: np.concatenate(parts[c]) for c in cols}
    print(f"Loaded {len(data['z'])} positions / {len(np.unique(data['game_id']))} games "
          f"from {len(shards)} shards")
    return data


@torch.no_grad()
def extract_features(model: Pos2MoveV2, data: dict, device: str, batch: int) -> torch.Tensor:
    """One frozen-trunk pass over all positions -> (N, D) state features.

    Mirrors Pos2MoveV2.forward up to (but excluding) the heads: embeddings +
    transformer layers + final_norm, then the mean of the 3 game-state tokens.
    """
    n = len(data["z"])
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
    print(f"  features {n}/{n}")
    return feats


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (weight * (pred - target) ** 2).sum() / weight.sum()


@torch.no_grad()
def evaluate(head, feats, target, weight, z, device, batch=16384):
    """Weighted val MSE + sign accuracy on decisive positions."""
    se_sum, w_sum, sign_ok, sign_n = 0.0, 0.0, 0, 0
    for i in range(0, len(target), batch):
        sl = slice(i, i + batch)
        f = feats[sl].to(device).float()
        t, w, zz = target[sl].to(device), weight[sl].to(device), z[sl].to(device)
        pred = head(f).squeeze(-1).float()
        se_sum += (w * (pred - t) ** 2).sum().item()
        w_sum += w.sum().item()
        decisive = zz != 0
        sign_ok += ((pred[decisive] > 0) == (zz[decisive] > 0)).sum().item()
        sign_n += int(decisive.sum().item())
    return se_sum / max(w_sum, 1e-9), sign_ok / max(sign_n, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="dir of positions_*.npz from selfplay_value_games.py")
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--feature-batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--soft-mix", type=float, default=0.0,
                   help="target = (1-mix)*z + mix*root_mcts_value")
    p.add_argument("--ramp-halfmoves", type=int, default=40,
                   help="down-weight plies before this (matches pretraining); 0 disables")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    base = Path(args.base)
    data = load_shards(Path(args.data))
    model = load_model(base, args.device).bfloat16()

    feats = extract_features(model, data, args.device, args.feature_batch)

    z = torch.from_numpy(data["z"]).float()
    root_v = torch.from_numpy(data["root_v"]).float()
    target = (1.0 - args.soft_mix) * z + args.soft_mix * root_v
    if args.ramp_halfmoves > 0:
        weight = (torch.from_numpy(data["halfmove"].astype(np.float32)) / args.ramp_halfmoves).clamp(max=1.0)
    else:
        weight = torch.ones_like(z)

    # Split by game so near-duplicate positions of one game can't leak.
    games = np.unique(data["game_id"])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(games)
    val_games = set(games[: max(1, int(args.val_frac * len(games)))].tolist())
    val_mask = torch.from_numpy(np.isin(data["game_id"], list(val_games)))
    tr_idx = torch.nonzero(~val_mask).squeeze(1)
    va_idx = torch.nonzero(val_mask).squeeze(1)
    print(f"Train {len(tr_idx)} / val {len(va_idx)} positions "
          f"({len(games) - len(val_games)}/{len(val_games)} games)")

    # Value head trains in fp32 on the cached features.
    head = copy.deepcopy(model.value_head).float().to(args.device)
    base_loss, base_sign = evaluate(head, feats[va_idx], target[va_idx], weight[va_idx],
                                    z[va_idx], args.device)
    print(f"Baseline (pretrained head): val MSE {base_loss:.4f}, decisive sign-acc {base_sign:.1%}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for i in range(0, len(perm), args.batch):
            idx = perm[i: i + args.batch]
            f = feats[idx].to(args.device).float()
            pred = head(f).squeeze(-1)
            loss = weighted_mse(pred, target[idx].to(args.device), weight[idx].to(args.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        val_loss, val_sign = evaluate(head, feats[va_idx], target[va_idx], weight[va_idx],
                                      z[va_idx], args.device)
        marker = ""
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(head.state_dict())
            marker = " *"
        print(f"epoch {epoch:3d}: val MSE {val_loss:.4f}, sign-acc {val_sign:.1%}{marker}")

    head.load_state_dict(best_state)
    head.eval()
    final_loss, final_sign = evaluate(head, feats[va_idx], target[va_idx], weight[va_idx],
                                      z[va_idx], args.device)
    print(f"\nBest epoch {best_epoch}: val MSE {base_loss:.4f} -> {final_loss:.4f} "
          f"({(1 - final_loss / base_loss):+.1%}), sign-acc {base_sign:.1%} -> {final_sign:.1%}")

    # Export: base weights with only value_head.* swapped, in the bot's layout.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    prefix = "_orig_mod." if any(k.startswith("_orig_mod.") for k in sd) else ""
    for k, v in head.state_dict().items():
        key = f"{prefix}value_head.{k}"
        sd[key] = v.cpu().to(sd[key].dtype)
    save_file(sd, str(out / "model.safetensors"))
    shutil.copy(base / "model_config.json", out / "model_config.json")
    (out / "value_training_meta.json").write_text(json.dumps({
        **vars(args),
        "positions": len(z), "games": len(games),
        "baseline_val_mse": base_loss, "final_val_mse": final_loss,
        "baseline_sign_acc": base_sign, "final_sign_acc": final_sign,
        "best_epoch": best_epoch,
    }, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
