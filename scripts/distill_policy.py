"""Distill a sharper policy into Pos2MoveV2 from Stockfish soft labels.

Fine-tunes the model with a KL-divergence loss toward the Stockfish move
distribution (from `gen_sf_policy_labels.py`). The value head is pinned to a
frozen teacher copy via an MSE preservation term, so the (already strong) value
estimates don't regress while the policy improves.

Usage
-----
    uv run python scripts/distill_policy.py \
        --labels data/distill/sf_policy_poc.npz \
        --base data/models/pos2move_v2.1 \
        --out data/models/pos2move_v2.1-distill \
        --epochs 6 --lr 2e-4
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
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.models.transformer.pos2move_v2 import NUM_ACTION_PLANES, Pos2MoveV2

ACTION_DIM = 64 * NUM_ACTION_PLANES  # 4672


class SFLabelDataset(Dataset):
    def __init__(self, npz_path: str, temp: float = 50.0):
        d = np.load(npz_path)
        self.tokens = d["tokens"].astype(np.int64)
        self.player = d["player"].astype(np.int64)
        self.castling = d["castling"].astype(np.int64)
        self.ep = d["ep"].astype(np.int64)
        self.tgt_idx = d["tgt_idx"].astype(np.int64)
        self.temp = temp
        # New format stores raw centipawns (temperature applied here); old PoC
        # files stored pre-softmaxed probabilities.
        if "tgt_cp" in d.files:
            self.tgt_cp = d["tgt_cp"].astype(np.float32)
            self.mode = "cp"
        else:
            self.tgt_prob = d["tgt_prob"].astype(np.float32)
            self.mode = "prob"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        target = torch.zeros(ACTION_DIM, dtype=torch.float32)
        idx = self.tgt_idx[i]
        valid = idx >= 0
        if self.mode == "cp":
            cp = self.tgt_cp[i][valid] / self.temp
            cp = cp - cp.max()
            w = np.exp(cp); w = w / w.sum()
        else:
            w = self.tgt_prob[i][valid]
        target[idx[valid]] = torch.from_numpy(w.astype(np.float32))
        return (
            torch.from_numpy(self.tokens[i]),
            int(self.player[i]), int(self.castling[i]), int(self.ep[i]),
            target,
        )


def load_model(base_dir: str, device: str) -> Pos2MoveV2:
    base = Path(base_dir)
    config = json.loads((base / "model_config.json").read_text())
    model = Pos2MoveV2(**config)
    with safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--value-pin", type=float, default=1.0, help="weight on value-preservation MSE")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="train only the policy head (move_head); backbone + value head frozen "
                        "so value output is identical and policy can't overfit the backbone")
    p.add_argument("--temp", type=float, default=50.0,
                   help="cp softmax temperature applied to raw-cp labels (lower = sharper)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    ds = SFLabelDataset(args.labels, temp=args.temp)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)
    print(f"Labels: {len(ds)}  (train {n_train} / val {n_val})")

    student = load_model(args.base, device)
    teacher = load_model(args.base, device)
    teacher.eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)

    if args.freeze_backbone:
        for name, pr in student.named_parameters():
            pr.requires_grad_(name.startswith("move_head"))
        trainable = [pr for pr in student.parameters() if pr.requires_grad]
        print(f"Frozen backbone: training {sum(p.numel() for p in trainable)} params (move_head only)")
    else:
        trainable = list(student.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def run_batch(batch, train: bool):
        tokens, player, castling, ep, target = batch
        tokens = tokens.to(device); player = player.to(device)
        castling = castling.to(device); ep = ep.to(device); target = target.to(device)
        move_logits, value = student(tokens, player, castling, ep)
        log_pred = F.log_softmax(move_logits.view(move_logits.size(0), -1).float(), dim=-1)
        kl = F.kl_div(log_pred, target, reduction="batchmean")
        with torch.no_grad():
            _, v_teacher = teacher(tokens, player, castling, ep)
        vpin = F.mse_loss(value.float(), v_teacher.float())
        loss = kl + args.value_pin * vpin
        # metric: top-1 agreement with the SF argmax target
        with torch.no_grad():
            pred_top = log_pred.argmax(dim=-1)
            tgt_top = target.argmax(dim=-1)
            top1 = (pred_top == tgt_top).float().mean()
        return loss, kl.detach(), vpin.detach(), top1

    student.train()
    for epoch in range(1, args.epochs + 1):
        tot = 0.0; nb = 0
        for batch in train_dl:
            opt.zero_grad()
            loss, kl, vpin, _ = run_batch(batch, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        # validation
        student.eval()
        with torch.no_grad():
            vkl = vt1 = vvp = 0.0; vb = 0
            for batch in val_dl:
                _, kl, vpin, top1 = run_batch(batch, False)
                vkl += kl.item(); vt1 += top1.item(); vvp += vpin.item(); vb += 1
        student.train()
        print(f"epoch {epoch}: train_loss={tot/nb:.4f}  val_kl={vkl/vb:.4f}  "
              f"val_top1(vs SF)={vt1/vb:.1%}  val_vpin={vvp/vb:.5f}")

    # Save
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student.eval()
    save_file(student.state_dict(), str(out / "model.safetensors"))
    shutil.copy(Path(args.base) / "model_config.json", out / "model_config.json")
    print(f"Saved distilled model to {out}")


if __name__ == "__main__":
    main()
