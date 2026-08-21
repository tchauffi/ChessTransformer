"""Expert-iteration training: trunk + both heads on self-play targets.

This is Step 2 of doc/selfplay_rl.md. Unlike the (now-removed) frozen-trunk
value-head trainer, this unfreezes the whole network and trains on the targets
produced by `scripts/selfplay_rust.py`:

  * policy: cross-entropy against the **MCTS visit distribution** (CSR
    visit_idx/visit_cnt/visit_ptr, indices = from_square * 73 + action_plane),
  * value : MSE against the game outcome z (stm-POV), with the same halfmove
    ramp as pretraining and an optional KataGo-style soft mix toward root_v,
  * a **KL anchor** to a frozen copy of the base net, to keep the policy from
    collapsing away from its 2100-Elo prior (cf. grpo_puzzles.py).

Validation splits by game id (consecutive positions of a game are near
duplicates and would leak). The frozen base net's val loss is reported as the
baseline before training. Output is a standard model dir loadable by every bot.

Usage
-----
    uv run python scripts/train_expert_iter.py \
        --data data/selfplay/v2.1-400sims-exit \
        --base data/models/pos2move_v2.1 \
        --out  data/models/pos2move_v2.1-exit1

Gate the result before promoting it:

    uv run python scripts/engine_match.py --a-mcts --b-mcts --a-sims 400 --b-sims 400 \
        --a-model-dir data/models/pos2move_v2.1-exit1 \
        --b-model-dir data/models/pos2move_v2.1
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo_puzzles import load_model  # shared model loader
from chesstransformer.models.transformer.pos2move_v2 import NUM_ACTION_PLANES

ACTION_SPACE = 64 * NUM_ACTION_PLANES  # 4672


# ── Dataset ────────────────────────────────────────────────────────────────
class SelfplayDataset(Dataset):
    """All shards with visit distributions, concatenated in RAM.

    Returns raw per-position fields; the dense policy target is built in the
    collate fn so we only ever materialise (B, 4672) for the current batch.
    """

    def __init__(self, data_dir: Path):
        shards = sorted(data_dir.glob("positions_*.npz"))
        if not shards:
            raise FileNotFoundError(f"no positions_*.npz in {data_dir}")

        boards, player, castling, ep, halfmove, z, root_v, game_id = ([] for _ in range(8))
        v_idx, v_cnt, v_len = [], [], []
        skipped = 0
        for s in shards:
            d = np.load(s)
            if "visit_idx" not in d:
                skipped += 1
                continue
            boards.append(d["boards"])
            player.append(d["player"]); castling.append(d["castling"]); ep.append(d["ep"])
            halfmove.append(d["halfmove"]); z.append(d["z"]); root_v.append(d["root_v"])
            game_id.append(d["game_id"])
            v_idx.append(d["visit_idx"].astype(np.int64))
            v_cnt.append(d["visit_cnt"].astype(np.float32))
            v_len.append(np.diff(d["visit_ptr"]))  # per-position support length

        if not boards:
            raise RuntimeError(f"{len(shards)} shard(s) found but none carry visit_idx "
                               "(regenerate with scripts/selfplay_rust.py)")
        if skipped:
            print(f"  skipped {skipped} shard(s) without visit distributions")

        self.boards = np.concatenate(boards).astype(np.int64)        # (N, 64)
        self.player = np.concatenate(player).astype(np.int64)
        self.castling = np.concatenate(castling).astype(np.int64)
        self.ep = np.concatenate(ep).astype(np.int64)
        self.halfmove = np.concatenate(halfmove).astype(np.float32)
        self.z = np.concatenate(z).astype(np.float32)
        self.root_v = np.concatenate(root_v).astype(np.float32)
        self.game_id = np.concatenate(game_id).astype(np.int64)

        self.v_idx = np.concatenate(v_idx)
        self.v_cnt = np.concatenate(v_cnt)
        lens = np.concatenate(v_len).astype(np.int64)
        self.v_ptr = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=self.v_ptr[1:])

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        a, b = self.v_ptr[i], self.v_ptr[i + 1]
        return {
            "board": self.boards[i], "player": self.player[i],
            "castling": self.castling[i], "ep": self.ep[i],
            "halfmove": self.halfmove[i], "z": self.z[i], "root_v": self.root_v[i],
            "v_idx": self.v_idx[a:b], "v_cnt": self.v_cnt[a:b],
        }


def collate(batch):
    B = len(batch)
    board = torch.from_numpy(np.stack([b["board"] for b in batch]))
    player = torch.tensor([b["player"] for b in batch], dtype=torch.long)
    castling = torch.tensor([b["castling"] for b in batch], dtype=torch.long)
    ep = torch.tensor([b["ep"] for b in batch], dtype=torch.long)
    halfmove = torch.tensor([b["halfmove"] for b in batch], dtype=torch.float32)
    z = torch.tensor([b["z"] for b in batch], dtype=torch.float32)
    root_v = torch.tensor([b["root_v"] for b in batch], dtype=torch.float32)

    # Dense, normalised visit distribution over the full action space.
    policy = torch.zeros(B, ACTION_SPACE, dtype=torch.float32)
    for i, b in enumerate(batch):
        cnt = torch.from_numpy(b["v_cnt"])
        total = cnt.sum()
        if total > 0:
            policy[i, torch.from_numpy(b["v_idx"])] = cnt / total
    return board, player, castling, ep, halfmove, z, root_v, policy


# ── Loss ──────────────────────────────────────────────────────────────────
def compute_loss(logits, value, policy_tgt, z, root_v, halfmove, ref_logits,
                 value_weight, kl_coef, soft_mix, ramp_halfmoves):
    B = logits.size(0)
    flat = logits.view(B, -1).float()
    logp = F.log_softmax(flat, dim=-1)

    # Policy: cross-entropy with the soft visit-distribution target.
    policy_loss = -(policy_tgt * logp).sum(dim=-1).mean()

    # KL anchor to the frozen base policy: KL(pi || ref).
    if kl_coef > 0:
        ref_logp = F.log_softmax(ref_logits.view(B, -1).float(), dim=-1)
        kl = (logp.exp() * (logp - ref_logp)).sum(dim=-1).mean()
    else:
        kl = torch.zeros((), device=logits.device)

    # Value: ramped MSE against outcome z, optionally mixed toward root_v.
    target_v = (1.0 - soft_mix) * z + soft_mix * root_v
    progress = (halfmove / ramp_halfmoves).clamp(max=1.0)
    value_loss = (progress * (value.float().squeeze(-1) - target_v) ** 2).mean()

    total = policy_loss + value_weight * value_loss + kl_coef * kl

    with torch.no_grad():
        decisive = z != 0
        sign_acc = ((value.float().squeeze(-1).sign() == z.sign()) & decisive).sum() / decisive.sum().clamp(min=1)
    metrics = {"policy": policy_loss.detach(), "value": value_loss.detach(),
               "kl": kl.detach(), "sign_acc": sign_acc}
    return total, metrics


# ── EMA (fp32, mirrors pos2move_v2_trainer) ─────────────────────────────────
def create_ema(model):
    return {n: p.data.detach().float().clone() for n, p in model.named_parameters()}


@torch.no_grad()
def update_ema(model, ema, decay):
    for n, p in model.named_parameters():
        ema[n].lerp_(p.data.float(), 1.0 - decay)


def swap_ema(model, ema):
    for n, p in model.named_parameters():
        tmp = p.data.detach().float().clone()
        p.data.copy_(ema[n].to(p.dtype))
        ema[n].copy_(tmp)


# ── Eval ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, args, ref_logits_fn=None):
    model.eval()
    tot = {"policy": 0.0, "value": 0.0, "sign_acc": 0.0}
    n = 0
    for board, player, castling, ep, halfmove, z, root_v, policy in loader:
        board, player, castling, ep = board.to(device), player.to(device), castling.to(device), ep.to(device)
        halfmove, z, root_v, policy = halfmove.to(device), z.to(device), root_v.to(device), policy.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, value = model(board, player, castling, ep)
        _, m = compute_loss(logits, value, policy, z, root_v, halfmove,
                            logits, args.value_weight, 0.0, args.soft_mix, args.ramp_halfmoves)
        b = board.size(0)
        tot["policy"] += m["policy"].item() * b
        tot["value"] += m["value"].item() * b
        tot["sign_acc"] += m["sign_acc"].item() * b
        n += b
    return {k: v / n for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dir of positions_*.npz with visit distributions")
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-embedding", type=float, default=4e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--kl-coef", type=float, default=0.1, help="KL anchor strength to frozen base policy")
    p.add_argument("--soft-mix", type=float, default=0.0, help="value target = (1-x)*z + x*root_v")
    p.add_argument("--ramp-halfmoves", type=float, default=40.0)
    p.add_argument("--val-frac", type=float, default=0.05, help="fraction of games held out")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--patience", type=int, default=3, help="early-stop epochs without val improvement")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # ── Data + game-level split ──────────────────────────────────────────
    ds = SelfplayDataset(Path(args.data))
    games = np.unique(ds.game_id)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(games)
    n_val = max(1, int(args.val_frac * len(games)))
    val_games = set(games[:n_val].tolist())
    is_val = np.fromiter((g in val_games for g in ds.game_id), dtype=bool, count=len(ds))
    train_idx = np.nonzero(~is_val)[0]
    val_idx = np.nonzero(is_val)[0]
    print(f"Positions: {len(ds):,} over {len(games):,} games | "
          f"train {len(train_idx):,} / val {len(val_idx):,} ({n_val} games)")

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    # ── Model + frozen reference ─────────────────────────────────────────
    model = load_model(Path(args.base), device)
    ref = copy.deepcopy(model).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    print(f"Model parameters: {sum(t.numel() for t in model.parameters()):,}")

    # ── Baseline: frozen base net on the val split ───────────────────────
    base_val = evaluate(model, val_loader, device, args)
    print(f"Baseline (frozen base): val policy_ce={base_val['policy']:.4f} | "
          f"value_mse={base_val['value']:.4f} | sign_acc={base_val['sign_acc']:.4f}")

    # ── Optimizer + warmup→linear-decay schedule ─────────────────────────
    emb, decay_p, nodecay_p = [], [], []
    for n_, t in model.named_parameters():
        if "embedding" in n_:
            emb.append(t)
        elif t.ndim >= 2:
            decay_p.append(t)
        else:
            nodecay_p.append(t)
    opt = torch.optim.AdamW([
        {"params": emb, "lr": args.lr_embedding, "weight_decay": args.weight_decay},
        {"params": decay_p, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": nodecay_p, "lr": args.lr, "weight_decay": 0.0},
    ])
    total_steps = max(1, len(train_loader) * args.epochs)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.05, 1.0 - prog)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ema = create_ema(model)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = {"policy": 0.0, "value": 0.0, "kl": 0.0}
        seen = 0
        for board, player, castling, ep, halfmove, z, root_v, policy in train_loader:
            board, player, castling, ep = board.to(device), player.to(device), castling.to(device), ep.to(device)
            halfmove, z, root_v, policy = halfmove.to(device), z.to(device), root_v.to(device), policy.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, value = model(board, player, castling, ep)
                with torch.no_grad():
                    ref_logits, _ = ref(board, player, castling, ep)
                loss, m = compute_loss(logits, value, policy, z, root_v, halfmove, ref_logits,
                                       args.value_weight, args.kl_coef, args.soft_mix, args.ramp_halfmoves)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            update_ema(model, ema, args.ema_decay)
            b = board.size(0)
            run["policy"] += m["policy"].item() * b
            run["value"] += m["value"].item() * b
            run["kl"] += m["kl"].item() * b
            seen += b
            step += 1
        tr = {k: v / seen for k, v in run.items()}

        # Validate with EMA weights.
        swap_ema(model, ema)
        val = evaluate(model, val_loader, device, args)
        val_loss = val["policy"] + args.value_weight * val["value"]
        print(f"Epoch {epoch}/{args.epochs} ({time.time()-t0:.0f}s) | "
              f"train policy={tr['policy']:.4f} value={tr['value']:.4f} kl={tr['kl']:.4f} | "
              f"val policy={val['policy']:.4f} value={val['value']:.4f} sign_acc={val['sign_acc']:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print(f"  -> new best (val_loss={val_loss:.4f})")
        else:
            bad_epochs += 1
        swap_ema(model, ema)  # restore live weights for next epoch
        if bad_epochs >= args.patience:
            print(f"  early stop: no improvement for {args.patience} epochs")
            break

    if best_state is None:  # never improved; export EMA of current
        swap_ema(model, ema)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ── Export a standard model dir ──────────────────────────────────────
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(best_state, str(out / "model.safetensors"))
    shutil.copy(Path(args.base) / "model_config.json", out / "model_config.json")
    meta = {"base": str(args.base), "data": str(args.data), "epochs": epoch,
            "kl_coef": args.kl_coef, "value_weight": args.value_weight, "soft_mix": args.soft_mix,
            "baseline_val": base_val, "best_val_loss": best_val}
    (out / "expert_iter_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {out}. Gate vs base with scripts/engine_match.py before promoting.")


if __name__ == "__main__":
    main()
