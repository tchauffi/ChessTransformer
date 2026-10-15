"""Expert-iteration training: trunk + both heads on self-play targets.

This is Step 2 of doc/selfplay_rl.md. Unlike the (now-removed) frozen-trunk
value-head trainer, this unfreezes the whole network and trains on the targets
produced by `scripts/selfplay_rust.py`:

  * policy: cross-entropy against the **MCTS visit distribution** (CSR
    visit_idx/visit_cnt/visit_ptr, indices = from_square * 73 + action_plane),
    on full searches only when the data was generated with playout cap
    randomization (the `full` column; fast searches are too shallow to teach);
    or, with `--policy-target gumbel`, against Gumbel MuZero's completed-Q
    improved policy over every legal move (`completed_q_targets`), which
    `--gumbel-fast` also takes from the fast searches,
  * value : the game outcome z (stm-POV) on every position, optionally mixed
    KataGo-style toward the search's root_v -- MSE for a scalar head,
    cross-entropy for a WDL head,
  * a **KL anchor** to a frozen copy of the base net, to keep the policy from
    collapsing away from its prior (cf. grpo_puzzles.py),
  * optionally, **human replay** (`--human-shards`): a fraction of every step
    is ordinary supervised loss on the human shards, so the net does not forget
    what the self-play data never shows it.

`--data` takes several directories: a replay window over the last generations.

Validation splits by game id (consecutive positions of a game are near
duplicates and would leak), on a **permanent held-out set**: the first training
that reads a directory draws `--val-frac` of its games and stores them in
`heldout_games.json` next to the shards, and every later training reads them
back. A replay window holds the champion's own training data, so a fresh random
split each generation scored the base net on games it had trained on (92% of
gen03's half of cand04's val split): a flattered baseline. With the stored set,
neither the base net nor the candidate has trained on any val game. The frozen
base net's val loss is reported as the baseline before training, per directory
and overall. Output is a standard model dir loadable by every bot.

**Checkpoint selection.** exit2 had the best held-out loss of any run and the
worst gate (doc/selfplay_rl.md): val loss anti-correlates with playing
strength. So the default `--select last` exports the final EMA weights and
leaves the verdict to a match; `--select val` restores the old behaviour.

Usage
-----
    uv run python scripts/train_expert_iter.py \
        --data data/selfplay/v2.1-400sims-exit \
        --base data/models/pos2move_v2.1 \
        --out  data/models/pos2move_v2.1-exit1

Several generations plus human replay (what scripts/rl_generation.sh runs):

    uv run python scripts/train_expert_iter.py \
        --data data/rl/selfplay/gen01 data/rl/selfplay/gen02 \
        --base data/rl/models/gen02 --out data/rl/models/cand03 \
        --epochs 1 --soft-mix 0.5 --human-shards data/shards/elite_k32_v2 --human-frac 0.2

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
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo_puzzles import load_model  # shared model loader
from chesstransformer.datasets.flat_shard_dataset import FlatShardDataset, make_dataloader
from chesstransformer.models.transformer.pos2move_v2 import NUM_ACTION_PLANES

ACTION_SPACE = 64 * NUM_ACTION_PLANES  # 4672
# Game ids restart at 0 in every self-play directory; offset them per directory
# so the game-level val split stays a split across a replay window.
GAME_ID_STRIDE = 1 << 32
HELDOUT_FILE = "heldout_games.json"


def heldout_games(data_dir: Path, game_ids: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """A directory's permanent held-out game ids: read from HELDOUT_FILE, or drawn and stored.

    The loop first trains on a generation right after its self-play, before any network
    has seen its games, so a set drawn then stays unseen by every later candidate too.
    """
    path = Path(data_dir) / HELDOUT_FILE
    if path.exists():
        return np.asarray(json.loads(path.read_text())["games"], dtype=np.int64)
    games = np.unique(game_ids)
    pick = np.sort(np.random.default_rng(seed).choice(games, max(1, int(frac * len(games))), replace=False))
    path.write_text(json.dumps({"frac": frac, "seed": seed, "games": pick.tolist()}))
    print(f"  drew {len(pick)} held-out games for {data_dir} -> {path}")
    return pick


def heldout_split(game_id: np.ndarray, data_dirs: list[Path], frac: float, seed: int) -> dict[str, np.ndarray]:
    """Dataset indices of each directory's held-out games, keyed by directory name."""
    src = game_id // GAME_ID_STRIDE
    names = [Path(d).name for d in data_dirs]
    if len(set(names)) < len(names):
        names = [str(d) for d in data_dirs]
    parts = {}
    for di, (d, name) in enumerate(zip(data_dirs, names)):
        idx = np.nonzero(src == di)[0]
        local = game_id[idx] - di * GAME_ID_STRIDE
        parts[name] = idx[np.isin(local, heldout_games(Path(d), local, frac, seed))]
    return parts


def completed_q_targets(prior, n, q, ptr, net_v, fallback_v, c_visit=50.0, c_scale=0.1,
                        rescale=False):
    """Gumbel MuZero's improved policy (Danihelka et al., "Policy improvement by planning
    with Gumbel", ICLR 2022) for each position's legal moves, CSR on `ptr`:

        softmax(log prior + (c_visit + max visits) * c_scale * q_hat)

    q_hat is the completed Q: a visited move's Q, an unvisited one the mixed value
    v_mix = (v + N * sum_visited(prior * Q) / sum_visited(prior)) / (1 + N), with v the
    network's own value (`fallback_v` where it is NaN) and N the root's visits -- mapped to
    [0, 1] by (Q + 1) / 2, since chess values are bounded, or min-max per position with
    `rescale` (mctx's default, which turns any Q gap into the full scale). Unvisited moves
    keep their prior ratios; a visited move gains with its Q, whatever its visit count.
    Values are from the side to move's POV, as recorded.
    """
    lens = np.diff(ptr)
    if (lens <= 0).any():
        raise ValueError("a position without legal moves")
    P, starts = len(lens), ptr[:-1]
    seg = np.repeat(np.arange(P), lens)
    prior, n, q = (np.asarray(a, dtype=np.float64) for a in (prior, n, q))
    vis = n > 0
    N = np.bincount(seg, weights=n, minlength=P)
    sum_p = np.bincount(seg, weights=prior * vis, minlength=P)
    sum_pq = np.bincount(seg, weights=prior * q * vis, minlength=P)
    v = np.where(np.isfinite(net_v), net_v, fallback_v).astype(np.float64)
    v_mix = (v + N * np.divide(sum_pq, sum_p, out=np.zeros(P), where=sum_p > 0)) / (1 + N)
    cq = np.where(vis, q, v_mix[seg])
    if rescale:
        lo, hi = np.minimum.reduceat(cq, starts), np.maximum.reduceat(cq, starts)
        q_hat = (cq - lo[seg]) / np.maximum(hi - lo, 1e-8)[seg]
    else:
        q_hat = (cq + 1.0) / 2.0
    logit = np.log(np.maximum(prior, 1e-12)) + (c_visit + np.maximum.reduceat(n, starts)[seg]) * c_scale * q_hat
    logit -= np.maximum.reduceat(logit, starts)[seg]
    e = np.exp(logit)
    return (e / np.bincount(seg, weights=e, minlength=P)[seg]).astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────
class SelfplayDataset(Dataset):
    """All shards with visit distributions, concatenated in RAM.

    Kept in the on-disk dtypes (a few million positions as int64 boards would
    be gigabytes) and widened per batch in the collate fn, which also builds the
    dense policy target so only (B, 4672) is ever materialised.
    """

    def __init__(self, data_dirs: list[Path], policy_target: str = "visits", gumbel: dict | None = None,
                 gumbel_fast: bool = False):
        boards, player, castling, ep, halfmove, z, root_v, game_id, full = ([] for _ in range(9))
        v_idx, v_cnt, v_len = [], [], []
        skipped = n_shards = 0
        for di, data_dir in enumerate(data_dirs):
            shards = sorted(Path(data_dir).glob("positions_*.npz"))
            if not shards:
                raise FileNotFoundError(f"no positions_*.npz in {data_dir}")
            n_shards += len(shards)
            for s in shards:
                d = np.load(s)
                if "visit_idx" not in d:
                    skipped += 1
                    continue
                boards.append(d["boards"])
                player.append(d["player"]); castling.append(d["castling"]); ep.append(d["ep"])
                halfmove.append(d["halfmove"]); z.append(d["z"]); root_v.append(d["root_v"])
                game_id.append(d["game_id"].astype(np.int64) + di * GAME_ID_STRIDE)
                # Data from before playout cap randomization: every search was full.
                full.append(d["full"] if "full" in d else np.ones(len(d["z"]), dtype=np.uint8))
                if policy_target == "gumbel":
                    if "legal_ptr" not in d:
                        raise ValueError(f"{s}: no root search columns (legal_*), which "
                                         "--policy-target gumbel needs")
                    v_idx.append(d["legal_idx"])
                    v_cnt.append(completed_q_targets(d["legal_prior"], d["legal_n"], d["legal_q"],
                                                     d["legal_ptr"], d["root_net_v"], d["root_v"],
                                                     **(gumbel or {})))
                    v_len.append(np.diff(d["legal_ptr"]))
                else:
                    v_idx.append(d["visit_idx"])
                    v_cnt.append(d["visit_cnt"])
                    v_len.append(np.diff(d["visit_ptr"]))  # per-position support length

        if not boards:
            raise RuntimeError(f"{n_shards} shard(s) found but none carry visit_idx "
                               "(regenerate with scripts/selfplay_rust.py)")
        if skipped:
            print(f"  skipped {skipped} shard(s) without visit distributions")

        self.boards = np.concatenate(boards)                          # (N, 64) uint8
        self.player = np.concatenate(player)
        self.castling = np.concatenate(castling)
        self.ep = np.concatenate(ep)
        self.halfmove = np.concatenate(halfmove).astype(np.float32)
        self.z = np.concatenate(z).astype(np.float32)
        self.root_v = np.concatenate(root_v).astype(np.float32)
        self.game_id = np.concatenate(game_id)
        self.full = np.concatenate(full).astype(np.float32)
        if policy_target == "gumbel" and gumbel_fast:
            self.full[:] = 1.0  # the policy mask: every search teaches

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
            "full": self.full[i], "v_idx": self.v_idx[a:b], "v_cnt": self.v_cnt[a:b],
        }


def collate(batch):
    B = len(batch)
    board = torch.from_numpy(np.stack([b["board"] for b in batch])).long()
    player = torch.tensor([b["player"] for b in batch], dtype=torch.long)
    castling = torch.tensor([b["castling"] for b in batch], dtype=torch.long)
    ep = torch.tensor([b["ep"] for b in batch], dtype=torch.long)
    halfmove = torch.tensor([b["halfmove"] for b in batch], dtype=torch.float32)
    z = torch.tensor([b["z"] for b in batch], dtype=torch.float32)
    root_v = torch.tensor([b["root_v"] for b in batch], dtype=torch.float32)
    full = torch.tensor([b["full"] for b in batch], dtype=torch.float32)

    # Dense, normalised visit distribution over the full action space.
    policy = torch.zeros(B, ACTION_SPACE, dtype=torch.float32)
    for i, b in enumerate(batch):
        cnt = torch.from_numpy(b["v_cnt"].astype(np.float32))
        total = cnt.sum()
        if total > 0:
            policy[i, torch.from_numpy(b["v_idx"].astype(np.int64))] = cnt / total
    return board, player, castling, ep, halfmove, z, root_v, full, policy


# ── Loss ──────────────────────────────────────────────────────────────────
def value_loss_per_sample(value, target):
    """(per-sample loss, scalar Q) for a scalar target in [-1, 1], stm-POV.

    A WDL head ([win, draw, loss] logits) is trained by cross-entropy against
    the distribution with expectation `target` and the most draw mass,
    (max(t, 0), 1 - |t|, max(-t, 0)) -- the one-hot outcome when t = z.
    """
    if value.size(-1) == 3:
        tgt = torch.stack([target.clamp(min=0.0), 1.0 - target.abs(), (-target).clamp(min=0.0)], dim=-1)
        logp = F.log_softmax(value.float(), dim=-1)
        p = logp.exp()
        return -(tgt * logp).sum(dim=-1), p[:, 0] - p[:, 2]
    q = value.float().squeeze(-1)
    return (q - target) ** 2, q


def compute_loss(logits, value, policy_tgt, full, z, root_v, halfmove, ref_logits,
                 value_weight, kl_coef, soft_mix, ramp_halfmoves):
    B = logits.size(0)
    flat = logits.view(B, -1).float()
    logp = F.log_softmax(flat, dim=-1)

    # Policy: cross-entropy with the soft visit-distribution target, on full
    # searches only.
    per_pos = -(policy_tgt * logp).sum(dim=-1)
    policy_loss = (per_pos * full).sum() / full.sum().clamp(min=1.0)

    # KL anchor to the frozen base policy: KL(pi || ref).
    if kl_coef > 0:
        ref_logp = F.log_softmax(ref_logits.view(B, -1).float(), dim=-1)
        kl = (logp.exp() * (logp - ref_logp)).sum(dim=-1).mean()
    else:
        kl = torch.zeros((), device=logits.device)

    # Value: outcome z, optionally mixed toward root_v, ramped by halfmove.
    target_v = (1.0 - soft_mix) * z + soft_mix * root_v
    per_v, q = value_loss_per_sample(value, target_v)
    progress = (halfmove / ramp_halfmoves).clamp(max=1.0) if ramp_halfmoves > 0 else torch.ones_like(z)
    value_loss = (progress * per_v).mean()

    total = policy_loss + value_weight * value_loss + kl_coef * kl

    with torch.no_grad():
        decisive = z != 0
        sign_acc = ((q.sign() == z.sign()) & decisive).sum() / decisive.sum().clamp(min=1)
    metrics = {"policy": policy_loss.detach(), "value": value_loss.detach(),
               "kl": kl.detach(), "sign_acc": sign_acc}
    return total, metrics


def human_loss(model, hb, device, value_weight, elo_gap=True):
    """Supervised loss on one batch of human shards: the played move and the
    (tablebase-adjudicated, when the shards carry it) game result.

    With ``elo_gap`` the rating gap goes through the value head's Elo offset, as in
    pretraining (pos2move_v2_trainer.prepare_batch). Without it a human result is read
    as a game between equals, so the head is taught to explain the stronger player's
    wins through the position -- the bias the offset exists to remove."""
    board = hb["position"].to(device, non_blocking=True).long()
    is_white = hb["is_white"].to(device, non_blocking=True).bool()
    gap = None
    if elo_gap and getattr(model, "elo_beta", None) is not None and "elo_gap" in hb:
        gap = hb["elo_gap"].to(device, non_blocking=True).float()
        if "result_tb" in hb and "value_exact" in hb:  # a tablebase result is exact
            gap = gap.masked_fill(hb["value_exact"].to(device).bool(), 0.0)
    logits, value = model(board, is_white.long(),
                          hb["castling_rights"].to(device, non_blocking=True).long(),
                          hb["en_passant_file"].to(device, non_blocking=True).long(),
                          elo_gap=gap)
    B = board.size(0)
    target = (hb["from_square"].to(device).long() * NUM_ACTION_PLANES
              + hb["action_plane"].to(device).long())
    policy = F.cross_entropy(logits.view(B, -1).float(), target)
    result = hb["result_tb" if "result_tb" in hb else "result"].to(device).long()
    z = torch.zeros(B, dtype=torch.float32, device=device)  # 1 = white won, 2 = black won
    z[((result == 1) & is_white) | ((result == 2) & ~is_white)] = 1.0
    z[((result == 1) & ~is_white) | ((result == 2) & is_white)] = -1.0
    per_v, _ = value_loss_per_sample(value, z)
    return policy + value_weight * per_v.mean()


def cycle(loader):
    while True:
        yield from loader


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
    for board, player, castling, ep, halfmove, z, root_v, full, policy in loader:
        board, player, castling, ep = board.to(device), player.to(device), castling.to(device), ep.to(device)
        halfmove, z, root_v, policy = halfmove.to(device), z.to(device), root_v.to(device), policy.to(device)
        full = full.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, value = model(board, player, castling, ep)
        _, m = compute_loss(logits, value, policy, full, z, root_v, halfmove,
                            logits, args.value_weight, 0.0, args.soft_mix, args.ramp_halfmoves)
        b = board.size(0)
        tot["policy"] += m["policy"].item() * b
        tot["value"] += m["value"].item() * b
        tot["sign_acc"] += m["sign_acc"].item() * b
        n += b
    return {k: v / n for k, v in tot.items()}


def evaluate_parts(model, loaders, device, args):
    """evaluate() on each directory's held-out games, plus their position-weighted mean as "all"."""
    res = {name: evaluate(model, dl, device, args) for name, dl in loaders.items()}
    n = {name: len(dl.dataset) for name, dl in loaders.items()}
    res["all"] = {k: sum(res[name][k] * n[name] for name in n) / sum(n.values())
                  for k in ("policy", "value", "sign_acc")}
    return res


def format_parts(res):
    return " | ".join(f"{name} policy={r['policy']:.4f} value={r['value']:.4f}"
                      for name, r in res.items() if name != "all")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, nargs="+",
                   help="dir(s) of positions_*.npz with visit distributions (a replay window)")
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = epochs only)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-embedding", type=float, default=4e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--new-lr-mult", type=float, default=1.0,
                   help="lr multiplier for smolgen / full-table attention-bias parameters")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="torch.compile the training forwards: 2.2x the steps/s of eager on gen01 and "
                        "9.4 vs >15.4 GiB on the smolgen model, which eager cannot fit at batch 512")
    p.add_argument("--human-elo-gap", action=argparse.BooleanOptionalAction, default=True,
                   help="feed the human shards' rating gap to the value head's Elo offset")
    p.add_argument("--train-dropout", action=argparse.BooleanOptionalAction, default=True,
                   help="keep the base config's dropout and layer drop while training. With "
                        "them on, the KL to the (eval-mode) reference is mostly dropout noise: "
                        "0.050 for identical weights on gen01")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--kl-coef", type=float, default=0.1, help="KL anchor strength to frozen base policy")
    p.add_argument("--soft-mix", type=float, default=0.0, help="value target = (1-x)*z + x*root_v")
    p.add_argument("--ramp-halfmoves", type=float, default=40.0)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help=f"fraction of a directory's games held out when its {HELDOUT_FILE} is first drawn")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--patience", type=int, default=3,
                   help="early-stop epochs without val improvement (--select val only)")
    p.add_argument("--policy-target", choices=["visits", "gumbel"], default="visits",
                   help="policy target: the visit distribution, or Gumbel MuZero's completed-Q "
                        "improved policy (needs self-play with the root search columns)")
    p.add_argument("--gumbel-c-visit", type=float, default=50.0)
    p.add_argument("--gumbel-c-scale", type=float, default=0.1)
    p.add_argument("--gumbel-rescale", action="store_true",
                   help="min-max the completed Q per position (mctx) instead of (Q + 1) / 2")
    p.add_argument("--gumbel-fast", action="store_true",
                   help="with --policy-target gumbel, fast searches are policy targets too")
    p.add_argument("--select", choices=["last", "val"], default="last",
                   help="export the final EMA weights (default) or the best val-loss epoch")
    p.add_argument("--human-shards", default=None, help="flat shard dir for human replay")
    p.add_argument("--human-frac", type=float, default=0.2,
                   help="share of each step's loss (and samples) taken from --human-shards")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # ── Data + game-level split ──────────────────────────────────────────
    ds = SelfplayDataset([Path(d) for d in args.data], policy_target=args.policy_target,
                         gumbel=dict(c_visit=args.gumbel_c_visit, c_scale=args.gumbel_c_scale,
                                     rescale=args.gumbel_rescale),
                         gumbel_fast=args.gumbel_fast)
    games = np.unique(ds.game_id)
    val_parts = {name: idx for name, idx in heldout_split(ds.game_id, args.data, args.val_frac, args.seed).items()
                 if len(idx)}
    is_val = np.zeros(len(ds), dtype=bool)
    for idx in val_parts.values():
        is_val[idx] = True
    train_idx = np.nonzero(~is_val)[0]
    n_val = len(np.unique(ds.game_id[is_val]))
    print(f"Positions: {len(ds):,} over {len(games):,} games ({ds.full.mean():.0%} policy targets, {args.policy_target}) | "
          f"train {len(train_idx):,} / val {is_val.sum():,} ({n_val} held-out games: "
          + ", ".join(f"{name} {len(np.unique(ds.game_id[idx]))}" for name, idx in val_parts.items()) + ")")

    human_iter, human_bs = None, 0
    if args.human_shards and args.human_frac > 0:
        human_bs = max(1, round(args.batch_size * args.human_frac / (1.0 - args.human_frac)))
        human_ds = FlatShardDataset(args.human_shards, "train")
        human_iter = cycle(make_dataloader(human_ds, human_bs, shuffle=True, num_workers=2,
                                           drop_last=True, seed=args.seed))
        print(f"Human replay: {args.human_shards} ({len(human_ds):,} samples), "
              f"{human_bs}/step, loss share {args.human_frac:.0%}")

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loaders = {name: DataLoader(torch.utils.data.Subset(ds, idx), batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, collate_fn=collate)
                   for name, idx in val_parts.items()}

    # ── Model + frozen reference ─────────────────────────────────────────
    model = load_model(Path(args.base), device)
    if not args.train_dropout:
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
            if hasattr(mod, "layer_drop"):
                mod.layer_drop = 0.0
            if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
                mod.dropout = 0.0  # attention-probability dropout inside SDPA
    ref = copy.deepcopy(model).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    # Training forwards go through the compiled wrappers; `model` keeps the parameters, so
    # EMA, evaluation and the saved state dict are unchanged (no `_orig_mod.` prefixes).
    fwd = torch.compile(model) if args.compile else model
    ref_fwd = torch.compile(ref) if args.compile else ref
    print(f"Model parameters: {sum(t.numel() for t in model.parameters()):,}")

    # ── Baseline: frozen base net on the val split ───────────────────────
    base_parts = evaluate_parts(model, val_loaders, device, args)
    base_val = base_parts["all"]
    print(f"Baseline (frozen base): val policy_ce={base_val['policy']:.4f} | "
          f"value_mse={base_val['value']:.4f} | sign_acc={base_val['sign_acc']:.4f}")
    print(f"  held out: {format_parts(base_parts)}")

    # ── Optimizer + warmup→linear-decay schedule ─────────────────────────
    # Attention-bias parameters added by scripts/convert_attn_bias.py start from zero (smolgen)
    # or from the bucket values (bias_full), so --new-lr-mult lets them move within one
    # generation. The full table is kept out of weight decay: decay would pull it toward
    # zero, away from the bucket values it was filled with.
    emb, decay_p, nodecay_p, new_decay, new_nodecay = [], [], [], [], []
    for n_, t in model.named_parameters():
        if "smolgen" in n_ or "bias_full" in n_:
            (new_decay if t.ndim >= 2 and "bias_full" not in n_ else new_nodecay).append(t)
        elif "embedding" in n_:
            emb.append(t)
        elif t.ndim >= 2:
            decay_p.append(t)
        else:
            nodecay_p.append(t)
    groups = [
        {"params": emb, "lr": args.lr_embedding, "weight_decay": args.weight_decay},
        {"params": decay_p, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": nodecay_p, "lr": args.lr, "weight_decay": 0.0},
    ]
    if new_decay or new_nodecay:
        new_lr = args.lr * args.new_lr_mult
        groups += [{"params": new_decay, "lr": new_lr, "weight_decay": args.weight_decay},
                   {"params": new_nodecay, "lr": new_lr, "weight_decay": 0.0}]
        print(f"Attention-bias parameters: {sum(t.numel() for t in new_decay + new_nodecay):,} "
              f"at lr {new_lr:g}")
    opt = torch.optim.AdamW(groups)
    total_steps = max(1, len(train_loader) * args.epochs)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

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
        for board, player, castling, ep, halfmove, z, root_v, full, policy in train_loader:
            board, player, castling, ep = board.to(device), player.to(device), castling.to(device), ep.to(device)
            halfmove, z, root_v, policy = halfmove.to(device), z.to(device), root_v.to(device), policy.to(device)
            full = full.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, value = fwd(board, player, castling, ep)
                with torch.no_grad():
                    ref_logits, _ = ref_fwd(board, player, castling, ep)
                loss, m = compute_loss(logits, value, policy, full, z, root_v, halfmove, ref_logits,
                                       args.value_weight, args.kl_coef, args.soft_mix, args.ramp_halfmoves)
                if human_iter is not None:
                    h = human_loss(fwd, next(human_iter), device, args.value_weight,
                                   elo_gap=args.human_elo_gap)
                    loss = (1.0 - args.human_frac) * loss + args.human_frac * h
                    m["human"] = h.detach()
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
            if "human" in m:
                run["human"] = run.get("human", 0.0) + m["human"].item() * b
            seen += b
            step += 1
            if step % 500 == 0:
                print(f"  step {step}/{total_steps} | policy={run['policy'] / seen:.4f} "
                      f"value={run['value'] / seen:.4f} kl={run['kl'] / seen:.4f}", flush=True)
            if step >= total_steps:
                break
        tr = {k: v / seen for k, v in run.items()}

        # Validate with EMA weights.
        swap_ema(model, ema)
        val_parts_res = evaluate_parts(model, val_loaders, device, args)
        val = val_parts_res["all"]
        val_loss = val["policy"] + args.value_weight * val["value"]
        print(f"Epoch {epoch}/{args.epochs} ({time.time()-t0:.0f}s) | "
              f"train policy={tr['policy']:.4f} value={tr['value']:.4f} kl={tr['kl']:.4f}"
              + (f" human={tr['human']:.4f}" if "human" in tr else "") + " | "
              f"val policy={val['policy']:.4f} value={val['value']:.4f} sign_acc={val['sign_acc']:.4f}")
        print(f"  held out: {format_parts(val_parts_res)}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            if args.select == "val":
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print(f"  -> new best (val_loss={val_loss:.4f})")
        else:
            bad_epochs += 1
        swap_ema(model, ema)  # restore live weights for next epoch
        if step >= total_steps:
            break
        if args.select == "val" and bad_epochs >= args.patience:
            print(f"  early stop: no improvement for {args.patience} epochs")
            break

    if best_state is None:  # --select last, or never improved: export the final EMA
        swap_ema(model, ema)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print("Held-out change, last epoch - base (below 0 = better): " + " | ".join(
        f"{name} policy {val_parts_res[name]['policy'] - r['policy']:+.4f} "
        f"value {val_parts_res[name]['value'] - r['value']:+.4f}" for name, r in base_parts.items()))

    # ── Export a standard model dir ──────────────────────────────────────
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(best_state, str(out / "model.safetensors"))
    shutil.copy(Path(args.base) / "model_config.json", out / "model_config.json")
    meta = {"base": str(args.base), "data": [str(d) for d in args.data], "epochs": epoch,
            "steps": step, "select": args.select,
            "kl_coef": args.kl_coef, "value_weight": args.value_weight, "soft_mix": args.soft_mix,
            "new_lr_mult": args.new_lr_mult, "human_elo_gap": args.human_elo_gap,
            "train_dropout": args.train_dropout, "compile": args.compile,
            "human_shards": args.human_shards, "human_frac": args.human_frac if args.human_shards else 0.0,
            "baseline_val": base_val, "last_val": val, "best_val_loss": best_val,
            "heldout_games": {name: int(len(np.unique(ds.game_id[idx]))) for name, idx in val_parts.items()},
            "baseline_val_by_dir": {k: v for k, v in base_parts.items() if k != "all"},
            "last_val_by_dir": {k: v for k, v in val_parts_res.items() if k != "all"}}
    (out / "expert_iter_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {out}. Gate vs base with scripts/engine_match.py before promoting.")


if __name__ == "__main__":
    main()
