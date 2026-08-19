#!/usr/bin/env python3
"""GRPO on self-play positions with a dense Stockfish centipawn reward.

The fourth thing tried against the human-data ceiling, and deliberately not a
repeat of the first three (see ``doc/selfplay_rl.md``):

* **Not policy distillation.** ``distill_policy.py`` minimised cross-entropy
  against Stockfish's distribution over all 4672 actions and moved nothing. This
  never asks the network to reproduce a distribution -- only to shift weight
  between moves it already considers, which is a far smaller ask of an 11.7M
  policy head that is plausibly at capacity for the harder one.
* **Not GRPO on puzzles.** That used a binary "did you find the tactic" reward on
  Lichess puzzles: sparse, and drawn from a distribution the engine never plays
  from. Here the reward is a *dense* cp difference and every position comes from
  the engine's own self-play games.
* **Not expert iteration.** No MCTS visit counts are involved, so the failure
  mode diagnosed there -- the policy being pulled toward a weaker search's
  sharpened copy of itself -- cannot arise.

The objective
-------------
For each position the reward table gives a candidate set and Stockfish's cp for
each candidate. cp becomes a value in [-1, 1] through the same Lichess win curve
``prep_eval_value_data.py`` uses, then advantages are group-normalised within
the position -- the GRPO part, and what makes the update scale-free across quiet
and sharp positions alike.

The policy gradient is taken in **closed form** over the candidate set rather
than by sampling K moves:

    loss_pg = - sum_c  pi(c).detach() * A_c * log pi(c)

Same expectation as sampling, zero sampling variance, and every candidate
contributes on every step -- which matters because the interesting candidates
are exactly the ones the policy currently assigns little mass to and would
rarely draw. ``--sample-k`` switches to the sampled estimator if you want the
literal grpo_puzzles behaviour.

``log pi`` is a softmax over the **full legal move set**, not over the candidate
set. Renormalising over candidates would optimise a distribution the engine
never plays; taking mass from a good candidate has to come out of the real
alternatives.

Entropy is a hard constraint here, not a nicety
-----------------------------------------------
The policy is not the final answer in this system -- it is the **prior that
feeds PUCT**. A near-deterministic prior is a broken engine no matter how good
its top move is, because search stops exploring anything else.

This is not hypothetical. The objective's optimum *is* a deterministic policy
(put all mass on the best candidate), so the KL anchor is the only thing holding
entropy up. Carrying over ``beta_kl=0.02`` from the puzzle GRPO, a first run on
148k positions reached KL 0.74 and **collapsed validation entropy from 1.714 to
0.199 within 250 steps** while its cp numbers still looked like an improvement
(+34.6 expected cp). Reward went up; the thing we actually ship went down.

So the coefficient is not a constant to guess: ``--target-kl`` sets a drift
budget and beta is steered to hold it, PPO-style. Small KL also bounds the
entropy change, so this protects the prior directly. Every run prints entropy
against the base's next to a collapse warning.

**But the anchor cannot hold a budget on its own under Adam.** Adam normalises
per parameter, so scaling the loss does not scale the step: a larger beta
changes the update's *direction*, not its magnitude, and the policy keeps
moving at roughly ``lr`` per step whatever the coefficient. Measured here, KL
climbed 0.017 -> 0.028 -> 0.060 -> 0.080 over 1000 steps while beta was driven
to ~200 and pinned there. Treat the controller as shaping *where* the policy
goes, and **step count and learning rate as what bounds how far**.

The practical consequence is that a run is not one candidate but a ladder of
them at increasing drift, which is why checkpoints are saved periodically and
gated. Gains saturate long before entropy does -- +5.2, +9.6, +13.8, +15.9
expected cp at steps 250/500/750/1000, so the increments halve while entropy
falls monotonically. The best trade lives early on that curve, and only the
match harness can say where.

The value head is frozen throughout, so this experiment isolates the policy --
the one variable being tested. Value is a separate stage.

One subtlety worth knowing: training runs with the model's configured
regularisation (``dropout=0.05``, ``layer_drop=0.1``), matching
``grpo_puzzles.py``, so the ``pi(c)`` weights in the closed-form gradient come
from a stochastic forward pass and are a noisy estimate of the true policy. The
update stays right in expectation and the regularisation is worth keeping --
expert iteration overfit this dataset size badly -- but it is why validation
here always runs under ``eval()``, and why a gradient-direction check measured
in train mode reads as pure noise.

What to watch
-------------
``exp_cp`` is the metric that matters: the expected centipawn value of the
policy's own move distribution. It answers "would this engine pick better moves"
directly, unlike CE or val loss -- which ``doc/selfplay_rl.md`` records as
*anti*-correlated with playing strength on this project. Checkpoint selection
still belongs to the match harness; this script only reports.

Usage
-----
    uv run python scripts/grpo_selfplay.py \
        --rewards data/rewards/exit128-sf10.npz \
        --base data/models/pos2move_v2.1 --out data/models/v2.1-grpo-sf \
        --steps 3000 --beta-kl 0.02
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

# The Lichess win curve, identical to scripts/prep_eval_value_data.py so cp->value
# means the same thing everywhere in this repo.
CP_K = 0.00368208


def cp_to_value(cp: np.ndarray | torch.Tensor):
    """Centipawns -> [-1, 1], mover's point of view."""
    if isinstance(cp, torch.Tensor):
        return 2.0 / (1.0 + torch.exp(-CP_K * cp)) - 1.0
    return 2.0 / (1.0 + np.exp(-CP_K * cp)) - 1.0


class RewardTable:
    """The npz written by scripts/gen_sf_move_rewards.py, split by game."""

    def __init__(self, path: Path):
        d = np.load(path)
        self.boards = torch.from_numpy(d["boards"].astype(np.int64))
        self.player = torch.from_numpy(d["player"].astype(np.int64))
        self.castling = torch.from_numpy(d["castling"].astype(np.int64))
        self.ep = torch.from_numpy(d["ep"].astype(np.int64))
        self.cand_idx = torch.from_numpy(d["cand_idx"].astype(np.int64))
        self.cand_cp = torch.from_numpy(d["cand_cp"].astype(np.float32))
        self.cand_mask = torch.from_numpy(d["cand_mask"])
        self.legal_idx = torch.from_numpy(d["legal_idx"].astype(np.int64))
        self.legal_ptr = torch.from_numpy(d["legal_ptr"].astype(np.int64))
        self.game_id = d["game_id"]
        self.meta = d["meta"]
        self.n = len(self.boards)

    def split_by_game(self, val_frac: float, seed: int):
        """Validation split by game, never by position.

        Consecutive positions of one game are near-duplicates; splitting by
        position leaks and makes the held-out number meaningless.
        """
        games = np.unique(self.game_id)
        rng = np.random.default_rng(seed)
        rng.shuffle(games)
        n_val = max(1, int(len(games) * val_frac))
        val_games = set(games[:n_val].tolist())
        is_val = np.array([g in val_games for g in self.game_id])
        return np.where(~is_val)[0], np.where(is_val)[0]

    def batch(self, rows: np.ndarray, device: str):
        """Dense padded batch: inputs, candidate columns, legal-move mask."""
        r = torch.from_numpy(rows)
        b = self.boards[r].to(device)
        pl = self.player[r].to(device)
        ca = self.castling[r].to(device)
        ep = self.ep[r].to(device)
        ci = self.cand_idx[r].to(device)
        cm = self.cand_mask[r].to(device)
        cp = self.cand_cp[r].to(device)

        # Legal moves vary in count, so pad to the batch max and carry a mask.
        starts = self.legal_ptr[r]
        ends = self.legal_ptr[r + 1]
        widths = (ends - starts).tolist()
        w = max(widths)
        li = torch.zeros(len(rows), w, dtype=torch.long)
        lm = torch.zeros(len(rows), w, dtype=torch.bool)
        for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
            li[i, : e - s] = self.legal_idx[s:e]
            lm[i, : e - s] = True
        return b, pl, ca, ep, ci, cm, cp, li.to(device), lm.to(device)


def disable_regularisation(model) -> None:
    """Turn off dropout and stochastic depth for RL fine-tuning.

    Not a style choice -- it is required for the KL controller to mean anything.
    Measured on this model with *identical weights*, a train-mode forward
    against the eval-mode reference reports **KL = 0.0356** from dropout and
    layer-drop alone. Against a 0.05 budget that is 71% of the allowance spent
    on noise before the policy has moved at all, and the controller responds by
    strangling the update (expected cp gained only +3.4 in such a run).

    Removing it also makes ``pi(c)`` in the closed-form gradient an exact
    estimate rather than a sampled one. The regularisation is not missed: the KL
    anchor is a far stronger constraint here than dropout, and unlike the value
    head in expert iteration this objective is anchored to a frozen reference.
    """
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "layer_drop"):
            mod.layer_drop = 0.0
        if hasattr(mod, "dropout") and isinstance(getattr(mod, "dropout"), float):
            mod.dropout = 0.0


def group_advantages(value: torch.Tensor, p_cand: torch.Tensor,
                     cand_mask: torch.Tensor, adv_clip: float) -> torch.Tensor:
    """Group-normalised advantages within each position (the GRPO part).

    The baseline is the POLICY-WEIGHTED mean, not the uniform mean over
    candidates. This is not cosmetic. The update weights each candidate by
    ``pi(c)``, so a baseline that does not satisfy ``sum_c pi~(c) A_c = 0``
    leaves a net push on the whole candidate block against the non-candidate
    legal moves -- a drift term that says nothing about which candidate is
    better. Measured with the uniform baseline, that term swamped the re-ranking
    signal: an unanchored run drove the expected cp of the policy's own moves
    *down* 68cp, where the policy-weighted baseline drives it up 93cp. GRPO
    draws its group from ``pi``, so this is also the faithful reading of the
    method.

    The invariant is worth stating because it is what the test checks:
    ``sum_c pi~(c) A_c == 0`` for every position.
    """
    pnorm = p_cand / p_cand.sum(1, keepdim=True).clamp(min=1e-9)
    mean = (pnorm * value).sum(1, keepdim=True)
    std = (pnorm * (value - mean) ** 2).sum(1, keepdim=True).sqrt()
    adv = (value - mean) / (std + 1e-6)
    # A group whose candidates are all equally good carries no signal, but
    # dividing its ~0 deviations by a ~0 std amplifies float32 rounding into a
    # spurious advantage of a few percent. Gate those groups off explicitly
    # rather than leaning on the epsilon: positions where every legal move is
    # equivalent are common in won and dead-drawn endgames.
    live = std > 1e-4
    return torch.where(live, adv.clamp(-adv_clip, adv_clip), torch.zeros_like(adv)) * cand_mask


def compute_loss(model, ref, batch, beta_kl: float, sample_k: int,
                 adv_clip: float, generator=None):
    """Returns (loss, metrics dict). See the module docstring for the objective."""
    b, pl, ca, ep, ci, cm, cp, li, lm = batch
    n = b.size(0)

    logits, _ = model(b, pl, ca, ep)
    flat = logits.view(n, -1).float()
    with torch.no_grad():
        ref_logits, _ = ref(b, pl, ca, ep)
        ref_flat = ref_logits.view(n, -1).float()

    # log pi over the *legal* set, padded entries masked to -inf so they take no
    # probability mass.
    neg = torch.finfo(flat.dtype).min
    legal_logits = torch.gather(flat, 1, li).masked_fill(~lm, neg)
    logp_legal = F.log_softmax(legal_logits, dim=1)
    ref_legal = torch.gather(ref_flat, 1, li).masked_fill(~lm, neg)
    ref_logp_legal = F.log_softmax(ref_legal, dim=1)

    # Candidates are a subset of the legal moves; find their columns so the
    # gradient flows through the same normalised log-probabilities.
    cand_col = (li.unsqueeze(2) == ci.unsqueeze(1)).float().argmax(dim=1)  # (n, C)
    logp_cand = torch.gather(logp_legal, 1, cand_col)
    p_cand = logp_cand.detach().exp() * cm

    value = cp_to_value(cp) * cm
    adv = group_advantages(value, p_cand, cm, adv_clip)

    if sample_k > 0:
        # Sampled estimator: draw K candidates from the current policy restricted
        # to the candidate set, as scripts/grpo_puzzles.py does.
        probs = (p_cand / p_cand.sum(1, keepdim=True).clamp(min=1e-9))
        samples = torch.multinomial(probs, sample_k, replacement=True,
                                    generator=generator)
        pg = -(torch.gather(adv, 1, samples)
               * torch.gather(logp_cand, 1, samples)).mean(1).mean()
    else:
        # Closed form over the candidate set: same expectation, no sampling noise.
        pg = -(p_cand * adv * logp_cand).sum(1).mean()

    p_legal = logp_legal.exp() * lm
    kl = (p_legal * (logp_legal - ref_logp_legal)).sum(1).mean()
    loss = pg + beta_kl * kl

    with torch.no_grad():
        # Expected cp of the policy's own move distribution, and the same for the
        # frozen base -- the direct answer to "does it pick better moves".
        pc = p_cand / p_cand.sum(1, keepdim=True).clamp(min=1e-9)
        exp_cp = (pc * cp * cm).sum(1).mean()
        ref_p = torch.gather(ref_logp_legal, 1, cand_col).exp() * cm
        ref_pc = ref_p / ref_p.sum(1, keepdim=True).clamp(min=1e-9)
        ref_exp_cp = (ref_pc * cp * cm).sum(1).mean()
        # Does the policy's favourite candidate coincide with the best-scoring one?
        best_cand = torch.where(cm, cp, torch.full_like(cp, -1e9)).argmax(1)
        policy_cand = logp_cand.masked_fill(~cm, neg).argmax(1)
        top1 = (policy_cand == best_cand)
        ent = -(p_legal * logp_legal).sum(1).mean()

    return loss, {
        "pg": float(pg.detach()), "kl": float(kl.detach()), "exp_cp": float(exp_cp),
        "ref_exp_cp": float(ref_exp_cp), "d_cp": float(exp_cp - ref_exp_cp),
        "top1": float(top1.float().mean()), "entropy": float(ent),
    }


@torch.no_grad()
def evaluate(model, ref, table, rows, device, batch_size, beta_kl, adv_clip):
    model.eval()
    agg: dict[str, float] = {}
    nb = 0
    for start in range(0, len(rows), batch_size):
        sl = rows[start:start + batch_size]
        if len(sl) < 2:
            continue
        _, m = compute_loss(model, ref, table.batch(sl, device), beta_kl, 0, adv_clip)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    model.train()
    return {k: v / max(nb, 1) for k, v in agg.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rewards", type=Path, required=True)
    p.add_argument("--base", default="data/models/pos2move_v2.1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta-kl", type=float, default=0.5,
                   help="INITIAL forward-KL coefficient; adapted at runtime to "
                        "hold --target-kl unless --no-adaptive-kl is passed.")
    p.add_argument("--target-kl", type=float, default=0.05,
                   help="KL budget against the frozen base. This is the real "
                        "knob. See the module docstring: a fixed beta of 0.02 "
                        "let KL reach 0.74 and entropy collapse from 1.71 to "
                        "0.20 within 250 steps, which destroys the MCTS prior.")
    p.add_argument("--adaptive-kl", action=argparse.BooleanOptionalAction,
                   default=True, help="steer beta_kl to hold --target-kl")
    p.add_argument("--no-regularisation", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="disable dropout and stochastic depth. On by default: "
                        "they contribute KL 0.0356 on identical weights, which "
                        "eats most of a 0.05 budget before the policy moves.")
    p.add_argument("--sample-k", type=int, default=0,
                   help="0 = closed-form policy gradient over the candidate set "
                        "(default, lower variance); >0 samples K candidates "
                        "instead, matching scripts/grpo_puzzles.py.")
    p.add_argument("--adv-clip", type=float, default=4.0,
                   help="clip normalised advantages; a single mate-scored "
                        "candidate can otherwise dominate its group")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=500,
                   help="checkpoint cadence. Every checkpoint is a gate "
                        "candidate: selection is by match play, not by any "
                        "number this script prints.")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    table = RewardTable(args.rewards)
    train_rows, val_rows = table.split_by_game(args.val_frac, args.seed)
    print(f"reward table: {table.n} positions "
          f"(depth {table.meta[0]}, top-K {table.meta[1]}, top-M {table.meta[2]})")
    print(f"  {len(train_rows)} train / {len(val_rows)} val, split by game")

    model = load_model(args.base, args.device)
    ref = load_model(args.base, args.device).eval()
    for pr in ref.parameters():
        pr.requires_grad_(False)
    if args.no_regularisation:
        disable_regularisation(model)
        disable_regularisation(ref)
    # The value head is frozen: this experiment isolates the policy.
    for name, pr in model.named_parameters():
        if name.startswith("value_head"):
            pr.requires_grad_(False)
    trainable = [pr for pr in model.parameters() if pr.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    base_m = evaluate(model, ref, table, val_rows, args.device,
                      args.batch_size, args.beta_kl, args.adv_clip)
    print(f"base (val): exp_cp {base_m['exp_cp']:+.1f}  "
          f"top1 {base_m['top1']:.3f}  entropy {base_m['entropy']:.3f}")

    base_entropy = base_m["entropy"]
    beta = args.beta_kl
    args.out.mkdir(parents=True, exist_ok=True)
    model.train()
    for step in range(1, args.steps + 1):
        rows = rng.choice(train_rows, size=min(args.batch_size, len(train_rows)),
                          replace=False)
        loss, m = compute_loss(model, ref, table.batch(rows, args.device),
                               beta, args.sample_k, args.adv_clip)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        opt.step()

        if args.adaptive_kl:
            # Standard PPO/RLHF controller. Without it the coefficient has to be
            # guessed per dataset, and guessing low is not a slow failure: the
            # objective's optimum IS a deterministic policy, so the anchor is the
            # only thing holding entropy up.
            if m["kl"] > 1.5 * args.target_kl:
                beta = min(beta * 1.5, 1e3)
            elif m["kl"] < args.target_kl / 1.5:
                beta = max(beta / 1.5, 1e-4)

        if step % 50 == 0:
            print(f"step {step:5d}  loss {float(loss.detach()):+.4f}  pg {m['pg']:+.4f}  "
                  f"kl {m['kl']:.4f} (beta {beta:.3f})  exp_cp {m['exp_cp']:+.1f} "
                  f"(base {m['ref_exp_cp']:+.1f}, d {m['d_cp']:+.1f})  "
                  f"top1 {m['top1']:.3f}  H {m['entropy']:.3f}", flush=True)
        if step % args.eval_every == 0:
            v = evaluate(model, ref, table, val_rows, args.device,
                         args.batch_size, beta, args.adv_clip)
            warn = ""
            if v["entropy"] < 0.6 * base_entropy:
                warn = ("   <-- ENTROPY COLLAPSE: this policy is a degraded MCTS "
                        "prior regardless of its cp numbers")
            print(f"  [val @ {step}] exp_cp {v['exp_cp']:+.1f} "
                  f"(base {v['ref_exp_cp']:+.1f}, d {v['d_cp']:+.1f})  "
                  f"top1 {v['top1']:.3f}  kl {v['kl']:.4f}  "
                  f"entropy {v['entropy']:.3f}/{base_entropy:.3f}{warn}", flush=True)
        if step % args.save_every == 0:
            save_checkpoint(model, args, step)

    save_checkpoint(model, args, args.steps, final=True)
    v = evaluate(model, ref, table, val_rows, args.device,
                 args.batch_size, args.beta_kl, args.adv_clip)
    print(f"\nfinal (val): exp_cp {v['exp_cp']:+.1f} "
          f"(base {base_m['exp_cp']:+.1f}, d {v['exp_cp'] - base_m['exp_cp']:+.1f})  "
          f"top1 {v['top1']:.3f} (base {base_m['top1']:.3f})")
    print("\nNothing here decides promotion. Gate the checkpoints with:")
    print(f"  uv run python scripts/head_to_head.py --book data/openings/book2k.json \\")
    print(f"      --model-a <candidate>.onnx --model-b data/models/pos2move_v2.1/model.onnx \\")
    print(f"      --openings 800 --sprt --nodes-a 1800 --nodes-b 1800")
    return 0


def save_checkpoint(model, args, step: int, final: bool = False) -> None:
    out = args.out if final else args.out / f"step_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    shutil.copy(Path(args.base) / "model_config.json", out / "model_config.json")
    (out / "grpo_meta.json").write_text(json.dumps({
        "base": str(args.base), "rewards": str(args.rewards), "step": step,
        "beta_kl": args.beta_kl, "lr": args.lr, "sample_k": args.sample_k,
        "adv_clip": args.adv_clip,
    }, indent=1))
    print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
