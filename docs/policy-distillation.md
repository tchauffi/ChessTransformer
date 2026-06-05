# Stockfish policy distillation — findings (negative result)

**TL;DR: post-hoc Stockfish policy distillation does not improve Pos2MoveV2 and
slightly hurts it. The best distilled model scored ~1650 Elo vs Stockfish, ~125
Elo *below* the v2.1 baseline (~1775). Do not pursue. Kept here as tooling +
a documented dead end.**

## Motivation

Probing the v2.1 heads against Stockfish (depth 12) showed an asymmetry:

| Head | Quality vs Stockfish |
|---|---|
| Value | r ≈ 0.97, sign-agreement 95%, MAE 0.12 — near-ceiling |
| Policy | top-1 ≈ 52%, top-3 ≈ 79% — the weaker head |

A uniform-prior ablation (real policy beat uniform 16–0 in MCTS) confirmed the
policy is load-bearing. Hypothesis: distilling a sharper policy from Stockfish
would raise strength.

## What we tried

`scripts/gen_sf_policy_labels.py` → Stockfish `multipv` soft labels over the
64×73 action planes (raw centipawns stored so the softmax temperature is tunable
at train time). `scripts/distill_policy.py` → KL-divergence distillation with the
value head pinned to a frozen teacher. `scripts/eval_policy.py` → apples-to-apples
held-out policy comparison (top-1/top-3/cp-loss vs Stockfish).

Configs swept: 24k and **200k** labels (depth 12); full fine-tune, frozen-backbone
(head-only), and gentle low-lr; temperatures 40–100 cp.

## Results

Held-out policy (vs Stockfish, depth 12):

| Model | top-1 | top-3 | cp lost |
|---|---|---|---|
| v2.1 baseline | 52–53% | 79% | 23–27c |
| distill-head (24k, frozen backbone) | 54.0% | 80.7% | 39c |
| distill (200k, gentle) | 49.2% | 76.8% | 33c |
| distill (24k, full FT) | 43.0% | 72.3% | 51c |

Stockfish gauntlet (MCTS @ 400 sims, 20 games/level, skills 0–12):

| Model | Weighted Elo | skill 8 | skill 10 | skill 12 |
|---|---|---|---|---|
| **v2.1** | **~1775** | 62.5% | 40% | 27.5% |
| distill-head (best by policy metric) | **~1650** | 52.5% | 20% | 10% |

## Why it failed

- **More data didn't help** (24k → 200k unchanged) → the policy is near the
  11.7M model's capacity ceiling, not data-limited.
- **top-1/top-3 is a misleading proxy.** distill-head had marginally better
  top-k but worse **cp-loss** (39c vs ~25c). Game strength tracks cp-loss (the
  quality of the chosen move), not top-k match rate. Distilling toward
  Stockfish's distribution nudged "name the popular move" up while making the
  model's wrong moves *more* wrong — and that lost ~125 Elo, worst at high skill.

## Takeaway

A stronger policy needs **more model capacity or a from-scratch retrain with
Stockfish targets blended into the loss**, not a post-hoc graft. The value head
is already excellent. The proven strength lever is **search (MCTS sims)**.
