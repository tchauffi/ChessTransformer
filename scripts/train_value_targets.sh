#!/usr/bin/env bash
# Value-target experiment: does a cleaner outcome label make a stronger bot?
#
# Two corrections to the value head's target z, both measured 2026-10-08 on the
# ELO>=2500 corpus (games with avg rating >= 2500):
#   * Rating gap. The players' rating gap alone explains R2 0.080 of game outcomes --
#     more than v2.1's head explains in the opening (0.015). The net has no rating input,
#     so it attributes the stronger player's results to the position; that is the 1.a4
#     drift in docs/search-scaling-and-value-bias.md. --elo-offset explains the gap with a
#     learned logit offset at training time only; inference reads the head as equals.
#   * Tablebase adjudication. 21.5% of games that reach <=5 men end with a result Syzygy
#     contradicts (blitz scrambles, flags). --value-labels tablebase trains on result_tb:
#     3.2% of samples change.
#
# Everything else matches run_023 (base preset, eff. batch 2048, 50k steps, lr 1e-3) and
# data/shards/full2500_k8_v2 holds byte-identical samples to full2500_k8 -- only the new
# label fields differ. The offset needs a WDL head, so this is also the first WDL run.
#
# --value-loss-weight 9 was meant to keep run_023's value/policy balance in the trunk
# (MSE 5.0, ~4.0 effective after the old ramp-broadcast bug), estimated from the loss
# gradient at the head's output. MEASURED 2026-10-08 at step 40k, through the trunk itself:
# |g_value|/|g_policy| per layer is 1.3-5.3 in run_026 vs 0.26-1.83 in run_023, i.e. ~2.75x
# run_023's value share in the shared layers (median over layers 0-13), and ~5x in the last
# three. Policy CE trailed run_023 by a flat ~0.035 from 15k on. Both optimizers are
# scale-invariant, so the weight sets only that share -- ~3 matches run_023. Override with
# trailing args (argparse keeps the last): `scripts/train_value_targets.sh --value-loss-weight 3`.
# --value-rampup-halfmoves 0 is what the old bug effectively did too.
#
# Control run (same WDL head and weight, human labels, no offset):
#   LABELS=human ELO_OFFSET=0 scripts/train_value_targets.sh
#
# Gate with scripts/gate_candidate.sh, never with val loss: val loss is computed on the
# run's own labels, so it is not comparable across label settings.
set -euo pipefail
cd "$(dirname "$0")/.."

LABELS="${LABELS:-tablebase}"
ELO_OFFSET="${ELO_OFFSET:-1}"
EXTRA=(--value-labels "$LABELS")
[[ "$ELO_OFFSET" == "1" ]] && EXTRA+=(--elo-offset)

exec /home/tchauffi/.local/bin/uv run python \
  src/chesstransformer/trainers/pos2move_v2_trainer.py \
  --shards data/shards/full2500_k8_v2 \
  --preset base \
  --batch-size 512 --grad-accum 4 \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-steps 1000 \
  --save-steps 5000 --max-checkpoints 10 \
  --num-workers 0 \
  --compile \
  --value-head wdl --value-loss-weight 9 --value-rampup-halfmoves 0 \
  "${EXTRA[@]}" \
  "$@"
