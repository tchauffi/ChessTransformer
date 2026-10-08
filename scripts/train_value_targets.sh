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
# --value-loss-weight 9: run_023 trained MSE at 5.0, which the old ramp-broadcast bug
# made ~4.0 effective, i.e. a per-sample value gradient of ~4 x 2|v-z| ~ 6.8 against the
# policy's ~0.8. WDL cross-entropy's gradient is ~0.77 per unit weight, so 9 keeps the
# value/policy balance in the shared trunk where production had it. --value-rampup-halfmoves
# 0 is what the bug effectively did too. Not swept -- a deliberate "change one thing" choice.
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
