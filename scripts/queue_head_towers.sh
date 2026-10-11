#!/usr/bin/env bash
# Unattended: after the run_026 queue (train + gate) finishes, train the per-head-tower
# variant of the value-target recipe and SPRT-gate it against production v2.1.
#
#   setsid nohup scripts/queue_head_towers.sh > logs/queue_head_towers.log 2>&1 &
#
# Same recipe as run_026 (tablebase labels, rating-gap offset, WDL, 50k steps) plus:
#   --head-tower-layers 3   the last 3 of 16 layers private to each head (13.8M params,
#                           +19% compute per evaluation)
#   --value-loss-weight 3   the value objective's share of the 13 shared layers back to
#                           run_023's (see the measurement in train_value_targets.sh)
#   --value-pool attn       the value head reads a learned-query attention pool over all 67
#                           tokens of its tower instead of the mean of the 3 state tokens
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# One GPU job at a time: the run_026 queue still has its gate to play.
echo "[$(date '+%F %T')] waiting for queue_value_targets.sh and any match"
while pgrep -f "[q]ueue_value_targets.sh" >/dev/null || pgrep -f "[h]ead_to_head.py" >/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] training"
before="$(ls -d logs/pos2move_v2/run_* 2>/dev/null | sort | tail -1)"
scripts/train_value_targets.sh --head-tower-layers 3 --value-loss-weight 3 --value-pool attn \
  || { echo "training failed"; exit 1; }
RUN="$(ls -d logs/pos2move_v2/run_* | sort | tail -1)"
if [[ "$RUN" == "$before" ]]; then echo "no new run dir"; exit 1; fi
echo "[$(date '+%F %T')] trained $RUN"

CAND="$RUN/checkpoints/best_model"
cp "$RUN/model_config.json" "$CAND/"
uv run --with onnxscript python scripts/export_onnx.py "$CAND" -o "$CAND/model.onnx" \
  || { echo "export failed"; exit 1; }
uv run python scripts/check_onnx_parity.py --model "$CAND" --onnx "$CAND/model.onnx" \
  || { echo "ONNX parity failed -- not gating a graph that disagrees with torch"; exit 1; }

echo "[$(date '+%F %T')] gating $CAND vs prod v2.1"
mkdir -p data/eval/value_targets
WORKERS=8 scripts/gate_candidate.sh "$CAND" --label-a "$(basename "$RUN")" \
  --games-out "data/eval/value_targets/$(basename "$RUN")_vs_v2.1.jsonl"
echo "[$(date '+%F %T')] done"
