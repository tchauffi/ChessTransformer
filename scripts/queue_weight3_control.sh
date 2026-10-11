#!/usr/bin/env bash
# Unattended: after the head-tower queue (train + gate) finishes, train the same recipe
# WITHOUT towers -- value-loss weight 3 only -- and SPRT-gate it against production v2.1.
#
#   setsid nohup scripts/queue_weight3_control.sh > logs/queue_weight3_control.log 2>&1 &
#
# The tower run changes two things vs run_026 (towers + weight 9 -> 3). This one changes
# only the weight, so tower run vs this run isolates what the towers add, at +19% compute
# per evaluation.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

echo "[$(date '+%F %T')] waiting for the value-target and head-tower queues and any match"
while pgrep -f "[q]ueue_value_targets.sh" >/dev/null || pgrep -f "[b]ash scripts/queue_head_towers.sh" >/dev/null \
    || pgrep -f "[h]ead_to_head.py" >/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] training"
before="$(ls -d logs/pos2move_v2/run_* 2>/dev/null | sort | tail -1)"
scripts/train_value_targets.sh --value-loss-weight 3 || { echo "training failed"; exit 1; }
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
