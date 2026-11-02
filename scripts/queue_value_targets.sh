#!/usr/bin/env bash
# Unattended: wait for the v2 shard build and any GPU match, train the value-target run,
# then SPRT-gate it against production v2.1 on the 2k book.
#
#   setsid nohup scripts/queue_value_targets.sh > logs/queue_value_targets.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

echo "[$(date '+%F %T')] waiting for data/shards/full2500_k8_v2/meta.json"
until [[ -f data/shards/full2500_k8_v2/meta.json ]]; do sleep 60; done
if pgrep -f "[b]uild_shards.py" >/dev/null; then
  echo "build_shards still running after meta.json appeared?"; exit 1
fi
tail -12 data/shards/full2500_k8_v2.build.log

# The GPU is shared: a match running beside training slows both and would make the
# training wall-clock meaningless.
echo "[$(date '+%F %T')] waiting for running head_to_head matches"
while pgrep -f "[h]ead_to_head.py" >/dev/null; do sleep 60; done

echo "[$(date '+%F %T')] training"
before="$(ls -d logs/pos2move_v2/run_* 2>/dev/null | sort | tail -1)"
scripts/train_value_targets.sh || { echo "training failed"; exit 1; }
RUN="$(ls -d logs/pos2move_v2/run_* | sort | tail -1)"
if [[ "$RUN" == "$before" ]]; then echo "no new run dir"; exit 1; fi
echo "[$(date '+%F %T')] trained $RUN"

# Gate best_model, as run_023 was, so the two gates are comparable.
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
