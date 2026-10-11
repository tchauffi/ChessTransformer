#!/usr/bin/env bash
# Unattended: the run_027 recipe (per-head towers, tablebase labels, rating-gap offset) on
# rapid + classical games only, then an SPRT gate against production v2.1.
#
#   setsid nohup scripts/queue_noblitz.sh > logs/queue_noblitz.log 2>&1 &
#
# Every model trained on the blitz-inclusive corpus has trailed v2.1 (run_019 -89 Elo,
# run_023 -24, run_027 -49 at 37 games), and v2.1 itself saw only rapid + classical.
# The 66 raw months hold 24.8M blitz vs 1.48M rapid + 59k classical games;
# data/elite_db.h5 (1,533,352 games) is that non-blitz set -- v2.1's own corpus, with no
# ELO filter, as v2.1 had none. So this run differs from v2.1 in recipe only, and from
# run_027 in data only.
#
# K=32: 50k steps x 2048 consumes ~102M samples, ~2 passes over ~48M. The value label is
# per game, so 1.5M games is the real limit on the value head; best_model is chosen on val
# loss, which catches overfitting.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SHARDS=data/shards/elite_k32_v2

if [[ ! -f "$SHARDS/meta.json" ]]; then
  echo "[$(date '+%F %T')] building $SHARDS"
  uv run python scripts/build_shards.py --h5 data/elite_db.h5 --out "$SHARDS" -k 32 \
    --syzygy data/syzygy --workers 6 --overwrite \
    || { echo "shard build failed"; exit 1; }
fi

echo "[$(date '+%F %T')] waiting for any match"
while pgrep -f "[h]ead_to_head.py" >/dev/null; do sleep 60; done

echo "[$(date '+%F %T')] training"
before="$(ls -d logs/pos2move_v2/run_* 2>/dev/null | sort | tail -1)"
scripts/train_value_targets.sh --shards "$SHARDS" \
  --head-tower-layers 3 --value-loss-weight 3 --value-pool attn \
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
