#!/usr/bin/env bash
# Unattended: gen00's recipe (run_028, scripts/queue_noblitz.sh) trained from scratch with the
# two architecture additions -- the full attention-bias table + smolgen, and the rule-state
# inputs (halfmove clock, repetition count) -- then a match against gen00 itself.
#
#   systemd-run --user --unit ct-human-sgr --working-directory "$PWD" \
#     bash -c 'scripts/queue_human_sgr.sh > logs/queue_human_sgr.log 2>&1'
#
# Same corpus (elite_db.h5, K=32, seed 0: the same samples as elite_k32_v2 plus the two v3
# fields), same steps, lr, value recipe and seed as run_028: only the architecture differs.
# The self-play fine-tune of smolgen (cand02_sg) lost -31 Elo after over-fitting 4.6M
# correlated self-play positions; here the new parameters learn from scratch with the rest.
#
# Code runs from this script's checkout (CODE_ROOT); data and logs live under the working
# directory, so a worktree can drive the main checkout's data/.
set -uo pipefail
CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$PWD/.venv/bin/python}"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SHARDS=data/shards/elite_k32_v3
BASE=data/rl/models/gen00
GAMES_OUT=data/rl/gates/human_sgr_vs_gen00.jsonl
log() { echo "[$(date '+%F %T')] $*"; }

if [[ ! -f "$SHARDS/meta.json" ]]; then
  log "building $SHARDS"
  nice -n 19 "$PY" "$CODE_ROOT/scripts/build_shards.py" --h5 data/elite_db.h5 --out "$SHARDS" \
    -k 32 --syzygy data/syzygy --workers 4 --overwrite \
    || { log "shard build failed"; exit 1; }
fi

log "training"
before="$(ls -d logs/pos2move_v2/run_* 2>/dev/null | sort | tail -1)"
"$PY" "$CODE_ROOT/src/chesstransformer/trainers/pos2move_v2_trainer.py" \
  --shards "$SHARDS" --preset base --batch-size 512 --grad-accum 4 --lr 1e-3 \
  --max-steps 50000 --eval-steps 1000 --save-steps 5000 --max-checkpoints 10 \
  --num-workers 0 --compile \
  --value-head wdl --value-loss-weight 3 --value-rampup-halfmoves 0 \
  --value-labels tablebase --elo-offset --head-tower-layers 3 --value-pool attn \
  --attn-bias full --smolgen --rule-state \
  || { log "training failed"; exit 1; }
RUN="$(ls -d logs/pos2move_v2/run_* | sort | tail -1)"
if [[ "$RUN" == "$before" ]]; then log "no new run dir"; exit 1; fi
log "trained $RUN"

CAND="$RUN/checkpoints/best_model"
cp "$RUN/model_config.json" "$CAND/"
"$PY" "$CODE_ROOT/scripts/export_onnx.py" "$CAND" -o "$CAND/model.onnx" \
  || { log "export failed"; exit 1; }
"$PY" "$CODE_ROOT/scripts/check_onnx_parity.py" --model "$CAND" --onnx "$CAND/model.onnx" \
  || { log "ONNX parity failed"; exit 1; }

# No GPU at night: the match stops at 22:00 and waits for 08:00 if training ends late.
hour=$((10#$(date +%H)))
if ((hour >= 21 || hour < 8)); then
  log "night window: match deferred to 08:00"
  day=$( ((hour < 8)) && echo today || echo tomorrow )
  sleep $(( $(date -d "$day 08:00" +%s) - $(date +%s) ))
fi
left=$(( $(date -d '22:00' +%s) - $(date +%s) ))
log "match $CAND vs $BASE, up to 400 games (stops in $((left / 60)) min at the latest)"
ok=0
for extra in "" "--no-compile --engines 1"; do
  rm -f "$GAMES_OUT"
  timeout "$left" "$PY" "$CODE_ROOT/scripts/gpu_match.py" --model-a "$CAND" --model-b "$BASE" \
    --label-a human_sgr --label-b gen00 --openings 200 --max-sims 4000 --ponder-max-sims 2000 \
    --sprt --games-out "$GAMES_OUT" $extra | tail -14
  rc=${PIPESTATUS[0]}
  ((rc == 0)) && { ok=1; break; }
  ((rc == 124)) && { log "stopped at 22:00 with $(wc -l < "$GAMES_OUT") games"; exit 0; }
  log "match failed${extra:+ ($extra)}"
done
((ok)) || exit 1
log "done"
