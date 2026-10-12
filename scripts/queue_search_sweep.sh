#!/usr/bin/env bash
# Search settings for a new network: each setting of MODEL (net A) against BASE at the
# deployed defaults (c_puct 1.0, fpu 0.2, prior_temp 1.0), one gpu_match per setting.
#
#   systemd-run --user --unit ct-search-sweep --working-directory "$PWD" \
#     bash -c 'scripts/queue_search_sweep.sh > logs/search_sweep.log 2>&1'
#
# gen00's defaults were tuned on older networks, and only towards a wider search (c_puct 2.0
# / fpu 0.1: -19 Elo, ns, 2026-08-19); the prior temperature was never matched. run_029
# (smolgen) lost -39 to gen00 at the defaults: a network that judges unusual positions worse
# may want a narrower search, one with a sharper policy a wider one -- both directions are
# here. Half the deployed budget so a setting costs ~50 min; confirm a winner at 4000 sims,
# and match it on BASE too: a setting that helps both networks is a bot gain, not an
# argument for the new one.
#
# Waits for any running match first; stops at 22:00 (no GPU at night). A setting whose
# games file holds a finished match is skipped, so a relaunch resumes.
set -uo pipefail
CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$PWD/.venv/bin/python}"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-logs/pos2move_v2/run_029_20261012_113626/checkpoints/best_model}"
BASE="${BASE:-data/rl/models/gen00}"
SIMS="${SIMS:-2000}"
PONDER="${PONDER:-1000}"
OPENINGS="${OPENINGS:-100}"
SETTINGS=(
  "default:"
  "pt1.3:--prior-temp-a 1.3"
  "pt0.8:--prior-temp-a 0.8"
  "cpuct1.5:--c-puct-a 1.5"
  "fpu0.3:--fpu-a 0.3"
)
log() { echo "[$(date '+%F %T')] $*"; }

while pgrep -f "python[^ ]* .*gpu_match\.py" >/dev/null; do sleep 60; done

for entry in "${SETTINGS[@]}"; do
  tag="${entry%%:*}"; flags="${entry#*:}"
  out="data/rl/gates/sweep_${tag}_vs_$(basename "$BASE").jsonl"
  done_flag="$out.done"
  [[ -f "$done_flag" ]] && { log "$tag: already done"; continue; }
  left=$(( $(date -d '22:00' +%s) - $(date +%s) ))
  ((left > 600)) || { log "night window: stopping before $tag"; exit 0; }
  log "$tag ($flags) at $SIMS sims + $PONDER ponder, up to $((2 * OPENINGS)) games"
  rm -f "$out"
  # shellcheck disable=SC2086
  timeout "$left" "$PY" "$CODE_ROOT/scripts/gpu_match.py" --model-a "$MODEL" --model-b "$BASE" \
    --label-a "$tag" --label-b "$(basename "$BASE")" --openings "$OPENINGS" \
    --max-sims "$SIMS" --ponder-max-sims "$PONDER" --sprt --games-out "$out" $flags 2>&1 \
    | grep -v -i warn | tail -12
  rc=${PIPESTATUS[0]}
  if ((rc == 0)); then touch "$done_flag"
  elif ((rc == 124)); then log "$tag: stopped at 22:00 with $(wc -l < "$out") games"; exit 0
  else log "$tag: match failed (rc $rc)"; fi
done
log "done"
