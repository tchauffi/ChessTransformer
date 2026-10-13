#!/usr/bin/env bash
# The smolgen network into the RL loop (user, 2026-10-13: "train the smolgen model with self
# play games then make it self play"). Runs as the loop's own unit, so the 22:00-08:00
# freeze applies:
#
#   systemd-run --user --unit ct-rl-loop --on-calendar "2026-10-14 08:00:00" \
#     --working-directory "$PWD" --setenv PATH=... /bin/bash scripts/queue_smolgen_rl.sh
#
# sg_base = run_029 (smolgen + full attention bias, human data) with its rule-state inputs
# folded out (convert_attn_bias.py --drop-rule-state, branch feat/rule-state-inputs): it
# plays as gpu_match --blind-rule-state does, -1 Elo vs gen00 over 400 games. The self-play
# data carries no fifty-move clock, so a rule-state net could not be trained on it anyway.
#
# 1. sg_base trained on gen01 + gen02 self-play with the loop's own recipe (rl_generation.sh
#    step 2: 2 epochs, lr 1e-4, KL 0.1 to sg_base, 20% human replay, compiled), exported,
#    parity-checked, then installed as cand02. The old-architecture cand02 (gen01 on the
#    same data, +6 Elo over gen01 after 387 of 400 gate games) moves to cand02_old with its
#    gate games.
# 2. rl_generation.sh gates the new cand02 against the champion gen01 and promotes it
#    whatever the score (FORCE_PROMOTE=1; user: "Let's use smolgen for now as it is
#    theoretically the best model"): the gate measures where the smolgen line starts, and
#    history.tsv reads "forced" if it lost. The smolgen net then plays gen03's self-play
#    and the loop goes on back to back, as ct-rl-loop always has.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RL=data/rl
SG=$RL/models/sg_base
CAND=$RL/models/cand02
NEW=$RL/models/cand02_sg2
LOG=logs/rl_generation.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

[[ "$(readlink "$RL/champion")" == models/gen01 ]] || { log "smolgen: champion is not gen01, not starting"; exit 1; }
[[ -f "$RL/selfplay/gen02/.selfplay_done" && ! -f "$RL/selfplay/gen02/.gate_done" ]] \
  || { log "smolgen: gen02 is not waiting for its gate, not starting"; exit 1; }

if [[ ! -f "$CAND/SOURCE" ]] || ! grep -q sg_base "$CAND/SOURCE"; then
  if [[ ! -f "$NEW/model.onnx" ]]; then
    log "smolgen: training $NEW from $SG on gen01 + gen02, 2 epochs"
    uv run python scripts/train_expert_iter.py --data "$RL/selfplay/gen01" "$RL/selfplay/gen02" \
      --base "$SG" --out "$NEW" --epochs 2 --batch-size 512 --lr 1e-4 --kl-coef 0.1 \
      --soft-mix 0.5 --value-weight 1.0 --human-shards data/shards/elite_k32_v2 --human-frac 0.2 \
      --select last --compile >> "$LOG" 2>&1 || { log "smolgen: training failed"; exit 1; }
    uv run --with onnxscript python scripts/export_onnx.py "$NEW" -o "$NEW/model.onnx" >> "$LOG" 2>&1 \
      || { log "smolgen: export failed"; rm -f "$NEW/model.onnx"; exit 1; }
    uv run python scripts/check_onnx_parity.py --model "$NEW" --onnx "$NEW/model.onnx" >> "$LOG" 2>&1 \
      || { log "smolgen: ONNX parity failed"; rm -f "$NEW/model.onnx"; exit 1; }
  fi
  echo "$SG trained on gen01 + gen02 by scripts/queue_smolgen_rl.sh" > "$NEW/SOURCE"
  if [[ -d "$CAND" && ! -e "$RL/models/cand02_old" ]]; then mv "$CAND" "$RL/models/cand02_old"; fi
  if [[ -f "$RL/gates/cand02_vs_gen01.jsonl" && ! -e "$RL/gates/cand02_old_vs_gen01.jsonl" ]]; then
    mv "$RL/gates/cand02_vs_gen01.jsonl" "$RL/gates/cand02_old_vs_gen01.jsonl"
  fi
  [[ -e "$CAND" ]] && { log "smolgen: $CAND still exists, not replacing it"; exit 1; }
  cp -r "$NEW" "$CAND"
  log "smolgen: $CAND is now sg_base trained on gen01 + gen02 (old cand02 in cand02_old)"
fi

FORCE_PROMOTE=1 /bin/bash scripts/rl_generation.sh >> "$LOG" 2>&1 \
  || { log "smolgen: generation failed, loop stopped"; exit 1; }
[[ "$(readlink "$RL/champion")" == models/gen02 ]] || { log "smolgen: gen02 not installed, loop stopped"; exit 1; }
log "smolgen: champion is the smolgen net (gen02); the loop goes on"
while [[ ! -f "$RL/STOP" ]]; do
  /bin/bash scripts/rl_generation.sh >> "$LOG" 2>&1 || { log "generation failed, loop stopped"; exit 1; }
done
log "STOP file found, loop ended"
