#!/usr/bin/env bash
# The smolgen network into the RL loop, learning only from its own KataGo self-play (user,
# 2026-10-13: "Let's use smolgen for now as it is theoretically the best model", then
# "regenerate katago self play with the smolgen model to avoid give in a prior that will
# lock the model in a too prune mode"). Runs as the loop's own unit, so the 22:00-08:00
# freeze applies:
#
#   systemd-run --user --unit ct-rl-loop --on-calendar "2026-10-14 08:00:00" \
#     --working-directory "$PWD" --setenv PATH=... /bin/bash scripts/queue_smolgen_rl.sh
#
# sg_base = run_029 (smolgen + full attention bias, human data) with its rule-state inputs
# folded out (convert_attn_bias.py --drop-rule-state, branch feat/rule-state-inputs): it
# plays as gpu_match --blind-rule-state does, -1 Elo vs gen00 over 400 games.
#
# 1. sg_base becomes the champion as gen02, untrained and ungated (history.tsv: "forced",
#    no gate score). gen02's self-play, played by gen01 with plain MCTS, is never trained
#    on: its targets carry the older networks' narrowing policy (entropy 1.72 -> 1.54 ->
#    1.48 from gen00 to cand02). The old-architecture cand02 stays unused in models/.
# 2. data/rl/forced_playouts = 2: every self-play from now on runs KataGo forced playouts +
#    policy target pruning on its full searches.
# 3. gen03 = sg_base's own self-play; cand03 trains on it alone (WINDOW=1) and is gated
#    against sg_base. Then the loop goes on back to back with the usual two-generation
#    window, all of it smolgen KataGo self-play.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RL=data/rl
SG=$RL/models/sg_base
LOG=logs/rl_generation.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [[ "$(readlink "$RL/champion")" == models/gen01 ]]; then
  [[ -f "$RL/selfplay/gen02/.selfplay_done" && ! -f "$RL/selfplay/gen02/.gate_done" ]] \
    || { log "smolgen: gen02 is not waiting for its gate, not starting"; exit 1; }
  [[ -e "$RL/models/gen02" ]] && { log "smolgen: $RL/models/gen02 already exists, not starting"; exit 1; }
  cp -r "$SG" "$RL/models/gen02"
  echo "$SG installed untrained by scripts/queue_smolgen_rl.sh (no gate)" > "$RL/models/gen02/SOURCE"
  read -r POS NGAMES < <(uv run python -c "import numpy as np, glob, sys; print(sum(len(np.load(f)['z']) for f in glob.glob(sys.argv[1] + '/positions_*.npz')), sum(1 for _ in open(sys.argv[1] + '/games.jsonl')))" "$RL/selfplay/gen02")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" gen02 "$POS" "$NGAMES" - "forced" - >> "$RL/history.tsv"
  touch "$RL/selfplay/gen02/.gate_done" "$RL/selfplay/gen02/.generation_done"
  ln -sfn models/gen02 "$RL/champion"
  echo 2 > "$RL/forced_playouts"
  log "smolgen: champion is sg_base (gen02, untrained); KataGo forced playouts (k=2) on from gen03"
fi
grep -q sg_base "$RL/models/gen02/SOURCE" 2>/dev/null || { log "smolgen: gen02 is not sg_base, not starting"; exit 1; }

if [[ ! -f "$RL/selfplay/gen03/.generation_done" ]]; then
  WINDOW=1 /bin/bash scripts/rl_generation.sh >> "$LOG" 2>&1 || { log "smolgen: gen03 failed, loop stopped"; exit 1; }
fi
while [[ ! -f "$RL/STOP" ]]; do
  /bin/bash scripts/rl_generation.sh >> "$LOG" 2>&1 || { log "generation failed, loop stopped"; exit 1; }
done
log "STOP file found, loop ended"
