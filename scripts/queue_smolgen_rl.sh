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
#    against sg_base.
# 4. Policy-target A/B (user: "Policy improvement by planning with Gumbel looks
#    promissing"): cand03g = the same training on the same games with Gumbel completed-Q
#    targets, then 400 games cand03g vs cand03 (SPRT +0/+15). Only if the SPRT accepts
#    H1 do later generations train on Gumbel targets ($RL/policy_target); the champion
#    is left as it is.
# 5. The loop goes on back to back with the usual two-generation window, all of it
#    smolgen KataGo self-play.
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
  WINDOW=1 POLICY_TARGET=visits /bin/bash scripts/rl_generation.sh >> "$LOG" 2>&1 \
    || { log "smolgen: gen03 failed, loop stopped"; exit 1; }
fi

AB=$RL/gates/cand03g_vs_cand03.jsonl
G3=$RL/models/cand03g
if [[ ! -f "$AB.done" ]]; then
  if [[ ! -f "$G3/model.safetensors" ]]; then
    log "gumbel A/B: training $G3 from gen02 on gen03 with Gumbel completed-Q targets"
    uv run python scripts/train_expert_iter.py --data "$RL/selfplay/gen03" --base "$RL/models/gen02" \
      --out "$G3" --epochs 2 --batch-size 512 --lr 1e-4 --kl-coef 0.1 --soft-mix 0.5 \
      --value-weight 1.0 --human-shards data/shards/elite_k32_v2 --human-frac 0.2 --select last \
      --compile --policy-target gumbel >> "$LOG" 2>&1 || { log "gumbel A/B: training failed"; exit 1; }
  fi
  log "gumbel A/B: cand03g vs cand03, up to 400 games"
  ok=0
  for extra in "" "--no-compile --engines 1"; do
    rm -f "$AB"
    # shellcheck disable=SC2086
    uv run python scripts/gpu_match.py --model-a "$G3" --model-b "$RL/models/cand03" --label-a cand03g \
      --label-b cand03 --openings 200 --max-sims 4000 --ponder-max-sims 2000 --sprt \
      --games-out "$AB" $extra 2>&1 | tail -14 >> "$LOG" && { ok=1; break; }
    log "gumbel A/B: match failed${extra:+ ($extra)}"
  done
  ((ok)) || exit 1
  verdict=$(uv run python - "$AB" <<'PY'
import json, sys
sys.path.insert(0, "src")
from chesstransformer.evaluation.sprt import pair_up, sprt_decision, sprt_llr
rows = [json.loads(l) for l in open(sys.argv[1])]
llr = sprt_llr(pair_up({(r["idx"], r["a_white"]): r["score_a"] for r in rows}))
print(f"{sprt_decision(llr) or 'undecided'} {llr:+.2f} {sum(r['score_a'] for r in rows) / len(rows):.4f} {len(rows)}")
PY
)
  read -r decision llr score games <<< "$verdict"
  log "gumbel A/B: cand03g scored $score over $games games, LLR $llr: $decision"
  if [[ "$decision" == H1 ]]; then
    echo gumbel > "$RL/policy_target"
    log "gumbel A/B: later generations train on Gumbel completed-Q targets"
  else
    log "gumbel A/B: later generations keep the visit-count targets"
  fi
  touch "$AB.done"
fi
while [[ ! -f "$RL/STOP" ]]; do
  /bin/bash scripts/rl_generation.sh >> "$LOG" 2>&1 || { log "generation failed, loop stopped"; exit 1; }
done
log "STOP file found, loop ended"
