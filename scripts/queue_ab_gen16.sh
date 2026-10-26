#!/usr/bin/env bash
# One-off (2026-10-26): A/B two training-recipe changes on gen16's games before the loop gates
# its candidate. Three candidates from gen15 on gen16 alone, identical but for one setting:
#   cand16a  the loop's recipe (visit targets at T=1, value target 0.5 z + 0.5 root_v)
#   cand16b  POLICY_TEMP 1.3: one RL generation at T=1 narrowed the policy from 1.75 to 1.45
#            nats (12 -> 8 moves >= 0.5%), worth ~40 Elo at the bot's 4000 sims
#   cand16c  SOFT_MIX 0.8: the 0.5 mix is twice as far from Stockfish as the root value
# b and c each play a at the bot's settings (the gate's match, SPRT +0/+15, up to 400 games).
# The best arm that beats a (SPRT H1) becomes cand16 and its setting the loop's from gen17,
# else cand16a does; then the loop resumes and gates cand16 against gen15 as usual.
#
#   scripts/queue_ab_gen16.sh --wait   # its own unit: once gen16's self-play is done, stop the
#                                      # loop before it trains, run this as ct-rl-loop
#   scripts/queue_ab_gen16.sh          # the A/B itself, then `make loop`
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export LC_ALL=C
RL=data/rl
SP=$RL/selfplay/gen16
CHAMP=$RL/models/gen15
HUMAN_SHARDS=data/shards/elite_k32_v2
log() { echo "[$(date '+%F %T')] ab-gen16: $*"; }

if [[ "${1:-}" == "--wait" ]]; then
  until [[ -f $SP/.selfplay_done ]]; do
    systemctl --user is-active -q ct-rl-loop.service || { log "loop stopped before gen16's self-play ended"; exit 1; }
    sleep 2
  done
  log "gen16 self-play done: stopping the loop before it trains cand16"
  systemctl --user stop ct-rl-loop.service
  systemctl --user reset-failed ct-rl-loop.service 2>/dev/null
  exec systemd-run --user --unit ct-rl-loop -p WorkingDirectory="$PWD" -E UV_NO_SYNC=1 -E LC_ALL=C \
    -E PATH="$PATH" /bin/bash -c "exec scripts/queue_ab_gen16.sh >> logs/rl_generation.log 2>&1"
fi

[[ -f $SP/.selfplay_done ]] || { log "gen16 self-play is not done"; exit 1; }
[[ ! -f $SP/.gate_done ]] || { log "cand16 is already gated"; exit 1; }
[[ "$(readlink $RL/champion)" == models/gen15 ]] || { log "the champion is not gen15"; exit 1; }
# The stopped loop may have begun cand16 and drawn gen16's held-out games (kept if whole).
[[ -f $RL/models/cand16/model.safetensors ]] || rm -rf $RL/models/cand16
if [[ -f $SP/heldout_games.json ]] && ! python3 -c "import json, sys; json.load(open(sys.argv[1]))" $SP/heldout_games.json; then
  rm -f $SP/heldout_games.json
fi

train() {  # <out> <policy temp> <soft mix>: the loop's training, export and parity check
  local out=$1
  [[ -f $out/.ready ]] && return 0
  log "training $out (policy temperature $2, soft mix $3)"
  uv run python scripts/train_expert_iter.py --data $SP --base $CHAMP --out $out \
    --epochs 1 --batch-size 512 --lr 1e-4 --kl-coef 0.1 --soft-mix "$3" --value-weight 1.0 \
    --human-shards $HUMAN_SHARDS --human-frac 0.2 --select last --compile \
    --policy-target visits --policy-temp "$2" || return 1
  uv run --with onnxscript python scripts/export_onnx.py $out -o $out/model.onnx || return 1
  uv run python scripts/check_onnx_parity.py --model $out --onnx $out/model.onnx || return 1
  touch $out/.ready
}

match() {  # <a> <b> <games out>: the gate's match, a vs b
  local f=$3 extra
  [[ -f $f.done ]] && return 0
  for extra in "" "--no-compile --engines 1"; do
    rm -f "$f"
    uv run python scripts/gpu_match.py --model-a "$1" --model-b "$2" --label-a "$(basename "$1")" \
      --label-b "$(basename "$2")" --openings 200 --max-sims 4000 --ponder-max-sims 2000 \
      --parallel-games 128 --threads 4 --games-out "$f" --sprt --elo0 0 --elo1 15 $extra | tail -14 \
      && { touch "$f.done"; return 0; }
    log "match failed${extra:+ ($extra)}"
  done
  return 1
}

A=$RL/models/cand16a B=$RL/models/cand16b C=$RL/models/cand16c
train $A 1.0 0.5 && train $B 1.3 0.5 && train $C 1.0 0.8 || { log "training failed"; exit 1; }
CUDA_VISIBLE_DEVICES= uv run python scripts/policy_entropy.py $SP 1500 gen15=$CHAMP cand16a=$A cand16b=$B cand16c=$C \
  || log "policy entropy failed (not needed for the decision)"

BEST=$A BEST_ELO=0
for arm in $B $C; do
  f=$RL/gates/$(basename $arm)_vs_cand16a.jsonl
  log "$(basename $arm) vs cand16a at the bot's settings"
  match $arm $A $f || { log "match failed"; exit 1; }
  read -r score games pairs elo lo hi llr dec < <(uv run python scripts/match_verdict.py "$f" --elo0=0 --elo1=15)
  log "$(basename $arm) vs cand16a: $score over $games games, $elo Elo [$lo, $hi], SPRT $dec"
  if [[ "$dec" == H1 ]] && awk -v e="$elo" -v b="$BEST_ELO" 'BEGIN { exit !(e > b) }'; then
    BEST=$arm BEST_ELO=$elo
  fi
done

case "$(basename $BEST)" in
  cand16b) echo 1.3 > $RL/policy_temp ;;
  cand16c) echo 0.8 > $RL/soft_mix ;;
esac
rm -rf $RL/models/cand16
cp -r $BEST $RL/models/cand16
log "cand16 = $(basename $BEST); policy temp $(cat $RL/policy_temp 2>/dev/null || echo 1.0), soft mix $(cat $RL/soft_mix 2>/dev/null || echo 0.5) from gen17; resuming the loop"
exec make loop
