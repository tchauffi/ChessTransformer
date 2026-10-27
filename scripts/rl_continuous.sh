#!/usr/bin/env bash
# Continuous expert iteration, after Braun 2026, "Engineering Efficient Self-Play Chess:
# Search, Replay, and Throughput Under Limited Compute" (arXiv 2609.37447). No generations:
#
#  * self-play runs without pause on the latest published network and loads each new one in
#    place (selfplay_rust.py --reload-from);
#  * the trainer follows self-play's output on a replay window of the last WINDOW_ROWS
#    positions, trains REUSE presentations per new position (4 in the paper) with one
#    optimizer throughout, and publishes its EMA weights every block (train_continuous.py);
#  * every EVAL_HOURS of running time both stop and the latest publication plays the last
#    accepted network at the bot's settings, EVAL_GAMES on paired book openings. A score of at
#    least ACCEPT (0.48, the paper's bar) makes it the accepted network; two misses in a row
#    send the trainer back to the accepted weights with a fresh optimizer. Every ANCHOR_EVERY
#    acceptances the accepted network plays v2.1 (ANCHOR_GAMES), the absolute line.
#
# GPUs (one machine, ids as nvidia-smi numbers them). With more than one, the trainer takes
# TRAIN_GPU ($RL/cont_train_gpu, default the first) to itself and self-play WORKERS_PER_GPU
# processes ($RL/cont_workers_per_gpu, default 1) on each of SELFPLAY_GPUS ($RL/cont_selfplay_gpus,
# default the others), PARALLEL games in flight per GPU ($RL/cont_parallel, default 256). Nothing
# then stops for an evaluation: its matches run on EVAL_GPU ($RL/cont_eval_gpu, default the last
# self-play GPU) beside that GPU's self-play, the trainer keeps training, and only a rollback
# restarts it. With one GPU, or the trainer's among self-play's, all share it and evaluations
# pause both, as above.
#
# This replaces the generational loop's fine-tune of each candidate on one or two
# generations, seen once with a fresh optimizer, behind a strict SPRT gate: where the paper
# trained 836M presentations on 3.25M games in 2.5 days, the loop had trained ~100M on 463k
# games in two weeks. Self-play settings are the loop's ($RL/sims, c_puct, ...).
#
#   RL=data/rl scripts/rl_continuous.sh       # until $RL/STOP exists (`make stop`)
#
# State in $RL/continuous: selfplay/ (games + shards), trainer/ (checkpoint), published/
# (step_NNNNNNN/, latest), best/ (the accepted network), evals/, history.tsv, anchors.tsv.
# It starts from $RL/champion, with the last generation's games in the replay window.
# Logs: logs/rl_continuous.log (this script), _selfplay.log, _train.log.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export LC_ALL=C

RL="${RL:-data/rl}"
C="${CONT:-$RL/continuous}"
LIVE=$C/selfplay PUB=$C/published BEST=$C/best
PY="${PY:-.venv/bin/python}"  # not `uv run`: the children must get this script's signals
setting() { local v; v="$(cat "$RL/$1" 2>/dev/null)"; echo "${v:-$2}"; }
SIMS="${SIMS:-$(setting sims 800)}"
FAST_SIMS="${FAST_SIMS:-160}"
FULL_FRAC="${FULL_FRAC:-$(setting full_frac 0.25)}"
C_PUCT="${C_PUCT:-$(setting c_puct 1.0)}"
FORCED_PLAYOUTS="${FORCED_PLAYOUTS:-$(setting forced_playouts 0)}"
DRAW_CONTEMPT="${DRAW_CONTEMPT:-$(setting draw_contempt 0.2)}"
TEMP_CUTOFF="${TEMP_CUTOFF:-$(setting temp_cutoff 0.05)}"
SIM_BATCH="${SIM_BATCH:-$(setting sim_batch 1)}"
PARALLEL="${PARALLEL:-$(setting cont_parallel 256)}"  # per self-play GPU
THREADS="${THREADS:-$(setting selfplay_threads 6)}"  # per self-play process
POLICY_TEMP="${POLICY_TEMP:-$(setting policy_temp 1.0)}"
SOFT_MIX="${SOFT_MIX:-$(setting soft_mix 0.5)}"
WINDOW_ROWS="${WINDOW_ROWS:-$(setting cont_window_rows 10000000)}"
REUSE="${REUSE:-$(setting cont_reuse 4)}"
LR="${LR:-$(setting cont_lr 5e-5)}"
EVAL_HOURS="${EVAL_HOURS:-$(setting cont_eval_hours 4)}"
EVAL_GAMES="${EVAL_GAMES:-$(setting cont_eval_games 200)}"
ACCEPT="${ACCEPT:-0.48}"
ANCHOR="${ANCHOR-$RL/anchors/v2.1}"
ANCHOR_EVERY="${ANCHOR_EVERY:-3}"
ANCHOR_GAMES="${ANCHOR_GAMES:-200}"
MATCH_SIMS="${MATCH_SIMS:-4000}"
PONDER_SIMS="${PONDER_SIMS:-2000}"
MATCH_PARALLEL="${MATCH_PARALLEL:-$(setting match_parallel 128)}"
MATCH_THREADS="${MATCH_THREADS:-$(setting match_threads 4)}"
HUMAN_SHARDS="${HUMAN_SHARDS-data/shards/elite_k32_v2}"
HUMAN_FRAC="${HUMAN_FRAC:-0.2}"
BOOK="${BOOK-data/openings/book2k.json}"
BOOK_SKIP="${BOOK_SKIP:-500}"
BOOK_SHARE="${BOOK_SHARE:-0.75}"
SEED_DATA="${SEED_DATA-$(ls -d "$RL"/selfplay/gen[0-9][0-9] 2>/dev/null | tail -1)}"
COMPILE="${COMPILE---compile}"
SP_EXTRA="${SP_EXTRA:-}" TR_EXTRA="${TR_EXTRA:-}" MATCH_EXTRA="${MATCH_EXTRA:-}"  # dry runs
LOGS="${LOGS:-logs}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # CUDA's ids = nvidia-smi's
if [[ -n ${CUDA_VISIBLE_DEVICES+x} ]]; then
  IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  read -ra GPUS <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
fi
default=""
(( ${#GPUS[@]} > 1 )) && default=${GPUS[0]}
TRAIN_GPU="${TRAIN_GPU-$(setting cont_train_gpu "$default")}"
rest=()
for g in "${GPUS[@]}"; do [[ $g == "$TRAIN_GPU" ]] || rest+=("$g"); done
default=""
[[ -n $TRAIN_GPU ]] && default="$(IFS=,; echo "${rest[*]}")"
SELFPLAY_GPUS="${SELFPLAY_GPUS-$(setting cont_selfplay_gpus "$default")}"
IFS=, read -ra SP_GPUS <<< "$SELFPLAY_GPUS"
NONSTOP=0  # the trainer has a GPU of its own: evaluations run beside training and self-play
[[ -n $TRAIN_GPU && -n $SELFPLAY_GPUS && ",$SELFPLAY_GPUS," != *",$TRAIN_GPU,"* ]] && NONSTOP=1
default=$TRAIN_GPU
(( NONSTOP )) && default=${SP_GPUS[-1]}
EVAL_GPU="${EVAL_GPU-$(setting cont_eval_gpu "$default")}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-$(setting cont_workers_per_gpu 1)}"
WORKERS=$((WORKERS_PER_GPU * (${#SP_GPUS[@]} > 0 ? ${#SP_GPUS[@]} : 1)))
# Children start as `env ${GPU:+CUDA_VISIBLE_DEVICES=$GPU} "$PY" ...`: env execs Python, so $!
# stays the process the signals must reach.

if [[ -n "$BOOK" ]] && (( EVAL_GAMES > 2 * BOOK_SKIP || ANCHOR_GAMES > 2 * BOOK_SKIP )); then
  echo "EVAL_GAMES / ANCHOR_GAMES above 2 x BOOK_SKIP ($BOOK_SKIP) would match on openings self-play trains on"
  exit 1
fi

log() { echo "[$(date '+%F %T')] $*"; }

# match <model A> <model B> <label A> <label B> <games out> <games>: the gate's match (the bot's
# search and move flow, both colours of the book's first openings), retried uncompiled once.
match() {
  local a=$1 b=$2 la=$3 lb=$4 out=$5 n=$6 extra
  [[ -f $out.done ]] && return 0
  for extra in "" "--no-compile --engines 1"; do
    rm -f "$out"
    env ${EVAL_GPU:+CUDA_VISIBLE_DEVICES=$EVAL_GPU} "$PY" scripts/gpu_match.py --model-a "$a" --model-b "$b" \
      --label-a "$la" --label-b "$lb" --openings $((n / 2)) --max-sims "$MATCH_SIMS" \
      --ponder-max-sims "$PONDER_SIMS" --parallel-games "$MATCH_PARALLEL" --threads "$MATCH_THREADS" \
      --games-out "$out" $MATCH_EXTRA $extra | tail -14 && { touch "$out.done"; return 0; }
    log "match failed${extra:+ ($extra)}"
  done
  return 1
}

ACTOR="" TRAINER=""
start_trainer() {
  local rb=()
  [[ -f $C/ROLLBACK ]] && rb=(--rollback-to "$BEST")
  env ${TRAIN_GPU:+CUDA_VISIBLE_DEVICES=$TRAIN_GPU} "$PY" scripts/train_continuous.py --live "$LIVE" \
    ${SEED_DATA:+--seed-data $SEED_DATA} --base "$BEST" --state "$C/trainer" --publish "$PUB" \
    --window-rows "$WINDOW_ROWS" --reuse "$REUSE" --lr "$LR" --soft-mix "$SOFT_MIX" \
    --policy-temp "$POLICY_TEMP" "${rb[@]}" \
    ${HUMAN_SHARDS:+--human-shards "$HUMAN_SHARDS" --human-frac "$HUMAN_FRAC"} $COMPILE $TR_EXTRA \
    >> "$LOGS/rl_continuous_train.log" 2>&1 &
  TRAINER=$!
  ((${#rb[@]})) && { rm -f "$C/ROLLBACK"; log "trainer rolled back to the accepted network"; }
}
start_actor() {  # each worker gets one GPU of --gpus (selfplay_rust.py's run_workers)
  env ${SELFPLAY_GPUS:+CUDA_VISIBLE_DEVICES=$SELFPLAY_GPUS} "$PY" scripts/selfplay_rust.py \
    --model "$PUB/latest" --reload-from "$PUB/latest" --out "$LIVE" --games 100000000 --max-hours 0 \
    --parallel-games $((PARALLEL * WORKERS / WORKERS_PER_GPU)) --workers "$WORKERS" \
    ${SELFPLAY_GPUS:+--gpus "$SELFPLAY_GPUS"} --threads "$THREADS" \
    --sims "$SIMS" --fast-sims "$FAST_SIMS" --full-frac "$FULL_FRAC" --cap-total all \
    --dirichlet-alpha 0.3 --dirichlet-eps 0.25 --forced-playouts "$FORCED_PLAYOUTS" \
    --draw-contempt "$DRAW_CONTEMPT" --temp-cutoff "$TEMP_CUTOFF" --sim-batch "$SIM_BATCH" \
    --c-puct "$C_PUCT" --fresh-full --seed 1 \
    ${BOOK:+--book "$BOOK" --book-skip "$BOOK_SKIP" --book-share "$BOOK_SHARE"} $SP_EXTRA \
    >> "$LOGS/rl_continuous_selfplay.log" 2>&1 &
  ACTOR=$!
}
stop() {  # SIGTERM: self-play flushes its games, the trainer saves its checkpoint
  local p
  for p in "$@"; do kill -TERM "$p" 2>/dev/null; done
  for p in "$@"; do wait "$p" 2>/dev/null; done
}
stop_all() { stop $ACTOR $TRAINER; ACTOR="" TRAINER=""; }
stop_trainer() { stop $TRAINER; TRAINER=""; }
trap 'log "terminated: stopping self-play and the trainer"; stop_all; exit 143' TERM INT

# wait_running <seconds>: running time (a gap over 10 min is the night freeze), 0 at the end or
# on STOP, 1 when a child exits.
wait_running() {
  local need=$1 active=0 last now
  last=$(date +%s)
  while (( active < need )); do
    sleep 20
    now=$(date +%s)
    (( now - last < 600 )) && active=$((active + now - last))
    last=$now
    [[ -e $RL/STOP ]] && return 0
    kill -0 "$ACTOR" 2>/dev/null || { log "self-play exited (see $LOGS/rl_continuous_selfplay.log)"; return 1; }
    kill -0 "$TRAINER" 2>/dev/null || { log "trainer exited (see $LOGS/rl_continuous_train.log)"; return 1; }
  done
}

# evaluate: the latest publication against the accepted network; accept, or count a miss.
evaluate() {
  local cand name f score games pairs elo lo hi llr dec decision
  cand="$(readlink -f "$PUB/latest")"
  name="$(basename "$cand")"
  if [[ "$(cat "$C/last_eval" 2>/dev/null)" == "$name" ]]; then
    log "no new publication since $name"
    return 0
  fi
  rm -rf "$C/candidate"
  cp -r "$cand" "$C/candidate"  # the trainer prunes old publications
  f=$C/evals/${name}_vs_best.jsonl
  log "evaluating $name against the accepted network ($(cat "$BEST/SOURCE")), $EVAL_GAMES games"
  match "$C/candidate" "$BEST" "$name" best "$f" "$EVAL_GAMES" || return 1
  read -r score games pairs elo lo hi llr dec < <("$PY" scripts/match_verdict.py "$f" --elo0=0 --elo1=15)
  if awk -v s="$score" -v a="$ACCEPT" 'BEGIN { exit !(s >= a) }'; then
    rm -rf "$BEST.new"
    cp -r "$C/candidate" "$BEST.new"
    echo "$name" > "$BEST.new/SOURCE"
    rm -rf "$BEST.old"
    mv "$BEST" "$BEST.old" && mv "$BEST.new" "$BEST" && rm -rf "$BEST.old"
    FAILS=0 ACCEPTED=$((ACCEPTED + 1)) decision=accepted
  else
    FAILS=$((FAILS + 1)) decision="missed ($FAILS in a row)"
    if (( FAILS >= 2 )); then
      touch "$C/ROLLBACK"
      FAILS=0 decision="missed twice: rollback to the accepted network"
    fi
  fi
  echo "$name" > "$C/last_eval"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$name" "$score" "$games" "$elo" "$lo" "$hi" \
    "$decision" >> "$C/history.tsv"
  printf "FAILS=%s ACCEPTED=%s\n" "$FAILS" "$ACCEPTED" > "$C/eval_state"
  log "$name vs accepted: $score over $games games, $elo Elo [$lo, $hi]: $decision"
  if [[ "$decision" == accepted && -n "$ANCHOR" ]] && (( ACCEPTED % ANCHOR_EVERY == 0 )); then
    f=$C/evals/${name}_vs_anchor.jsonl
    log "anchor: $name against $ANCHOR, $ANCHOR_GAMES games"
    match "$BEST" "$ANCHOR" "$name" anchor "$f" "$ANCHOR_GAMES" || return 1
    read -r score games pairs elo lo hi llr dec < <("$PY" scripts/match_verdict.py "$f" --elo0=0 --elo1=15)
    [[ -f $C/anchors.tsv ]] || printf "time\tnetwork\tscore\tgames\telo\tlo\thi\n" > "$C/anchors.tsv"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$name" "$score" "$games" "$elo" "$lo" "$hi" \
      >> "$C/anchors.tsv"
    log "anchor: $name $elo Elo [$lo, $hi] against $(basename "$ANCHOR")"
  fi
}

mkdir -p "$C" "$LIVE" "$C/evals" "$LOGS"
if [[ ! -f $BEST/model.safetensors ]]; then
  src="$RL/$(readlink "$RL/champion")"
  rm -rf "$BEST.new"
  cp -r "$src" "$BEST.new"
  basename "$src" > "$BEST.new/SOURCE"
  mv "$BEST.new" "$BEST"
  log "accepted network initialised from $src"
fi
[[ -f $C/history.tsv ]] || printf "time\tcandidate\tscore\tgames\telo\tlo\thi\tdecision\n" > "$C/history.tsv"
FAILS=0 ACCEPTED=0
[[ -f $C/eval_state ]] && source "$C/eval_state"
EVAL_S="$(awk -v h="$EVAL_HOURS" 'BEGIN { printf "%d", h * 3600 }')"
log "continuous loop: self-play ${SIMS}/${FAST_SIMS} sims, c_puct $C_PUCT, $PARALLEL games in flight per GPU; trainer window $WINDOW_ROWS rows, reuse $REUSE, lr $LR, soft mix $SOFT_MIX, policy temperature $POLICY_TEMP; evaluations every $EVAL_HOURS h, $EVAL_GAMES games, accept >= $ACCEPT"
if (( NONSTOP )); then
  log "GPUs: trainer $TRAIN_GPU; self-play $SELFPLAY_GPUS ($WORKERS worker(s)); evaluations on $EVAL_GPU beside self-play and training"
else
  log "GPUs: trainer ${TRAIN_GPU:-default}; self-play ${SELFPLAY_GPUS:-default} ($WORKERS worker(s)); evaluations ${EVAL_GPU:+on $EVAL_GPU }pause both"
fi
CRASHES=0
while [[ ! -e $RL/STOP ]]; do
  if [[ -z $TRAINER ]]; then
    start_trainer
    for _ in $(seq 360); do  # the first start loads the window and publishes the base network
      [[ -e $PUB/latest ]] && break
      kill -0 "$TRAINER" 2>/dev/null || break
      sleep 5
    done
    [[ -e $PUB/latest ]] || { log "no publication from the trainer"; stop_all; exit 1; }
  fi
  [[ -n $ACTOR ]] || start_actor
  if wait_running "$EVAL_S"; then
    CRASHES=0
    [[ -e $RL/STOP ]] && break
    (( NONSTOP )) || stop_all
    evaluate || { log "evaluation failed"; stop_all; exit 1; }
    [[ -f $C/ROLLBACK && -n $TRAINER ]] && stop_trainer  # restarts from the accepted network
  else
    stop_all
    CRASHES=$((CRASHES + 1))
    (( CRASHES >= 3 )) && { log "three crashes in a row: stopping"; exit 1; }
    sleep 60
  fi
done
stop_all
log "STOP found: continuous loop ended"
