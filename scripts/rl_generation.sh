#!/usr/bin/env bash
# One generation of expert iteration (AlphaZero-style RL) on top of the current champion.
#
#   scripts/rl_generation.sh --seed logs/pos2move_v2/run_028_20261010_081319/checkpoints/best_model
#   setsid nohup scripts/rl_generation.sh >> logs/rl_generation.log 2>&1 &
#
# 1. Self-play, on the deployed bot's search (selfplay-core SelfPlayEngine: the tree reused
#    move to move, 16-leaf waves), for SELFPLAY_HOURS of running time; unset, it is read from
#    $RL/selfplay_hours (default 8). The GPU is saturated, so more games per candidate means
#    longer generations: cand11, trained on one 8 h generation, got worse on held-out games and
#    lost (-31), while cand10 and cand12, on two, improved and tied.
#    The champion plays itself with playout cap randomization: FULL_FRAC of
#    moves are full SIMS searches with root Dirichlet noise -- the policy targets -- and
#    the rest fast FAST_SIMS moves that only move the game on. Every position keeps the
#    game outcome, so fast moves buy more independent value labels per GPU hour. The
#    earlier expert-iteration runs taught at 128 sims and were gated at 400: the teacher
#    must search at least as deep as the gate, hence 800 vs ct-bot's 1800 nodes with
#    tree reuse (~800 fresh). CAP_TOTAL=all counts the reused tree's warm visits toward
#    both budgets (fast: fast moves only; none: fresh sims on top of the reused tree, which
#    cost 2.7x the old engine's evaluations per position in gen02's first half hour).
#    FORCED_PLAYOUTS=k adds KataGo forced playouts + policy target pruning to the full
#    searches (k=2 in KataGo); unset, k is read from $RL/forced_playouts, which
#    KATAGO_ON_FAIL=1 writes (k=2) the first time a candidate fails its gate.
# 2. Train. A candidate from the champion, EPOCHS passes over the last WINDOW generations
#    of self-play: policy on the visit distributions, WDL value on the outcome mixed 50/50
#    with the root Q (less label variance -- the value head overfit last time), KL-anchored
#    to the champion, 20% human replay. gen01's single pass (4k steps) ended with the policy
#    loss still falling, hence 2 up to cand05. But the second pass memorises: cand04's train
#    loss dropped at the epoch boundary without the val loss following, and on unseen gen04
#    games its value loss was no better than gen03's (a game's positions share one outcome),
#    so 1 from cand06. A generation the champion already trained on is a second pass too:
#    every candidate that retrained on one tied or lost (cand04 +9, cand05 -44, cand07 -6),
#    those on unseen games only won (cand03 +45, cand06 +46), so FRESH_ONLY=1 (from cand09)
#    skips the generations up to the champion's own: one generation after a promotion,
#    WINDOW after a rejection. The held-out line after each epoch shows what a pass buys.
#    The final weights are exported: val loss does not pick checkpoints
#    (exit2 had the best val loss and the worst gate). Compiled: 2.2x the eager speed, and a
#    smolgen champion does not fit eager at batch 512 (>15.4 GiB; compiled 9.4).
#    POLICY_TARGET=gumbel trains the policy on Gumbel completed-Q targets instead of the
#    visit counts; unset, it is read from $RL/policy_target (default visits).
# 3. Gate. Candidate vs champion over GATE_GAMES games; promoted on a score >= 50%. A worse
#    candidate rarely clears that (a -30 Elo net passes ~1% of the time at 400 games) and an
#    equal one is a free reshuffle, as AlphaZero always promoted. FORCE_PROMOTE=1 promotes
#    it whatever the score (history reads "forced"): an architecture change taken on its
#    ceiling, not on its first gate. With V21_GAMES > 0 a new
#    champion then also plays prod v2.1 (off by default). Matches run on scripts/gpu_match.py: every
#    side plays the deployed lichess bot's own search code (chess_core::search) and move
#    flow (tree reuse, 4000 sims/move clipped by warm visits, 2000-sim ponder), with every
#    game's waves batched into one GPU forward per network. They stop early once the SPRT
#    decides; promotion still reads the score over the games played.
# 4. Back up. scripts/hf_sync.py uploads the champions and the finished generations' games
#    to the private Hugging Face repos, in the background so the next generation does not
#    wait on the network; a failed upload is caught up by the next run (HF_SYNC=0: off).
#
# Every stage leaves a marker, so rerunning after a crash resumes the same generation.
# State lives in data/rl/: champion (symlink), models/genNN, models/candNN,
# selfplay/genNN, gates/*.jsonl and history.tsv.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RL="${RL:-data/rl}"
SELFPLAY_HOURS="${SELFPLAY_HOURS:-$(cat "$RL/selfplay_hours" 2>/dev/null || echo 8)}"
SIMS="${SIMS:-800}"
FAST_SIMS="${FAST_SIMS:-160}"
FULL_FRAC="${FULL_FRAC:-0.25}"
CAP_TOTAL="${CAP_TOTAL:-all}"
PARALLEL="${PARALLEL:-128}"
WINDOW="${WINDOW:-2}"
EPOCHS="${EPOCHS:-1}"
FRESH_ONLY="${FRESH_ONLY:-1}"
GATE_GAMES="${GATE_GAMES:-400}"
V21_GAMES="${V21_GAMES:-0}"
FORCE_PROMOTE="${FORCE_PROMOTE:-0}"
HF_SYNC="${HF_SYNC:-1}"
MATCH_SIMS="${MATCH_SIMS:-4000}"
PONDER_SIMS="${PONDER_SIMS:-2000}"
V21=data/models/pos2move_v2.1
HUMAN_SHARDS="${HUMAN_SHARDS:-data/shards/elite_k32_v2}"
FORCED_PLAYOUTS="${FORCED_PLAYOUTS:-$(cat "$RL/forced_playouts" 2>/dev/null || echo 0)}"
KATAGO_ON_FAIL="${KATAGO_ON_FAIL:-0}"
POLICY_TARGET="${POLICY_TARGET:-$(cat "$RL/policy_target" 2>/dev/null || echo visits)}"

log() { echo "[$(date '+%F %T')] $*"; }

score() {  # per-game score of side A in a games jsonl, plus the report
  uv run python - "$1" <<'EOF'
import json, sys
sys.path.insert(0, "src")
from chesstransformer.evaluation.sprt import format_report, pair_up, sprt_llr, summarize
rows = [json.loads(l) for l in open(sys.argv[1])]
res = {(r["idx"], r["a_white"]): r["score_a"] for r in rows}
pairs = pair_up(res)
print(format_report(summarize(pairs, [res[k] for k in res if (k[0], not k[1]) in res]),
                    "A", "B", llr=sprt_llr(pairs)), file=sys.stderr)
print(f"{sum(r['score_a'] for r in rows) / max(1, len(rows)):.4f} {len(rows)}")
EOF
}

# gate <model A> <model B> <label A> <label B> <games out> <games>
# Up to <games>, stopping as soon as the SPRT (H0 +0 / H1 +15 Elo) decides. Retried once
# uncompiled with one engine half, in case two compiled nets do not fit the GPU or the
# overlapped path fails its run-time check.
gate() {
  local extra
  for extra in "" "--no-compile --engines 1"; do
    rm -f "$5"
    uv run python scripts/gpu_match.py --model-a "$1" --model-b "$2" --label-a "$3" \
      --label-b "$4" --openings $(($6 / 2)) --max-sims "$MATCH_SIMS" \
      --ponder-max-sims "$PONDER_SIMS" --sprt --games-out "$5" $extra | tail -14 && return 0
    log "match failed${extra:+ ($extra)}"
  done
  return 1
}

if [[ "${1:-}" == "--seed" ]]; then
  SRC="${2:?--seed <model_dir>}"
  [[ -e "$RL/champion" ]] && { echo "$RL/champion already exists; refusing to reseed"; exit 1; }
  mkdir -p "$RL/models/gen00" "$RL/selfplay" "$RL/gates"
  cp "$SRC/model.safetensors" "$SRC/model_config.json" "$RL/models/gen00/"
  [[ -f "$SRC/model.onnx" ]] && cp "$SRC/model.onnx" "$RL/models/gen00/"
  echo "$SRC" > "$RL/models/gen00/SOURCE"
  ln -sfn models/gen00 "$RL/champion"
  printf "gen\tpositions\tgames\tcand_vs_champ\tpromoted\tchamp_vs_v2.1\n" > "$RL/history.tsv"
  log "seeded $RL/champion from $SRC"
  exit 0
fi

[[ -e "$RL/champion" ]] || { echo "no $RL/champion -- seed first: $0 --seed <model_dir>"; exit 1; }
CHAMP="$RL/$(readlink "$RL/champion")"
CHAMP_NAME="$(basename "$CHAMP")"

# The generation in progress is the first self-play dir without a finished gate.
G=1
while [[ -f "$RL/selfplay/gen$(printf %02d $G)/.generation_done" ]]; do G=$((G + 1)); done
GEN="gen$(printf %02d $G)"
SP="$RL/selfplay/$GEN"
CAND="$RL/models/cand$(printf %02d $G)"
mkdir -p "$SP" "$RL/gates"
log "=== $GEN: champion $CHAMP_NAME ==="

# ── 1. self-play ─────────────────────────────────────────────────────────────
if [[ ! -f "$SP/.selfplay_done" ]]; then
  log "self-play: ${SELFPLAY_HOURS} h, ${SIMS}/${FAST_SIMS} sims, ${FULL_FRAC} full, cap-total $CAP_TOTAL, forced playouts $FORCED_PLAYOUTS"
  selfplay() {
    uv run python scripts/selfplay_rust.py --model "$CHAMP" --out "$SP" \
      --games 1000000 --max-hours "$SELFPLAY_HOURS" --parallel-games "$PARALLEL" \
      --sims "$SIMS" --fast-sims "$FAST_SIMS" --full-frac "$FULL_FRAC" --cap-total "$CAP_TOTAL" \
      --dirichlet-alpha 0.3 --dirichlet-eps 0.25 --forced-playouts "$FORCED_PLAYOUTS" \
      --seed "$G" "$@"
  }
  # The bot's search; should it crash, the older engine finishes the generation (it
  # resumes the games file, but restarts the hour budget) -- except with forced playouts,
  # which the older engine lacks: its games would be plain-MCTS data.
  if ! selfplay; then
    if awk -v k="$FORCED_PLAYOUTS" 'BEGIN { exit !(k > 0) }'; then
      log "self-play failed (no legacy fallback: it has no forced playouts)"; exit 1
    fi
    log "self-play failed -- rerunning on the legacy engine"
    selfplay --legacy-engine || { log "self-play failed"; exit 1; }
  fi
  touch "$SP/.selfplay_done"
fi

# ── 2. train ─────────────────────────────────────────────────────────────────
if [[ ! -f "$CAND/model.safetensors" ]]; then
  # Champion genNN was trained on generations up to NN (candNN's window ends at its own).
  SEEN=0
  [[ "$FRESH_ONLY" == 1 && "$CHAMP_NAME" =~ ^gen([0-9]+)$ ]] && SEEN=$((10#${BASH_REMATCH[1]}))
  DATA=()
  for ((k = G - WINDOW + 1; k <= G; k++)); do
    d="$RL/selfplay/gen$(printf %02d $k)"
    ((k >= 1 && k > SEEN)) && [[ -f "$d/.selfplay_done" ]] && DATA+=("$d")
  done
  log "training $CAND from $CHAMP_NAME on ${DATA[*]}, $EPOCHS epochs, $POLICY_TARGET policy targets"
  uv run python scripts/train_expert_iter.py --data "${DATA[@]}" --base "$CHAMP" --out "$CAND" \
    --epochs "$EPOCHS" --batch-size 512 --lr 1e-4 --kl-coef 0.1 --soft-mix 0.5 --value-weight 1.0 \
    --human-shards "$HUMAN_SHARDS" --human-frac 0.2 --select last --compile \
    --policy-target "$POLICY_TARGET" \
    || { log "training failed"; exit 1; }
  uv run --with onnxscript python scripts/export_onnx.py "$CAND" -o "$CAND/model.onnx" \
    || { log "export failed"; exit 1; }
  uv run python scripts/check_onnx_parity.py --model "$CAND" --onnx "$CAND/model.onnx" \
    || { log "ONNX parity failed -- not gating a graph that disagrees with torch"; exit 1; }
fi

# ── 3. gate vs champion, promote (match vs v2.1 if V21_GAMES > 0) ───────────
GATE="$RL/gates/cand$(printf %02d $G)_vs_${CHAMP_NAME}.jsonl"
if [[ ! -f "$SP/.gate_done" ]]; then
  log "gating $CAND vs $CHAMP_NAME, $GATE_GAMES games"
  gate "$CAND" "$CHAMP" "cand$(printf %02d $G)" "$CHAMP_NAME" "$GATE" "$GATE_GAMES" \
    || { log "gate failed"; exit 1; }
  touch "$SP/.gate_done"
fi
read -r CSCORE CGAMES < <(score "$GATE")
log "cand vs champion: $CSCORE over $CGAMES games"

PROMOTED=no
V21_SCORE="-"
if awk -v s="$CSCORE" 'BEGIN { exit !(s >= 0.5) }'; then
  PROMOTED=yes
elif [[ "$FORCE_PROMOTE" == 1 ]]; then
  PROMOTED=forced
fi
if [[ "$PROMOTED" != no ]]; then
  NEW="$RL/models/$GEN"
  rm -rf "$NEW"
  cp -r "$CAND" "$NEW"
  ln -sfn "models/$GEN" "$RL/champion"
  log "promoted ($PROMOTED): champion is now $GEN"
  if ((V21_GAMES > 0)); then
    VGATE="$RL/gates/${GEN}_vs_v2.1.jsonl"
    log "matching $GEN vs v2.1, $V21_GAMES games"
    gate "$NEW" "$V21" "$GEN" v2.1 "$VGATE" "$V21_GAMES" || log "v2.1 match failed (champion kept)"
    [[ -s "$VGATE" ]] && read -r V21_SCORE _ < <(score "$VGATE")
    log "$GEN vs v2.1: $V21_SCORE"
  fi
else
  log "not promoted: $CHAMP_NAME stays champion"
  if [[ "$KATAGO_ON_FAIL" == 1 && ! -f "$RL/forced_playouts" ]]; then
    echo 2 > "$RL/forced_playouts"
    log "self-play gets KataGo forced playouts (k=2) from the next generation on"
  fi
fi

POS="$(uv run python -c "import numpy as np, glob, sys; print(sum(len(np.load(f)['z']) for f in glob.glob(sys.argv[1] + '/positions_*.npz')))" "$SP")"
NGAMES="$(wc -l < "$SP/games.jsonl")"
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$GEN" "$POS" "$NGAMES" "$CSCORE" "$PROMOTED" "$V21_SCORE" >> "$RL/history.tsv"
touch "$SP/.generation_done"
log "=== $GEN done ==="

# ── 4. back up to Hugging Face ────────────────────────────────────────────────
if [[ "$HF_SYNC" == 1 ]]; then
  HF_HUB_DISABLE_PROGRESS_BARS=1 setsid nohup uv run python scripts/hf_sync.py \
    >> logs/hf_sync.log 2>&1 < /dev/null &
  log "backing up to Hugging Face in the background (logs/hf_sync.log)"
fi
