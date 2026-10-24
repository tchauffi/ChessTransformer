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
#    A datacenter GPU outruns one Python driver, so SELFPLAY_WORKERS processes (read from
#    $RL/selfplay_workers, default 1) share the generation, spread over SELFPLAY_GPUS
#    ($RL/selfplay_gpus, e.g. "0,1") when set, with PARALLEL games in flight between them
#    ($RL/selfplay_parallel, default 128).
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
#    Games start from BOOK's openings (book2k: 2000 positions 8 plies deep) on BOOK_SHARE of
#    them, past the first BOOK_SKIP that the gates and anchor matches play, and from the 82
#    shallow built-in openings on the rest, so the first moves keep self-play data; up to
#    gen14 every game started from those 82. FRESH_FULL=1 ($RL/fresh_full) gives full
#    searches a fresh tree: a reused subtree's warm visits, spent at depth 1 without root
#    noise or forced playouts, were a third of gen14's policy targets on average.
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
#    visit counts; unset, it is read from $RL/policy_target (default visits). POLICY_TEMP=T
#    ($RL/policy_temp, default 1) trains on visits^(1/T): pruned 800-sim targets carry 0.91
#    nats against gen14's own 1.34, and T=1.25 brings them to 1.12.
# 3. Gate. Candidate vs champion, up to GATE_GAMES games ($RL/gate_games, default 400), with
#    an SPRT of H0 +0 vs H1 +GATE_ELO1 Elo (15) that stops the match once it decides. The
#    candidate passes only when the SPRT accepts H1 (PROMOTE_RULE=score restores the rule up to
#    gen14, a score >= 50%): that rule promoted noise -- gen04 at +9 (not significant), gen10
#    at +2, gen12 at a paired -9 -- and nine such steps chained to +168 over gen00 while gen14
#    beat gen00 by +38 head to head. A candidate that passes then plays every anchor in
#    ANCHORS ("label=model_dir", default v2.1=$RL/anchors/v2.1), ANCHOR_GAMES games each
#    ($RL/anchor_games, default 400), and is promoted only if its Elo against each is within
#    ANCHOR_MARGIN/2 (10) of the champion's ($RL/anchors.tsv; measured first when missing):
#    beating its parent, gen14 had lost 60 Elo against v2.1 relative to gen00 (-117 vs -58).
#    FORCE_PROMOTE=1 promotes whatever the result (history reads "forced"): an architecture
#    change taken on its ceiling, not on its first gate. Matches run on scripts/gpu_match.py:
#    every side plays the deployed lichess bot's own search code (chess_core::search) and move
#    flow (tree reuse, 4000 sims/move clipped by warm visits, 2000-sim ponder), with every
#    game's waves batched into one GPU forward per network, on the book's first openings.
# 4. Back up. scripts/hf_sync.py uploads the champions and the finished generations' games
#    to the private Hugging Face repos, in the background so the next generation does not
#    wait on the network; a failed upload is caught up by the next run (HF_SYNC=0: off).
#
# Every stage leaves a marker, so rerunning after a crash resumes the same generation.
# State lives in data/rl/: champion (symlink), models/genNN, models/candNN,
# selfplay/genNN, gates/*.jsonl and history.tsv.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# awk prints numbers in the locale's format: -77,5 under fr_FR, which the scripts it feeds reject.
export LC_ALL=C

RL="${RL:-data/rl}"
SELFPLAY_HOURS="${SELFPLAY_HOURS:-$(cat "$RL/selfplay_hours" 2>/dev/null || echo 8)}"
SIMS="${SIMS:-800}"
FAST_SIMS="${FAST_SIMS:-160}"
FULL_FRAC="${FULL_FRAC:-0.25}"
CAP_TOTAL="${CAP_TOTAL:-all}"
PARALLEL="${PARALLEL:-$(cat "$RL/selfplay_parallel" 2>/dev/null || echo 128)}"
SELFPLAY_WORKERS="${SELFPLAY_WORKERS:-$(cat "$RL/selfplay_workers" 2>/dev/null || echo 1)}"
SELFPLAY_GPUS="${SELFPLAY_GPUS:-$(cat "$RL/selfplay_gpus" 2>/dev/null)}"
# Search threads per self-play worker, and the gate's games in flight and threads: the home
# box keeps 6 / 128 / 4 (the lichess bot shares its CPU); a big machine wants more of each.
SELFPLAY_THREADS="${SELFPLAY_THREADS:-$(cat "$RL/selfplay_threads" 2>/dev/null || echo 6)}"
MATCH_PARALLEL="${MATCH_PARALLEL:-$(cat "$RL/match_parallel" 2>/dev/null || echo 128)}"
MATCH_THREADS="${MATCH_THREADS:-$(cat "$RL/match_threads" 2>/dev/null || echo 4)}"
WINDOW="${WINDOW:-2}"
EPOCHS="${EPOCHS:-1}"
FRESH_ONLY="${FRESH_ONLY:-1}"
GATE_GAMES="${GATE_GAMES:-$(cat "$RL/gate_games" 2>/dev/null || echo 400)}"
GATE_ELO1="${GATE_ELO1:-15}"
PROMOTE_RULE="${PROMOTE_RULE:-sprt}"
# Anchors, "label=model_dir" each (empty: none), and their matches' games and margin.
ANCHORS="${ANCHORS-v2.1=$RL/anchors/v2.1}"
ANCHOR_GAMES="${ANCHOR_GAMES:-$(cat "$RL/anchor_games" 2>/dev/null || echo 400)}"
ANCHOR_MARGIN="${ANCHOR_MARGIN:-20}"
FORCE_PROMOTE="${FORCE_PROMOTE:-0}"
HF_SYNC="${HF_SYNC:-1}"
MATCH_SIMS="${MATCH_SIMS:-4000}"
PONDER_SIMS="${PONDER_SIMS:-2000}"
HUMAN_SHARDS="${HUMAN_SHARDS:-data/shards/elite_k32_v2}"
FORCED_PLAYOUTS="${FORCED_PLAYOUTS:-$(cat "$RL/forced_playouts" 2>/dev/null || echo 0)}"
KATAGO_ON_FAIL="${KATAGO_ON_FAIL:-0}"
POLICY_TARGET="${POLICY_TARGET:-$(cat "$RL/policy_target" 2>/dev/null || echo visits)}"
POLICY_TEMP="${POLICY_TEMP:-$(cat "$RL/policy_temp" 2>/dev/null || echo 1.0)}"
FRESH_FULL="${FRESH_FULL:-$(cat "$RL/fresh_full" 2>/dev/null || echo 0)}"
# Self-play openings: BOOK's past its first BOOK_SKIP (the gates' and anchors'), on BOOK_SHARE
# of the games; BOOK= (empty) keeps the shallow built-in ones only.
BOOK="${BOOK-data/openings/book2k.json}"
BOOK_SKIP="${BOOK_SKIP:-500}"
BOOK_SHARE="${BOOK_SHARE:-0.75}"
if [[ -n "$BOOK" ]] && (( GATE_GAMES > 2 * BOOK_SKIP || ANCHOR_GAMES > 2 * BOOK_SKIP )); then
  echo "GATE_GAMES / ANCHOR_GAMES above 2 x BOOK_SKIP ($BOOK_SKIP) would match on openings self-play trains on"
  exit 1
fi

log() { echo "[$(date '+%F %T')] $*"; }

# verdict <games jsonl> <elo0> <elo1>: "score games pairs elo lo hi llr decision" for side A
# (scripts/match_verdict.py; the report goes to the log).
verdict() {
  uv run python scripts/match_verdict.py "$1" --elo0="$2" --elo1="$3"
}

# match <model A> <model B> <label A> <label B> <games out> <games> [gpu_match args...]
# Book openings 0..games/2-1, both colours; <games out>.done marks a finished match. Retried
# once uncompiled with one engine half, in case two compiled nets do not fit the GPU or the
# overlapped path fails its run-time check.
match() {
  local a=$1 b=$2 la=$3 lb=$4 out=$5 n=$6 extra
  shift 6
  for extra in "" "--no-compile --engines 1"; do
    rm -f "$out" "$out.done"
    uv run python scripts/gpu_match.py --model-a "$a" --model-b "$b" --label-a "$la" \
      --label-b "$lb" --openings $((n / 2)) --max-sims "$MATCH_SIMS" \
      --ponder-max-sims "$PONDER_SIMS" --parallel-games "$MATCH_PARALLEL" --threads "$MATCH_THREADS" \
      --games-out "$out" "$@" $extra | tail -14 && { touch "$out.done"; return 0; }
    log "match failed${extra:+ ($extra)}"
  done
  return 1
}

# anchors.tsv: a network's paired Elo against each anchor (gen anchor elo lo hi games pairs).
anchor_elo() {  # <gen> <anchor>: its Elo, or nothing
  awk -F'\t' -v g="$1" -v a="$2" '$1 == g && $2 == a { e = $3 } END { if (e != "") print e }' \
    "$RL/anchors.tsv" 2>/dev/null
}
record_anchor() {  # <gen> <anchor> <games jsonl>
  local score games pairs elo lo hi _
  read -r score games pairs elo lo hi _ < <(verdict "$3" 0 15 2>/dev/null)
  [[ -f "$RL/anchors.tsv" ]] || printf "gen\tanchor\telo\tlo\thi\tgames\tpairs\n" > "$RL/anchors.tsv"
  awk -F'\t' -v g="$1" -v a="$2" '!($1 == g && $2 == a)' "$RL/anchors.tsv" > "$RL/anchors.tsv.tmp"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$elo" "$lo" "$hi" "$games" "$pairs" >> "$RL/anchors.tsv.tmp"
  mv "$RL/anchors.tsv.tmp" "$RL/anchors.tsv"
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
  log "self-play: ${SELFPLAY_HOURS} h, ${SIMS}/${FAST_SIMS} sims, ${FULL_FRAC} full, cap-total $CAP_TOTAL, forced playouts $FORCED_PLAYOUTS, fresh full trees $FRESH_FULL, book ${BOOK:-none}, $PARALLEL games in flight, $SELFPLAY_WORKERS worker(s)${SELFPLAY_GPUS:+ on GPUs $SELFPLAY_GPUS}"
  selfplay() {
    uv run python scripts/selfplay_rust.py --model "$CHAMP" --out "$SP" \
      --games 1000000 --max-hours "$SELFPLAY_HOURS" --parallel-games "$PARALLEL" \
      --sims "$SIMS" --fast-sims "$FAST_SIMS" --full-frac "$FULL_FRAC" --cap-total "$CAP_TOTAL" \
      --dirichlet-alpha 0.3 --dirichlet-eps 0.25 --forced-playouts "$FORCED_PLAYOUTS" \
      --seed "$G" --workers "$SELFPLAY_WORKERS" --threads "$SELFPLAY_THREADS" \
      ${SELFPLAY_GPUS:+--gpus "$SELFPLAY_GPUS"} \
      ${BOOK:+--book "$BOOK" --book-skip "$BOOK_SKIP" --book-share "$BOOK_SHARE"} \
      $([[ "$FRESH_FULL" == 1 ]] && echo --fresh-full) "$@"
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
  log "training $CAND from $CHAMP_NAME on ${DATA[*]}, $EPOCHS epochs, $POLICY_TARGET policy targets (temperature $POLICY_TEMP)"
  uv run python scripts/train_expert_iter.py --data "${DATA[@]}" --base "$CHAMP" --out "$CAND" \
    --epochs "$EPOCHS" --batch-size 512 --lr 1e-4 --kl-coef 0.1 --soft-mix 0.5 --value-weight 1.0 \
    --human-shards "$HUMAN_SHARDS" --human-frac 0.2 --select last --compile \
    --policy-target "$POLICY_TARGET" --policy-temp "$POLICY_TEMP" \
    || { log "training failed"; exit 1; }
  uv run --with onnxscript python scripts/export_onnx.py "$CAND" -o "$CAND/model.onnx" \
    || { log "export failed"; exit 1; }
  uv run python scripts/check_onnx_parity.py --model "$CAND" --onnx "$CAND/model.onnx" \
    || { log "ONNX parity failed -- not gating a graph that disagrees with torch"; exit 1; }
fi

# ── 3. gate vs champion, then the anchors; promote ──────────────────────────
CANDN="cand$(printf %02d $G)"
GATE="$RL/gates/${CANDN}_vs_${CHAMP_NAME}.jsonl"
if [[ ! -f "$SP/.gate_done" ]]; then
  log "gating $CAND vs $CHAMP_NAME, up to $GATE_GAMES games (SPRT +0 / +$GATE_ELO1 Elo)"
  match "$CAND" "$CHAMP" "$CANDN" "$CHAMP_NAME" "$GATE" "$GATE_GAMES" \
    --sprt --elo0 0 --elo1 "$GATE_ELO1" || { log "gate failed"; exit 1; }
  touch "$SP/.gate_done"
fi
read -r CSCORE CGAMES _ CELO CLO CHI _ CDEC < <(verdict "$GATE" 0 "$GATE_ELO1")
log "cand vs champion: $CSCORE over $CGAMES games, $CELO Elo [$CLO, $CHI], SPRT $CDEC"

PROMOTED=no
case "$PROMOTE_RULE" in
  sprt) [[ "$CDEC" == H1 ]] && PROMOTED=yes ;;
  score) awk -v s="$CSCORE" 'BEGIN { exit !(s >= 0.5) }' && PROMOTED=yes ;;
  *) log "PROMOTE_RULE must be sprt or score, not $PROMOTE_RULE"; exit 1 ;;
esac
[[ "$PROMOTED" == no ]] && log "gate: not better than $CHAMP_NAME ($PROMOTE_RULE rule)"

# A candidate that beat its parent must also hold its ground against every anchor: its Elo
# against the anchor no further below the champion's than ANCHOR_MARGIN/2 (the SPRT's LLR for
# H1 = the champion's Elo vs H0 = ANCHOR_MARGIN below it is positive). Anchor matches play a
# fixed ANCHOR_GAMES, so the Elo they record is the next comparison's reference; a champion
# without one (gen14 when the anchors came in, a forced promotion) is measured first.
ANCHOR_SCORE="-"
if [[ "$PROMOTED" == yes && -n "$ANCHORS" ]]; then
  for entry in $ANCHORS; do
    label="${entry%%=*}" path="${entry#*=}"
    [[ -f "$path/model.safetensors" ]] || { log "anchor $label: no network at $path"; exit 1; }
    if [[ -z "$(anchor_elo "$CHAMP_NAME" "$label")" ]]; then
      f="$RL/gates/${CHAMP_NAME}_vs_${label}.jsonl"
      if [[ ! -f "$f.done" ]]; then
        log "anchor $label: measuring $CHAMP_NAME first, $ANCHOR_GAMES games"
        match "$CHAMP" "$path" "$CHAMP_NAME" "$label" "$f" "$ANCHOR_GAMES" \
          || { log "anchor match failed"; exit 1; }
      fi
      record_anchor "$CHAMP_NAME" "$label" "$f"
    fi
    REF="$(anchor_elo "$CHAMP_NAME" "$label")"
    f="$RL/gates/${CANDN}_vs_${label}.jsonl"
    if [[ ! -f "$f.done" ]]; then
      log "anchor $label: $CANDN, $ANCHOR_GAMES games ($CHAMP_NAME: $REF Elo)"
      match "$CAND" "$path" "$CANDN" "$label" "$f" "$ANCHOR_GAMES" || { log "anchor match failed"; exit 1; }
    fi
    record_anchor "$CANDN" "$label" "$f"
    FLOOR="$(awk -v r="$REF" -v m="$ANCHOR_MARGIN" 'BEGIN { print r - m }')"
    read -r ASCORE _ _ AELO ALO AHI ALLR _ < <(verdict "$f" "$FLOOR" "$REF")
    [[ "$ANCHOR_SCORE" == "-" ]] && ANCHOR_SCORE="$ASCORE"
    if awk -v l="$ALLR" 'BEGIN { exit !(l > 0) }'; then
      log "anchor $label: $CANDN $AELO Elo [$ALO, $AHI] vs $CHAMP_NAME's $REF: holds"
    else
      log "anchor $label: $CANDN $AELO Elo [$ALO, $AHI] vs $CHAMP_NAME's $REF: lost ground"
      PROMOTED=no
    fi
  done
fi
[[ "$PROMOTED" == no && "$FORCE_PROMOTE" == 1 ]] && PROMOTED=forced

if [[ "$PROMOTED" != no ]]; then
  NEW="$RL/models/$GEN"
  rm -rf "$NEW"
  cp -r "$CAND" "$NEW"
  ln -sfn "models/$GEN" "$RL/champion"
  # The new champion's anchor Elo is its candidate's (none if it was forced without them).
  for entry in $ANCHORS; do
    f="$RL/gates/${CANDN}_vs_${entry%%=*}.jsonl"
    [[ -f "$f.done" ]] && record_anchor "$GEN" "${entry%%=*}" "$f"
  done
  log "promoted ($PROMOTED): champion is now $GEN"
else
  log "not promoted: $CHAMP_NAME stays champion"
  if [[ "$KATAGO_ON_FAIL" == 1 && ! -f "$RL/forced_playouts" ]]; then
    echo 2 > "$RL/forced_playouts"
    log "self-play gets KataGo forced playouts (k=2) from the next generation on"
  fi
fi

POS="$(uv run python -c "import numpy as np, glob, sys; print(sum(len(np.load(f)['z']) for f in glob.glob(sys.argv[1] + '/positions_*.npz')))" "$SP")"
NGAMES="$(wc -l < "$SP/games.jsonl")"
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$GEN" "$POS" "$NGAMES" "$CSCORE" "$PROMOTED" "$ANCHOR_SCORE" >> "$RL/history.tsv"
touch "$SP/.generation_done"
log "=== $GEN done ==="

# ── 4. back up to Hugging Face ────────────────────────────────────────────────
if [[ "$HF_SYNC" == 1 ]]; then
  HF_HUB_DISABLE_PROGRESS_BARS=1 setsid nohup uv run python scripts/hf_sync.py \
    >> logs/hf_sync.log 2>&1 < /dev/null &
  log "backing up to Hugging Face in the background (logs/hf_sync.log)"
fi
