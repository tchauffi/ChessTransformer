#!/usr/bin/env bash
# The Stockfish ladder: place one network on Stockfish 16's UCI_Elo scale (UCI_LimitStrength,
# calibrated to the CCRL Blitz list) and count its blunders. At each level, 16 openings x 2
# colours: our engine (ct-bot, int8 ONNX on the CPU) at 800 visits a move against Stockfish at
# 0.5 s a move. A full-strength Stockfish then scores every move of those games
# (scripts/blunder_report.py): blunders and mistakes per game for both sides, the men on the
# board when they happen, and how many of our losses hold one of our blunders.
#
#   scripts/sf_ladder.sh <label> <model.int8.onnx> <level>...
#   scripts/sf_ladder.sh scratch250k /path/to/model.int8.onnx 1600 1900 2200 2500
#
# Export a checkpoint first:
#   .venv/bin/python scripts/export_onnx.py <model dir> -o <dir>/model.onnx --quantize
# Environment: NODES (800), SF_TIME (0.5), OPENINGS (16), WORKERS (5), BLUNDER_NODES (200000,
# 0 = no blunder pass), SYZYGY (table dirs: a game ends at its first position they settle, for
# both sides, so a won endgame counts as won; default none), OUT (logs/scratch/anchor).
# Games append to $OUT/<label>_vs_sf<level>.jsonl; use a new label for a new run.
# CPU-heavy: run it while the lichess bot is off (19:00-08:00), or the bot loses search.
set -euo pipefail
cd "$(dirname "$0")/.."
(( $# >= 3 )) || { sed -n '2,18p' "$0"; exit 2; }
label=$1 model=$2
shift 2
OUT=${OUT:-logs/scratch/anchor}
mkdir -p "$OUT"
for elo in "$@"; do
  base="$OUT/${label}_vs_sf${elo}"
  nice -n 5 .venv/bin/python scripts/head_to_head.py --bot rust/target/release/ct-bot \
    --model-a "$model" --label-a "$label" --nodes-a "${NODES:-800}" \
    --stockfish-b /usr/games/stockfish --sf-elo "$elo" --sf-time "${SF_TIME:-0.5}" \
    --openings "${OPENINGS:-16}" --workers "${WORKERS:-5}" \
    ${SYZYGY:+--adjudicate-syzygy "$SYZYGY"} --blunder-nodes "${BLUNDER_NODES:-200000}" \
    --games-out "$base.jsonl" > "$base.log" 2>&1
  echo "[$(date '+%F %T')] $(grep -E ' score [0-9.]+$' "$base.log" | tail -1)"
  sed -n '/^Blunders:/,$p' "$base.log" | sed -n '2,3p'  # our side (A)
done
