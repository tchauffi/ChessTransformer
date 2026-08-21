#!/usr/bin/env bash
# Wait for the in-flight trainer to finish, then audit prior visibility across the three
# base-size models: run_019 (unfiltered blitz), run_023 (ELO>=2500), v2.1 (production).
#
# Answers two things at once:
#   1. how often Stockfish's best move sits below the PUCT visibility floor (INVIS@N), and
#   2. whether the ELO filter narrows the policy (entropy / effective-moves columns) --
#      the prediction being that a sharper prior pushes MORE candidates below the floor.
set -uo pipefail
cd /home/tchauffi/Documents/dev/ChessTransformer || exit 1

echo "waiting for trainer to exit..."
while pgrep -f "pos2move_v2_trainer" > /dev/null; do sleep 60; done
echo "trainer done at $(date)"

RUN023="$(ls -d logs/pos2move_v2/run_023_*/checkpoints/best_model 2>/dev/null | head -1)"
if [[ -z "$RUN023" ]]; then echo "ERROR: no run_023 best_model found"; exit 1; fi

exec /home/tchauffi/.local/bin/uv run python scripts/eval_search_coverage.py \
  --models logs/pos2move_v2/run_019_20260817_084707/checkpoints/best_model \
           "$RUN023" \
           data/models/pos2move_v2.1 \
  --positions 500 \
  --sims 800 1800
