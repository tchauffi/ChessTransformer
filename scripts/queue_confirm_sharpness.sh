#!/usr/bin/env bash
# Confirmation match for the sharpness plateau head (S=3.0) vs production v2.1.
#
# The sweep put S in [2,4] at 53.1% (+22 Elo) over 48 games, but that is only
# 0.63 sigma -- and S=2/3/4 pick identical moves on all 25 probe positions, so
# those three gates are ONE correlated result, not three replications. This
# re-runs on the full 81-opening book (162 games) to tighten the interval, at
# both the sweep config and the deployed search settings.
#
# Even 162 games leaves ~1.1 sigma on a +22 Elo effect; resolving that to 2
# sigma needs ~500 games. Read the CI, not the point estimate.
set -euo pipefail
cd "$(dirname "$0")/.."

M=data/models/pos2move_v2.1-sfvalue-s3.0
B=data/models/pos2move_v2.1

echo "=== S=3.0 vs base, 162 games @400 sims cpuct 1.5 (sweep config) ==="
uv run python scripts/engine_match.py --a-mcts --b-mcts --a-sims 400 --b-sims 400 \
    --a-model-dir "$M" --b-model-dir "$B" --openings 81 \
    2>&1 | tee logs/confirm_s3_400.log

echo "=== S=3.0 vs base, 162 games @800 sims cpuct 1.0 / fpu 0.2 (production) ==="
uv run python scripts/engine_match.py --a-mcts --b-mcts --a-sims 800 --b-sims 800 \
    --a-cpuct 1.0 --b-cpuct 1.0 --a-fpu 0.2 --b-fpu 0.2 \
    --a-model-dir "$M" --b-model-dir "$B" --openings 81 \
    2>&1 | tee logs/confirm_s3_800_tuned.log
