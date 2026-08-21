#!/usr/bin/env bash
# Gate every sharpness-sweep head against production v2.1.
#
# 400 sims / cpuct 1.5 -- engine_match.py's own defaults, matching
# logs/gate_bounded_vs_base_400.log so the S=1.0 result (43.8%, -44 Elo) is a
# like-for-like reference point in the same table.
#
# Deliberately gating ALL of them rather than pre-screening on val MSE or the
# calibration diagnostic: on this model val CE, val MSE and ECE have each
# anti-correlated with playing strength, so proxies cannot rank these heads.
set -euo pipefail
cd "$(dirname "$0")/.."

for S in 0.5 0.7 1.4 2.0; do
    M="data/models/pos2move_v2.1-sfvalue-s${S}"
    [ -d "$M" ] || { echo "skip $M (missing)"; continue; }
    echo "=========== S=$S vs base @400 sims ==========="
    uv run python scripts/engine_match.py \
        --a-mcts --b-mcts --a-sims 400 --b-sims 400 \
        --a-model-dir "$M" --b-model-dir data/models/pos2move_v2.1 \
        --openings 24 2>&1 | tee "logs/gate_sharpness_s${S}.log"
done
