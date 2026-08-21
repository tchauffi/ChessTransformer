#!/usr/bin/env bash
# Gate the tanh-bounded Stockfish-cp value head against production v2.1.
#
# The unbounded version of this head (data/models/pos2move_v2.1-sfvalue) fit the
# target best of any head tried and gated WORST -- 33.3% over 48 games -- with
# overconfidence past +-1 as the diagnosed cause. This runs the same match for
# the bounded head, which cannot express that failure.
#
# Two configs, because the original gate did NOT use production search params:
#   control  cpuct 1.5 / fpu default / 400 sims -- engine_match.py's own
#            defaults, exactly what logs/gate_sfvalue_vs_base_400.log ran, so
#            the 33.3% number is a like-for-like control.
#   prod     cpuct 1.0 / fpu 0.2 / 800 sims -- the deployed search settings,
#            which is the result that decides promotion.
#
# Both engines share the trunk and the policy; only value_head.* differs.
set -euo pipefail
cd "$(dirname "$0")/.."

BOUNDED=${BOUNDED:-data/models/pos2move_v2.1-sfvalue-bounded}
BASE=${BASE:-data/models/pos2move_v2.1}
GAMES_OPENINGS=${GAMES_OPENINGS:-24}   # x2 colours = 48 games

echo "=== control: bounded vs base @400 sims, cpuct 1.5 (matches the sfvalue gate) ==="
uv run python scripts/engine_match.py \
    --a-mcts --b-mcts --a-sims 400 --b-sims 400 \
    --a-model-dir "$BOUNDED" --b-model-dir "$BASE" \
    --openings "$GAMES_OPENINGS" \
    2>&1 | tee logs/gate_bounded_vs_base_400.log

echo
echo "=== production: bounded vs base @800 sims, cpuct 1.0 / fpu 0.2 ==="
uv run python scripts/engine_match.py \
    --a-mcts --b-mcts --a-sims 800 --b-sims 800 \
    --a-cpuct 1.0 --b-cpuct 1.0 --a-fpu 0.2 --b-fpu 0.2 \
    --a-model-dir "$BOUNDED" --b-model-dir "$BASE" \
    --openings "$GAMES_OPENINGS" \
    2>&1 | tee logs/gate_bounded_vs_base_800_tuned.log
