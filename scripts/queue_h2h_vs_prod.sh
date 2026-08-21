#!/usr/bin/env bash
# Second match, queued behind the first: run_023 (ELO>=2500) vs the PRODUCTION model v2.1.
#
# Sequential, not concurrent: this box has 8 physical cores and the match already runs 6
# workers, so two matches at once would just make both slower (see the h2h CPU tuning note).
#
# The two matches answer different questions. vs run_019 is the controlled experiment --
# same corpus size, same recipe, ELO filter the only variable -- so it says whether the
# filter *works*. vs v2.1 is the deployment decision: v2.1 is what is live, and it beat
# run_019 by -89 Elo, so run_023 has to clear v2.1, not merely run_019, to be promotable.
set -uo pipefail
cd /home/tchauffi/Documents/dev/ChessTransformer || exit 1

echo "waiting for the run_023-vs-run_019 match to finish..."
while pgrep -f "scripts/head_to_head.py" > /dev/null; do sleep 60; done
echo "first match done at $(date); starting run_023 vs v2.1"

exec /home/tchauffi/.local/bin/uv run python scripts/head_to_head.py \
  --model-a logs/pos2move_v2/run_023_20260818_134218/checkpoints/best_model/model.int8.onnx \
  --label-a run_023_elo2500 \
  --model-b data/models/pos2move_v2.1/model.int8.onnx \
  --label-b prod_v2.1 \
  --nodes-a 1800 --nodes-b 1800 \
  --openings 32 --threads 1 --workers 6
