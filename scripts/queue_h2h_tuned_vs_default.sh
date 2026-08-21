#!/usr/bin/env bash
# The deployable question: is c_puct 2.0 / fpu 0.1 actually worth Elo on the LIVE model?
#
# Same model (v2.1) on both sides, so search parameters are the only variable -- no model
# confound at all. A win here is deployable to ct-bot-lichess.service immediately and is
# independent of which checkpoint is in production.
#
# The screen (scripts/tune_mcts_params.py) said tuned > default on SF-agreement for both
# models, but that is a static-position proxy: it cannot see the failure mode the high-sim
# investigation found, where MORE exploration surfaces MORE value-head error and costs
# strength. That is exactly what this match tests.
#
# Restarts the lichess bot when done -- it is stopped for the match to keep the box quiet.
set -uo pipefail
cd /home/tchauffi/Documents/dev/ChessTransformer || exit 1

echo "waiting for the in-flight match to finish..."
while pgrep -f 'head_to_hea[d]\.py' > /dev/null; do sleep 60; done
echo "previous match done at $(date); starting v2.1 tuned vs v2.1 default"

scripts/h2h_gpu.sh \
  --model-a data/models/pos2move_v2.1/model.onnx --label-a v21_tuned \
  --model-b data/models/pos2move_v2.1/model.onnx --label-b v21_default \
  --nodes-a 1800 --nodes-b 1800 \
  --cpuct-a 2.0 --fpu-a 0.1 \
  --cpuct-b 1.0 --fpu-b 0.2 \
  --openings 64 --workers 4 --threads 1
rc=$?

echo "match finished at $(date); restarting the lichess bot"
systemctl --user start ct-bot-lichess.service
exit $rc
