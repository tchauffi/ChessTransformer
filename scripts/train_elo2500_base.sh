#!/usr/bin/env bash
# Data-quality experiment: does filtering the blitz corpus to avg ELO >= 2500 fix the
# -89 Elo regression run_019 suffered against v2.1?
#
# Everything matches run_019 -- base preset, batch 512 x accum 4 = 2048 effective, 50k
# steps, lr 1e-3 -- so the training corpus is the only variable. run_019 trained on
# data/shards/full_k4 (min_elo None, 105,076,308 samples); this trains on
# data/shards/full2500_k8 (min_elo 2500, 105,790,928 samples, +0.7%). k went 4 -> 8
# precisely to hold the sample budget fixed while the filter halves the games.
#
# NOTE: val CE is NOT comparable to run_019's. The val shards are ELO-filtered too, so
# this model is scored on a different (easier) distribution. Gate with
# scripts/head_to_head.py against run_019, not with val loss.
cd /home/tchauffi/Documents/dev/ChessTransformer || exit 1
exec /home/tchauffi/.local/bin/uv run python \
  src/chesstransformer/trainers/pos2move_v2_trainer.py \
  --shards data/shards/full2500_k8 \
  --preset base \
  --batch-size 512 --grad-accum 4 \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-steps 1000 \
  --save-steps 5000 --max-checkpoints 10 \
  --num-workers 0 \
  --compile \
  "$@"
