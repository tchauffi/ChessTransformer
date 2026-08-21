#!/usr/bin/env bash
# Model-size scaling experiment: `large` (156.4M) against run_019's `base` (11.7M).
#
# Everything except geometry matches run_019 (batch 512 x accum 4 = 2048 effective,
# 50k steps, lr 1e-3, full_k4 shards) so model size is the only variable. run_019's
# reference number is best_val_loss 4.2065 at step 50000 -- same val shards, so the
# two are directly comparable.
#
# Micro-batch is 128 rather than 512 because `large` peaks at 14.5 GB/step at 256 on
# this 16 GB card -- too tight to leave running for two days. 128 peaks at 8.0 GB and
# --grad-accum 16 restores the same 2048 effective batch. Measured 568 samples/s
# compiled, so 50k x 2048 = 102.4M samples lands around 50 h.
#
# NOT a bid for playing strength on its own: the 46M run already lost its head-to-head
# at 44.8%, on the 1.5M-game elite_db. What is new is the data -- full_k4 holds 105M
# samples from 26.3M games -- which is why the capacity question is worth reopening.
cd /home/tchauffi/Documents/dev/ChessTransformer || exit 1
exec /home/tchauffi/.local/bin/uv run python \
  src/chesstransformer/trainers/pos2move_v2_trainer.py \
  --shards data/shards/full_k4 \
  --preset large \
  --batch-size 128 --grad-accum 16 \
  --lr 1e-3 \
  --max-steps 50000 \
  --eval-steps 1000 \
  --save-steps 5000 --max-checkpoints 10 \
  --num-workers 0 \
  --compile \
  "$@"
