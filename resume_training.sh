#!/usr/bin/env bash
# Auto-resume the 46M (512x16) training run from its latest checkpoint.
cd /home/tchauffi/Documents/dev/ChessTransformer || exit 1
exec /home/tchauffi/.local/bin/uv run python src/chesstransformer/trainers/pos2move_v2_trainer.py \
  --embed-dim 512 --num-layers 16 --num-heads 8 \
  --batch-size 256 --grad-accum 16 \
  --max-steps 70000 --epochs 300 \
  --compile --save-steps 5000 --max-checkpoints 10 \
  --resume-from logs/pos2move_v2/run_025_20260608_224846/checkpoints/best_model \
  >> resume_training.log 2>&1
