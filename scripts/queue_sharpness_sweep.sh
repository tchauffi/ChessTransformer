#!/usr/bin/env bash
# Sweep the cp-target sharpness for the bounded value head.
#
# S<1 flattens the target (less confident), S>1 sharpens it. Tests the two rival
# readings of why cp-trained heads lose ~47 Elo to production: that the targets
# are too confident for MCTS (flatten helps), or that the head loses
# discrimination by hedging -- measured output std 0.422 vs the target's own
# 0.484 and base's 0.495 (sharpen helps).
#
# S=1.0 already exists as data/models/pos2move_v2.1-sfvalue-bounded.
# Features depend only on --base/--data, so they are extracted once and reused.
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE=data/eval/lichess-sf/features_v2.1.pt
for S in 0.5 0.7 1.4 2.0; do
    OUT="data/models/pos2move_v2.1-sfvalue-s${S}"
    echo "=========== target sharpness S=$S -> $OUT ==========="
    uv run python scripts/train_value_head.py \
        --data data/eval/lichess-sf --base data/models/pos2move_v2.1 \
        --out "$OUT" --bounded --target-sharpness "$S" \
        --feature-cache "$CACHE"
done
