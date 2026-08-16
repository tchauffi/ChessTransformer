#!/usr/bin/env bash
# Launch pos2move_v2_trainer under torchrun.
#
# torchrun rather than `accelerate launch` on purpose: the rendezvous arguments are
# explicit here, which is the part worth understanding. accelerate/ddp.yaml holds the
# equivalent config for comparison -- `accelerate launch --config_file accelerate/ddp.yaml`
# generates roughly what this script spells out.
#
#   NPROC        ranks to start (default: every visible GPU, else 1)
#   MASTER_PORT  rendezvous port (default 29500); change it to run two jobs at once
#   CPU=1        force gloo/CPU -- the free way to debug DDP correctness, no GPU needed
#
# Examples:
#   scripts/launch_train.sh --shards data/shards/elite_k16 --preset large --max-steps 20000
#   NPROC=2 scripts/launch_train.sh --shards data/shards/elite_k16 --preset large
#   CPU=1 NPROC=4 scripts/launch_train.sh --shards data/shards/elite_k16 \
#       --embed-dim 32 --num-layers 2 --max-steps 20 --no-compile --precision fp32
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${CPU:-0}" == "1" ]]; then
  export ACCELERATE_USE_CPU=1
  NPROC="${NPROC:-2}"
else
  if [[ -z "${NPROC:-}" ]]; then
    NPROC="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
    [[ "$NPROC" -lt 1 ]] && NPROC=1
  fi
fi
MASTER_PORT="${MASTER_PORT:-29500}"

# One dataloader worker per rank at most. The shard path is fastest at 0 workers when the
# shards fit page cache (a fetch is a memmap read, cheaper than the IPC to hand it off),
# and needs workers only when they don't and the bottleneck becomes NVMe queue depth.
# Oversubscribing here is how an 8-rank node ends up with 96 worker processes on 16 cores.
CORES="$(nproc)"
echo "launch: ${NPROC} rank(s) | master_port=${MASTER_PORT} | ${CORES} cores | CPU=${CPU:-0}"

exec torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  --max-restarts=0 \
  src/chesstransformer/trainers/pos2move_v2_trainer.py "$@"
