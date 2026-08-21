#!/usr/bin/env bash
# tune_mcts_params.py under the same CUDA env as scripts/h2h_gpu.sh. See that script's
# header for why the nvidia-cu13 libs are mandatory (without them ORT silently runs CPU).
set -uo pipefail
cd /home/tchauffi/Documents/dev/ChessTransformer || exit 1
ORT_DIR="${ORT_DIR:-/tmp/claude-1000/-home-tchauffi-Documents-dev-ChessTransformer/37467f43-c160-4d73-a1ab-eeb7f74c48ab/scratchpad/ortgpu/onnxruntime/capi}"
CU13="$(dirname "$(find /home/tchauffi/.cache/uv/archive-v0/*/lib/python3.12/site-packages/nvidia -name libcublas.so.13 2>/dev/null | head -1)")"
CUDNN="$(dirname "$(find /home/tchauffi/.cache/uv/archive-v0/*/lib/python3.12/site-packages/nvidia -name 'libcudnn.so*' 2>/dev/null | head -1)")"
if [[ ! -f "$ORT_DIR/libonnxruntime.so.1.28.0" || -z "$CU13" ]]; then
  echo "ERROR: onnxruntime-gpu or CUDA 13 libs missing; refusing to run a silent CPU fallback"; exit 1
fi
export ORT_DYLIB_PATH="$ORT_DIR/libonnxruntime.so.1.28.0"
export LD_LIBRARY_PATH="$ORT_DIR:$CU13:$CUDNN:${LD_LIBRARY_PATH:-}"
exec /home/tchauffi/.local/bin/uv run python scripts/tune_mcts_params.py "$@"
