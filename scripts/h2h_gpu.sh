#!/usr/bin/env bash
# Head-to-head on the GPU (fp32 ONNX + CUDA execution provider).
#
# THE ENV IS THE WHOLE POINT. The `ort` crate's bundled ONNX Runtime has no sm_120 kernels
# for the RTX 5070 Ti, so ORT_DYLIB_PATH must point at an official onnxruntime-gpu build --
# and that build's CUDA provider links libcublas/libcublasLt/libcudart .so.13, which live in
# the nvidia-cu13 wheel, NOT next to libonnxruntime.so. Miss them and ORT does not error: it
# silently falls back to the CPU provider and runs fp32, the slowest combination there is.
# Measured on one 1800-node search: 201 nps without the CUDA libs, 4166 nps with them (20.7x).
# Always confirm the nps in the log before trusting a GPU match.
#
# Requires fp32 model.onnx, not model.int8.onnx -- the CUDA provider does not take the
# quantised graph. That means a GPU match is NOT numerically comparable to an int8 CPU match.
set -uo pipefail
cd /home/tchauffi/Documents/dev/ChessTransformer || exit 1

ORT_DIR="${ORT_DIR:-/tmp/claude-1000/-home-tchauffi-Documents-dev-ChessTransformer/37467f43-c160-4d73-a1ab-eeb7f74c48ab/scratchpad/ortgpu/onnxruntime/capi}"
CU13="$(dirname "$(find /home/tchauffi/.cache/uv/archive-v0/*/lib/python3.12/site-packages/nvidia -name libcublas.so.13 2>/dev/null | head -1)")"
CUDNN="$(dirname "$(find /home/tchauffi/.cache/uv/archive-v0/*/lib/python3.12/site-packages/nvidia -name 'libcudnn.so*' 2>/dev/null | head -1)")"
if [[ ! -f "$ORT_DIR/libonnxruntime.so.1.28.0" || -z "$CU13" ]]; then
  echo "ERROR: onnxruntime-gpu or the CUDA 13 libs are missing; refusing to run a silent CPU fallback"; exit 1
fi
export ORT_DYLIB_PATH="$ORT_DIR/libonnxruntime.so.1.28.0"
export LD_LIBRARY_PATH="$ORT_DIR:$CU13:$CUDNN:${LD_LIBRARY_PATH:-}"

exec /home/tchauffi/.local/bin/uv run python scripts/head_to_head.py \
  --bot rust/target-cuda/release/ct-bot --device cuda "$@"
