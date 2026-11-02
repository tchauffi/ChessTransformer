#!/usr/bin/env bash
# Head-to-head on the GPU (fp32 ONNX + CUDA execution provider).
#
# THE ENV IS THE WHOLE POINT. The `ort` crate's bundled ONNX Runtime has no sm_120 kernels
# for the RTX 5070 Ti, so ORT_DYLIB_PATH must point at an official onnxruntime-gpu build --
# and that build's CUDA provider links libcublas/libcublasLt/libcudart .so.13, which live in
# the nvidia-cu13 wheel, NOT next to libonnxruntime.so. Miss them and ORT does not error: it
# silently falls back to the CPU provider and runs fp32, the slowest combination there is.
# Measured on one 1800-node search: 201 nps without the CUDA libs, 4166 nps with them (20.7x).
#
# Rather than ask you to eyeball the nps afterwards, this script now PROBES before running:
# one short search, parse the nps ct-bot reports, and refuse to start the match if it looks
# like a CPU fallback. A silent fallback does not just make a gate slow, it makes an
# overnight gate quietly not finish -- which is how a match gets read at half the games it
# was supposed to play.
#
# Requires fp32 model.onnx, not model.int8.onnx -- the CUDA provider does not take the
# quantised graph. That means a GPU match is NOT numerically comparable to an int8 CPU match.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Minimum nps to accept as "really on the GPU". CPU fp32 measures ~200 nps, GPU ~4000; 800
# leaves room for a contended box without ever admitting a fallback.
MIN_NPS="${MIN_NPS:-800}"
PROBE_MODEL="${PROBE_MODEL:-data/models/pos2move_v2.1/model.onnx}"
BOT="${BOT:-rust/target-cuda/release/ct-bot}"

# Locate an onnxruntime-gpu: a directory holding both the runtime and its CUDA provider.
# Prefer $ORT_DIR when set, else search the uv cache -- a stable location, unlike the
# per-session scratchpad this script used to hardcode.
#
# The VERSION IS NOT OPTIONAL. ct-bot links a specific `ort` crate release and the C ABI is
# not compatible across ONNX Runtime minors: pointing it at 1.24 when it wants 1.28 does not
# fail at load, it aborts inside release_env_on_exit with an unwind panic. The uv cache
# holds several versions at once, so pin rather than take whatever `find` returns first.
ORT_VERSION="${ORT_VERSION:-1.28.0}"
if [[ -z "${ORT_DIR:-}" ]]; then
  for cand in $(find /home/tchauffi/.cache/uv/archive-v0 \
      -name "libonnxruntime.so.${ORT_VERSION}" 2>/dev/null | sort); do
    if [[ -f "$(dirname "$cand")/libonnxruntime_providers_cuda.so" ]]; then
      ORT_DIR="$(dirname "$cand")"
      break
    fi
  done
fi
ORT_LIB="${ORT_DIR:+$ORT_DIR/libonnxruntime.so.${ORT_VERSION}}"
if [[ -n "$ORT_LIB" && ! -f "$ORT_DIR/libonnxruntime_providers_cuda.so" ]]; then
  echo "ERROR: $ORT_DIR has no libonnxruntime_providers_cuda.so -- that is a CPU-only" >&2
  echo "       onnxruntime build. Install onnxruntime-gpu." >&2
  exit 1
fi
CU13="$(dirname "$(find /home/tchauffi/.cache/uv/archive-v0/*/lib/python3.12/site-packages/nvidia -name libcublas.so.13 2>/dev/null | head -1)")"
CUDNN="$(dirname "$(find /home/tchauffi/.cache/uv/archive-v0/*/lib/python3.12/site-packages/nvidia -name 'libcudnn.so*' 2>/dev/null | head -1)")"

if [[ -z "$ORT_LIB" || ! -f "$ORT_LIB" ]]; then
  echo "ERROR: no onnxruntime-gpu found (looked in ORT_DIR='$ORT_DIR')." >&2
  echo "       Install onnxruntime-gpu or set ORT_DIR to a dir with libonnxruntime.so.* " >&2
  echo "       and libonnxruntime_providers_cuda.so." >&2
  exit 1
fi
if [[ -z "$CU13" ]]; then
  echo "ERROR: CUDA 13 libs (libcublas.so.13) not found; the CUDA provider would fail to" >&2
  echo "       load and ORT would fall back to CPU silently. Refusing to run." >&2
  exit 1
fi
if [[ ! -x "$BOT" ]]; then
  echo "ERROR: $BOT missing. Build it with:" >&2
  echo "       cargo build --release --features cuda --target-dir rust/target-cuda" >&2
  exit 1
fi
if [[ ! -f "$PROBE_MODEL" ]]; then
  echo "ERROR: $PROBE_MODEL missing. The CUDA provider needs the fp32 graph, not int8." >&2
  exit 1
fi

export ORT_DYLIB_PATH="$ORT_LIB"
export LD_LIBRARY_PATH="$ORT_DIR:$CU13:$CUDNN:${LD_LIBRARY_PATH:-}"

echo "ORT_DYLIB_PATH=$ORT_DYLIB_PATH"
echo "probing the CUDA provider (need >= ${MIN_NPS} nps) ..."
PROBE="$(printf 'uci\nisready\nposition startpos\ngo nodes 800\nquit\n' \
    | timeout 180 "$BOT" uci --model "$PROBE_MODEL" --device cuda --threads 1 2>&1)"
NPS="$(printf '%s\n' "$PROBE" | grep -o 'nps [0-9]*' | head -1 | awk '{print $2}')"

if [[ -z "$NPS" ]]; then
  echo "ERROR: probe produced no nps line; ct-bot output was:" >&2
  printf '%s\n' "$PROBE" | tail -20 >&2
  exit 1
fi
if (( NPS < MIN_NPS )); then
  echo "ERROR: probe ran at ${NPS} nps, below the ${MIN_NPS} floor -- this is the silent" >&2
  echo "       CPU fallback, not the GPU. Check that $ORT_DIR really is an" >&2
  echo "       onnxruntime-gpu build and that the CUDA 13 libs are on LD_LIBRARY_PATH." >&2
  exit 1
fi
echo "probe OK: ${NPS} nps, CUDA provider is live."
echo

exec /home/tchauffi/.local/bin/uv run python scripts/head_to_head.py \
  --bot "$BOT" --device cuda "$@"
