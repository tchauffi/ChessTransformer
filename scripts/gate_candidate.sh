#!/usr/bin/env bash
# Export a candidate model dir and SPRT-gate it against production v2.1.
#
# This exists to make the right thing the easy thing. doc/selfplay_rl.md records
# that supervised val loss ANTI-correlates with playing strength on this project
# -- exit2 had the best held-out metrics of any expert-iteration run and the
# worst gate. So no training script here is allowed to pick its own winner:
# every checkpoint is a candidate, and this is what decides between them.
#
# Defaults are the production search config (1800 nodes, ct-bot's c_puct 1.0 /
# fpu 0.2) because results here flip sign with the node budget -- the S=3.0
# value head scored 46.9% at 400 sims and 53.4% at 800, and expert-iteration
# exit2 scored 42.7% at 400 and 47.9% at 128. Gate at what you deploy.
#
# Usage
#   scripts/gate_candidate.sh data/models/v2.1-grpo-sf/step_001500
#   OPENINGS=1200 ELO1=10 scripts/gate_candidate.sh <dir>
#   DEVICE=cpu WORKERS=6 scripts/gate_candidate.sh <dir>      # no GPU available
set -euo pipefail
cd "$(dirname "$0")/.."

CAND="${1:?usage: gate_candidate.sh <model_dir> [extra head_to_head args...]}"
shift || true

BASELINE="${BASELINE:-data/models/pos2move_v2.1}"
OPENINGS="${OPENINGS:-1000}"
NODES="${NODES:-1800}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-15}"
BOOK="${BOOK:-data/openings/book2k.json}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-6}"

if [[ ! -f "$CAND/model.safetensors" ]]; then
  echo "ERROR: $CAND has no model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$BOOK" ]]; then
  echo "ERROR: $BOOK missing. Build one with scripts/build_opening_book.py --" >&2
  echo "       the built-in book tops out at 132 games, which cannot resolve" >&2
  echo "       anything under roughly +80 Elo." >&2
  exit 1
fi

# The CUDA execution provider needs the fp32 graph, so export fp32 either way
# and let --device pick. onnxscript lives in the 'optimized' dependency group.
CAND_ONNX="$CAND/model.onnx"
if [[ ! -f "$CAND_ONNX" ]]; then
  echo "=== exporting $CAND -> $CAND_ONNX ==="
  uv run --with onnxscript python scripts/export_onnx.py "$CAND" -o "$CAND_ONNX"
fi
BASE_ONNX="$BASELINE/model.onnx"
if [[ ! -f "$BASE_ONNX" ]]; then
  echo "=== exporting $BASELINE -> $BASE_ONNX ==="
  uv run --with onnxscript python scripts/export_onnx.py "$BASELINE" -o "$BASE_ONNX"
fi

LABEL_A="$(basename "$(dirname "$CAND_ONNX")")"
echo
echo "=== SPRT gate: $LABEL_A vs prod v2.1 @ ${NODES} nodes, up to $((2 * OPENINGS)) games ==="

ARGS=(--book "$BOOK" --openings "$OPENINGS"
      --model-a "$CAND_ONNX" --model-b "$BASE_ONNX"
      --label-a "$LABEL_A" --label-b prod_v2.1
      --nodes-a "$NODES" --nodes-b "$NODES"
      --sprt --elo0 "$ELO0" --elo1 "$ELO1" --workers "$WORKERS" "$@")

if [[ "$DEVICE" == "cuda" ]]; then
  # h2h_gpu.sh probes the CUDA provider and refuses a silent CPU fallback.
  exec scripts/h2h_gpu.sh "${ARGS[@]}"
else
  exec uv run python scripts/head_to_head.py --device cpu --threads 1 "${ARGS[@]}"
fi
