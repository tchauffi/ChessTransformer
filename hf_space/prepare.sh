#!/usr/bin/env bash
# Assemble a self-contained Hugging Face Space in hf_space/space_build/, ready to
# push to a Space git remote. Bundles the app, slim requirements, the
# chesstransformer package source, and the v2.1 model weights.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BUILD="$HERE/space_build"
WEIGHTS="$REPO/data/models/pos2move_v2.1"

if [ ! -f "$WEIGHTS/model.safetensors" ]; then
  echo "ERROR: weights not found at $WEIGHTS (run git lfs pull?)" >&2
  exit 1
fi

rm -rf "$BUILD"
mkdir -p "$BUILD/model"

cp "$HERE/app.py" "$BUILD/app.py"
cp "$HERE/requirements.txt" "$BUILD/requirements.txt"
cp "$HERE/README.md" "$BUILD/README.md"

# Vendor the package source (inference path only needs this subtree).
cp -r "$REPO/src/chesstransformer" "$BUILD/chesstransformer"
# Drop bundled data (old model weights / tokenizer blobs ~200 MB) and caches —
# the inference path loads weights from ./model, not from inside the package.
rm -rf "$BUILD/chesstransformer/data"
find "$BUILD/chesstransformer" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$BUILD/chesstransformer" -name '*.pyc' -delete

# Bundle weights (HF will store the large file via LFS on push).
cp "$WEIGHTS/model_config.json" "$BUILD/model/model_config.json"
cp "$WEIGHTS/model.safetensors" "$BUILD/model/model.safetensors"

# A .gitattributes so the Space tracks the weights with LFS.
cat > "$BUILD/.gitattributes" <<'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
EOF

echo "Built Space at: $BUILD"
echo "Contents:"
( cd "$BUILD" && find . -maxdepth 2 -not -path '*/.*' | sort )
