#!/usr/bin/env bash
# Assemble the Hugging Face Docker Space in hf_space/space_build/, ready to push
# or upload: the Dockerfile and Space README, the engine and frontend sources,
# and the int8 model. Sources are taken from git (tracked plus new, unignored
# files), so local build directories never end up in the Space.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BUILD="$HERE/space_build"
MODEL="$REPO/data/models/pos2move_v2.1/model.int8.onnx"

# An LFS pointer is a ~130-byte text file; the real model is ~16 MB.
if [ ! -f "$MODEL" ] || [ "$(stat -c %s "$MODEL")" -lt 1000000 ]; then
  echo "ERROR: model missing or an LFS pointer at $MODEL (run git lfs pull?)" >&2
  exit 1
fi

rm -rf "$BUILD"
mkdir -p "$BUILD"

cp "$HERE/Dockerfile" "$HERE/README.md" "$BUILD/"
( cd "$REPO" && git ls-files -z --cached --others --exclude-standard rust frontend \
    | rsync -a --from0 --files-from=- ./ "$BUILD/" )
cp "$MODEL" "$BUILD/model.int8.onnx"

cat > "$BUILD/.gitattributes" <<'ATTR'
*.onnx filter=lfs diff=lfs merge=lfs -text
ATTR
cat > "$BUILD/.dockerignore" <<'IGNORE'
**/node_modules
**/.next
**/target
IGNORE

echo "Built Space at: $BUILD ($(du -sh "$BUILD" | cut -f1))"
( cd "$BUILD" && find . -maxdepth 2 -not -path './.git*' | sort )
