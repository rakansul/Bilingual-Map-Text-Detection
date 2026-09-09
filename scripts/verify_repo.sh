#!/usr/bin/env bash
set -euo pipefail

fail=0
note_fail() { echo "[FAIL] $1"; fail=1; }

echo "==> Verifying repository structure..."
required_files=(
  "README.md"
  "LICENSE"
  "requirements.txt"
  ".gitignore"
  ".gitattributes"
  "configs/data.yaml"
  "configs/data.sample.yaml"
  "src/__init__.py"
  "src/audit_dataset.py"
  "src/draw.py"
  "src/evaluate.py"
  "src/make_sample.py"
  "src/predict.py"
  "src/train.py"
  "docs/RUNBOOK.md"
  "docs/metrics.md"
  "docs/metrics.json"
  "docs/audit.md"
  "assets/hero.png"
  "data/sample/README.md"
  "scripts/selftest.py"
)

for file in "${required_files[@]}"; do
  [ -f "$file" ] || note_fail "Missing file: $file"
done
[ "$fail" -eq 0 ] && echo "[OK] All required repository files present."

echo "==> Checking bundled sample..."
sample_imgs=$(find data/sample/images -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
sample_lbls=$(find data/sample/labels -type f -name "*.txt" 2>/dev/null | wc -l)

if [ "$sample_imgs" -lt 1 ] || [ "$sample_lbls" -lt 1 ]; then
  note_fail "Sample set is empty (found $sample_imgs images, $sample_lbls labels in data/sample/)."
elif [ "$sample_imgs" -ne "$sample_lbls" ]; then
  note_fail "Sample image/label count mismatch: $sample_imgs images vs $sample_lbls labels."
else
  echo "[OK] Sample set present ($sample_imgs tiles with matching labels)."
fi

echo "==> Checking sample labels are 9-token OBB..."
bad_lines=$(awk 'NF>0 && NF!=9 {c++} END {print c+0}' data/sample/labels/*.txt 2>/dev/null || echo 0)
if [ "$bad_lines" -ne 0 ]; then
  note_fail "$bad_lines sample label line(s) are not 9-token OBB (cls x1 y1 x2 y2 x3 y3 x4 y4)."
else
  echo "[OK] All sample labels are 9-token OBB."
fi

echo "==> Checking every image has a label and vice versa..."
orphans=0
for img in data/sample/images/*.png data/sample/images/*.jpg; do
  [ -e "$img" ] || continue
  stem=$(basename "${img%.*}")
  [ -f "data/sample/labels/$stem.txt" ] || { note_fail "No label for $stem"; orphans=1; }
done
[ "$orphans" -eq 0 ] && echo "[OK] Every sample tile has a matching label file."

echo "==> Checking docs do not reference the old sample path..."
stale_path="data/sample/Sam""ples"
if grep -rn "$stale_path" --include="*.md" --include="*.yaml" --include="*.py" \
     --exclude-dir=scripts . >/dev/null 2>&1; then
  note_fail "Found stale references to the pre-rename sample folder:"
  grep -rn "$stale_path" --include="*.md" --include="*.yaml" --include="*.py" --exclude-dir=scripts .
else
  echo "[OK] No stale sample-path references."
fi

echo "==> Checking generated-section markers are intact..."
grep -q "<!-- AUDIT:BEGIN -->" docs/audit.md \
  || note_fail "docs/audit.md is missing its AUDIT:BEGIN marker."
grep -q "<!-- METRICS:BEGIN -->" docs/metrics.md \
  || note_fail "docs/metrics.md is missing its METRICS:BEGIN marker."
[ "$fail" -eq 0 ] && echo "[OK] Generated-section markers present."

echo "==> Checking source files compile..."
if python -m compileall -q src >/dev/null 2>&1; then
  echo "[OK] src/ compiles."
else
  note_fail "src/ failed to compile."
fi

echo "==> Running offline self-test..."
if python scripts/selftest.py >/tmp/selftest.log 2>&1; then
  echo "[OK] $(tail -1 /tmp/selftest.log)"
else
  note_fail "Self-test failed. Output:"
  cat /tmp/selftest.log
fi

echo "==> Checking for model binaries in the tree..."
if find . -maxdepth 3 -type f \( -name "*.pt" -o -name "*.onnx" \) | grep -q .; then
  echo "[WARNING] Model binaries present. Confirm .gitignore excludes them."
else
  echo "[OK] No model binaries in the tree."
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "Repository validation FAILED."
  exit 1
fi
echo "Repository validation completed successfully."
