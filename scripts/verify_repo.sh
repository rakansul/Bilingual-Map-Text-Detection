#!/usr/bin/env bash
set -euo pipefail

echo "==> Verifying repository structure..."
required_files=(
  "README.md"
  "LICENSE"
  "requirements.txt"
  ".gitignore"
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
)

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing file: $file"
    exit 1
  fi
done

echo "✅ All required repository files present."

# Check data/sample
sample_imgs=$(find data/sample/images -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
sample_lbls=$(find data/sample/labels -type f -name "*.txt" 2>/dev/null | wc -l)

if [ "$sample_imgs" -lt 20 ] || [ "$sample_lbls" -lt 20 ]; then
  echo "❌ Expected at least 20 sample images and labels, found $sample_imgs images and $sample_lbls labels."
  exit 1
else
  echo "✅ Sample dataset verified ($sample_imgs images, $sample_lbls label files)."
fi

echo "==> Checking for uncommitted large model binaries (.pt / .onnx)..."
if find . -maxdepth 3 -type f \( -name "*.pt" -o -name "*.onnx" \) | grep -q .; then
  echo "⚠️ Warning: Found model binaries in repository directory. Ensure these are excluded via .gitignore."
else
  echo "✅ No model binaries found in git tree."
fi

echo "==> Repository validation completed successfully."
