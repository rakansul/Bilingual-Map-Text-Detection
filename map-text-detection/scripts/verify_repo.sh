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
)

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing file: $file"
    exit 1
  fi
done

echo "✅ All required repository files present."

echo "==> Checking for uncommitted large model binaries (.pt / .onnx)..."
if find . -maxdepth 3 -type f \( -name "*.pt" -o -name "*.onnx" \) | grep -q .; then
  echo "⚠️ Warning: Found model binaries in repository directory. Ensure these are excluded via .gitignore before pushing."
else
  echo "✅ No model binaries found in git tree."
fi

echo "==> Repository validation completed successfully."
