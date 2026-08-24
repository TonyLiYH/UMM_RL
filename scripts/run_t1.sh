#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${COMPPARETO_PYTHON:-}" ]]; then
  PYTHON_BIN="$COMPPARETO_PYTHON"
elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
OUTPUT_DIR="${1:-$REPO_DIR/runs/t1_synthetic}"

cd "$REPO_DIR"
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m comppareto.synthetic \
  --config configs/t1_synthetic.json \
  --output "$OUTPUT_DIR/t1_manifest.json"

