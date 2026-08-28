#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TNNN" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-.venv/bin/python}"
exec "$python_bin" -m comppareto.repo_state.submission_cli \
  --root . \
  --task "$1"
