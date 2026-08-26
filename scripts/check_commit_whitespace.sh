#!/usr/bin/env bash
set -euo pipefail

BASE_REVISION="${1:-HEAD^}"
HEAD_REVISION="${2:-HEAD}"

git -c core.whitespace=blank-at-eol,space-before-tab,-blank-at-eof \
  diff --check "$BASE_REVISION" "$HEAD_REVISION"
