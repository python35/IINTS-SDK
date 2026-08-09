#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${IINTS_PLAYBOOK_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplt_eucys_playbook}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" tools/research/build_eucys_complete_playbook.py
