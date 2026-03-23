#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.mplt_demo}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache_demo}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PYTHONPATH=src python3 examples/demos/06_booth_demo.py "$@"
