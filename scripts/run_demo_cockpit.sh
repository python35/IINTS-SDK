#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplt_demo}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache_demo}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PYTHONPATH=src python3 examples/demos/08_demo_cockpit.py "$@"
