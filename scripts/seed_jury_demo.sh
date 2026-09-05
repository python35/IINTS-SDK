#!/usr/bin/env bash
# Seed one folder with a worked example for every panel of the desktop app.
#
# Run this before a demonstration, then set the app's output folder to the
# seeded path. The script prints, per panel, whether it is ready to show.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.mplt_demo}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache_demo}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PYTHONPATH=src python3 -m iints_desktop.jury_demo "$@"
