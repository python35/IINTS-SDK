#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplt_eucys_dossier}"
mkdir -p "$MPLCONFIGDIR"

python3 tools/research/build_eucys_dossier.py
