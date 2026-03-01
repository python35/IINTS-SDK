#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANUAL_DIR="$ROOT_DIR/docs/manuals"

PANDOC_CONFIG="$MANUAL_DIR/pandoc.yaml"
PREAMBLE="$MANUAL_DIR/preamble.tex"
CACHE_DIR="$MANUAL_DIR/.tectonic-cache"
XDG_CACHE="$MANUAL_DIR/.cache"

build_pdf () {
  local input="$1"
  local output="$2"
  TECTONIC_CACHE_DIR="$CACHE_DIR" XDG_CACHE_HOME="$XDG_CACHE" \
  pandoc "$input" \
    --from markdown \
    --pdf-engine=tectonic \
    --metadata-file "$PANDOC_CONFIG" \
    --include-in-header "$PREAMBLE" \
    --output "$output"
}

build_pdf "$MANUAL_DIR/IINTS-AF_SDK_Manual_Improved.md" "$MANUAL_DIR/IINTS-AF_SDK_Manual_Improved.pdf"
build_pdf "$MANUAL_DIR/IINTS-AF_SDK_Manual_Improved_clean.md" "$MANUAL_DIR/IINTS-AF_SDK_Manual_Improved_clean.pdf"

echo "Manuals built:"
echo " - $MANUAL_DIR/IINTS-AF_SDK_Manual_Improved.pdf"
echo " - $MANUAL_DIR/IINTS-AF_SDK_Manual_Improved_clean.pdf"
