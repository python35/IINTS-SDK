#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

build_manual_with_fallback () {
  local input="$1"
  local output="$2"
  if build_pdf "$input" "$output"; then
    return 0
  fi

  echo "[IINTS] Pandoc/Tectonic build failed; falling back to offline Python renderer..."
  python3 "$ROOT_DIR/tools/docs/render_manual_pdf.py" --input "$input" --output "$output"
}

build_manual_with_fallback "$MANUAL_DIR/IINTS-AF_SDK_Manual.md" "$MANUAL_DIR/IINTS-AF_SDK_Manual.pdf"

echo "Manuals built:"
echo " - $MANUAL_DIR/IINTS-AF_SDK_Manual.pdf"
