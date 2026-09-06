#!/usr/bin/env bash
# Two-pass PDF build. Pass 1 establishes pagination, the outline gives each
# chapter its page, pass 2 rebuilds the contents list with real page numbers.
set -euo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"
PY="${IINTS_PY:-$SP/../../.venv/bin/python}"
HTML="$SP/iints-af-model-companion.print.html"
OUT="$SP/IINTS-AF-Model-Companion.pdf"

echo "== pass 1: pagination =="
python3 "$SP/build_pdf.py"
node "$SP/pdfbuild/render_pdf.mjs" "$HTML" "$SP/pass1.pdf"

echo "== outline -> page numbers =="
"$PY" "$SP/read_outline.py" "$SP/pass1.pdf" "$SP/chapters.json" "$SP/pages.json"

echo "== pass 2: final =="
python3 "$SP/build_pdf.py" "$SP/pages.json"
node "$SP/pdfbuild/render_pdf.mjs" "$HTML" "$OUT"
echo "== done: $OUT =="
