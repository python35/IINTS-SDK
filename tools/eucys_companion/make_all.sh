#!/usr/bin/env bash
# Full rebuild: figures from the SDK, embedded fonts, web page, then the PDF.
#
#   ./make_all.sh            everything
#   ./make_all.sh --no-figs  skip the (slow) simulation runs
#
# Requires: the SDK's virtualenv python (IINTS_PY), node with ws installed in
# pdfbuild/, and any Chromium (CHROME_BIN, or a puppeteer-cached one).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${IINTS_PY:-$HERE/../../.venv/bin/python}"

[ -x "$PY" ] || { echo "No SDK python at $PY -- set IINTS_PY" >&2; exit 1; }
[ -d "$HERE/pdfbuild/node_modules/ws" ] || (cd "$HERE/pdfbuild" && npm install ws@8 --no-audit --no-fund)

if [ "${1:-}" != "--no-figs" ]; then
  echo "== figures: running the SDK =="
  (cd "$HERE/../.." && "$PY" "$HERE/make_figures.py" "$HERE/figures")
fi

[ -f "$HERE/src/91_fonts.css" ] || { echo "== fonts =="; python3 "$HERE/fetch_fonts.py"; }

echo "== web edition =="
python3 "$HERE/build.py"

echo "== pdf edition =="
bash "$HERE/make_pdf.sh"
