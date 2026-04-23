#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

input_path="research/EUCYS_REPORT.md"
output_path="research/EUCYS_REPORT.pdf"

usage() {
  cat <<'USAGE'
Usage:
  tools/research/render_eucys_report_pdf.sh [--input research/EUCYS_REPORT.md] [--output research/EUCYS_REPORT.pdf]

Options:
  --input PATH         Markdown report to render.
                       Default: research/EUCYS_REPORT.md
  --output PATH        Output PDF path.
                       Default: research/EUCYS_REPORT.pdf
  --help               Show this help text.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      input_path="$2"
      shift 2
      ;;
    --output)
      output_path="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'." >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "$output_path")"

tectonic_cache_dir="$ROOT_DIR/docs/manuals/.tectonic-cache"
xdg_cache_dir="$ROOT_DIR/.cache_eucys/xdg"

if [[ ! -d "$tectonic_cache_dir" ]]; then
  tectonic_cache_dir="$ROOT_DIR/.cache_eucys/tectonic"
fi

mkdir -p "$tectonic_cache_dir" "$xdg_cache_dir"

if command -v pandoc >/dev/null 2>&1 && command -v tectonic >/dev/null 2>&1; then
  render_log="$ROOT_DIR/.cache_eucys/pandoc_render.log"
  mkdir -p "$(dirname "$render_log")"
  set +e
  TECTONIC_CACHE_DIR="$tectonic_cache_dir" \
  XDG_CACHE_HOME="$xdg_cache_dir" \
  pandoc "$input_path" \
    --from markdown \
    --pdf-engine=tectonic \
    --metadata-file research/pandoc.yaml \
    --output "$output_path" \
    >"$render_log" 2>&1
  pandoc_status=$?
  set -e

  if [[ $pandoc_status -eq 0 ]]; then
    echo "[IINTS] EUCYS PDF ready: $output_path"
    exit 0
  fi

  echo "[IINTS] Pandoc/Tectonic PDF render failed; falling back to the offline renderer." >&2
  echo "[IINTS] Render log: $render_log" >&2
fi

python3 tools/research/render_eucys_report_pdf.py --input "$input_path" --output "$output_path"

echo "[IINTS] EUCYS PDF ready: $output_path"
