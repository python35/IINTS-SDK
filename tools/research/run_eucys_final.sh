#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

algo=""
output_dir="results/eucys_2026"
seeds="1,2,3,4,5,6,7,8,9,10"
extra_args=()

usage() {
  cat <<'USAGE'
Usage:
  tools/research/run_eucys_final.sh --algo algorithms/example_algorithm.py [options]

Options:
  --algo PATH          Required path to the candidate algorithm file.
  --output-dir PATH    Output directory for the final EUCYS study bundle.
                       Default: results/eucys_2026
  --seeds LIST         Comma-separated seed list.
                       Default: 1,2,3,4,5,6,7,8,9,10
  --help               Show this help text.

Any additional flags are forwarded to the source-tree `run-eucys-study` CLI command.
Example:
  tools/research/run_eucys_final.sh \
    --algo algorithms/example_algorithm.py \
    --output-dir results/eucys_2026 \
    --seeds 1,2,3,4,5,6,7,8,9,10
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      algo="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --seeds)
      seeds="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$algo" ]]; then
  echo "Error: --algo is required." >&2
  echo >&2
  usage >&2
  exit 1
fi

echo "[IINTS] Starting final EUCYS multi-seed study"
echo "  Algo:       $algo"
echo "  Output dir: $output_dir"
echo "  Seeds:      $seeds"
echo

python_cmd="${PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplt_eucys}"
mkdir -p "$MPLCONFIGDIR"

"$python_cmd" -c 'from iints.cli.cli import app; app()' -- run-eucys-study \
  --algo "$algo" \
  --output-dir "$output_dir" \
  --seeds "$seeds" \
  "${extra_args[@]}"

echo
echo "[IINTS] Building competition-ready EUCYS result package"
"$python_cmd" -c 'from iints.cli.cli import app; app()' -- eucys-results "$output_dir"

echo
echo "[IINTS] Final EUCYS study workflow complete"
echo "  Study bundle:         $output_dir"
echo "  Results package:      $output_dir/EUCYS_RESULTS"
echo "  Summary markdown:     $output_dir/EUCYS_RESULTS/EUCYS_SUMMARY.md"
echo "  Filled abstract:      $output_dir/EUCYS_RESULTS/EUCYS_ABSTRACT_FILLED.md"
echo "  Main figure:          $output_dir/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.png"
echo
echo "Next:"
echo "  1. Update research/EUCYS_REPORT.md with the final multi-seed numbers from $output_dir/EUCYS_RESULTS/."
echo "  2. Render the PDF with: tools/research/render_eucys_report_pdf.sh"
