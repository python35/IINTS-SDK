#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PACK_DIR="research/eucys_pack"
PDF_DIR="$PACK_DIR/pdf"
mkdir -p "$PDF_DIR"

sources=(
  "EUCYS_00_INDEX.md"
  "EUCYS_01_TECHNICAL_BRIEF.md"
  "EUCYS_02_PHYSIOLOGY_AND_DATA_BRIEF.md"
  "EUCYS_03_IMPACT_ETHICS_AND_MAINTENANCE.md"
  "EUCYS_04_JURY_QA.md"
  "EUCYS_05_PHYSIOLOGY_REFERENCE_BROCHURE.md"
  "EUCYS_06_JURY_PHYSIOLOGY_BRIEF.md"
)

for source_name in "${sources[@]}"; do
  input_path="$PACK_DIR/$source_name"
  output_path="$PDF_DIR/${source_name%.md}.pdf"
  python3 tools/research/render_eucys_report_pdf.py \
    --input "$input_path" \
    --output "$output_path"
done

tools/research/build_eucys_dossier.sh

printf '[IINTS] EUCYS evidence pack ready: %s\n' "$PDF_DIR"
