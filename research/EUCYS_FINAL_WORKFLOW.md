# EUCYS Final Workflow

This page is the shortest path from a **seed-1 pilot** to a **final multi-seed EUCYS submission bundle**.

## Goal

For the final submission, the benchmark should no longer rely on one seed. The recommended path is:

1. run the locked EUCYS matrix over multiple seeds
2. generate the packaged EUCYS outputs
3. update `research/EUCYS_REPORT.md` with the final multi-seed results
4. export `research/EUCYS_REPORT.pdf`

## 1. Run the Final Multi-Seed Study

```bash
tools/research/run_eucys_final.sh \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10
```

What this does:
- runs `iints run-eucys-study`
- uses the fixed EUCYS matrix
- packages the outputs with `iints eucys-results`

Main output folder:
- `results/eucys_2026/`
- packaged competition outputs: `results/eucys_2026/EUCYS_RESULTS/`

## 2. Pull the Final Numbers Into the Report

Use these generated artifacts as the source of truth:
- `results/eucys_2026/study_summary.json`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_SUMMARY.md`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_RESULTS_TABLE.csv`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_ABSTRACT_FILLED.md`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.png`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.csv`

Update these sections in `research/EUCYS_REPORT.md`:
- abstract
- study design and seed policy
- aggregate results table
- candidate-vs-baseline deltas
- reproducibility/provenance section
- discussion and limitations

Important:
- change the language from `seed-1 pilot` to `final multi-seed benchmark`
- report the exact final seed list
- prefer `mean ± std` and the bundle confidence intervals where available

## 3. Export the PDF

```bash
tools/research/render_eucys_report_pdf.sh
```

Default output:
- `research/EUCYS_REPORT.pdf`

Custom paths are also supported:

```bash
tools/research/render_eucys_report_pdf.sh \
  --input research/EUCYS_REPORT.md \
  --output research/EUCYS_REPORT_vfinal.pdf
```

Notes:
- the script first tries `pandoc + tectonic` for a typeset PDF
- if that fails offline, it automatically falls back to a simpler built-in PDF renderer so you still end up with a submission-ready PDF file

## 4. Final Submission Checklist

Before sending anything to a jury, confirm:
- the report no longer describes the work as a seed-1 pilot
- the final seed list is visible in the report
- the main figure in the report matches the generated EUCYS figure package
- the README and report use the same research question wording
- the PDF opens cleanly and has no missing sections

## Recommended Final Command Sequence

```bash
tools/research/run_eucys_final.sh \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10

# update research/EUCYS_REPORT.md with the new numbers

tools/research/render_eucys_report_pdf.sh
```
