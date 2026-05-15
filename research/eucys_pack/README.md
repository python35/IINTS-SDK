# EUCYS Evidence Pack

This folder contains jury-facing explanation bundles for the IINTS-AF SDK. The files are written as Markdown and rendered to PDF by `tools/research/build_eucys_pack.sh`.

## Bundles

| Source | PDF | Purpose |
|---|---|---|
| `EUCYS_00_INDEX.md` | `pdf/EUCYS_00_INDEX.pdf` | Reading order, one-page story, verification map |
| `EUCYS_01_TECHNICAL_BRIEF.md` | `pdf/EUCYS_01_TECHNICAL_BRIEF.pdf` | SDK architecture, reproducibility, safety supervisor, MDMP |
| `EUCYS_02_PHYSIOLOGY_AND_DATA_BRIEF.md` | `pdf/EUCYS_02_PHYSIOLOGY_AND_DATA_BRIEF.pdf` | Type 1 diabetes physiology, CGM metrics, realistic data requirements |
| `EUCYS_03_IMPACT_ETHICS_AND_MAINTENANCE.md` | `pdf/EUCYS_03_IMPACT_ETHICS_AND_MAINTENANCE.pdf` | Why the SDK matters, ethics, limitations, maintenance evidence |
| `EUCYS_04_JURY_QA.md` | `pdf/EUCYS_04_JURY_QA.pdf` | Short answers to likely judge questions |

## Build

```bash
tools/research/build_eucys_pack.sh
```

The build uses the offline PDF renderer in `tools/research/render_eucys_report_pdf.py`, so the pack can be regenerated without online services.

## Related Evidence

The full benchmark report remains available at:

- `research/EUCYS_REPORT.md`
- `research/EUCYS_REPORT.pdf`

The main figure and result tables copied into this folder come from the May 8, 2026 final benchmark bundle at `results/eucys_2026/EUCYS_RESULTS/`.
