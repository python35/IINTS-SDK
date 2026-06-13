---
tags:
  - iints/results
  - iints/public
cssclasses:
  - iints-dashboard
status: public
---
# Output Bundle Anatomy

Use this to explain generated SDK result folders.

| Artifact | What it records |
| --- | --- |
| `results.csv` | simulation time-series output |
| `summary.json` | high-level metrics and completion state |
| `report.pdf` | human-readable research report |
| `run_config` | exact run settings |
| `patient_config` | virtual patient assumptions |
| `scenario` | meals/events/stress assumptions |
| `source_manifest` | source/evidence traceability |
| `validation` | pass/fail gates and warnings |

The bundle matters because it keeps outputs reproducible rather than just visually impressive.
