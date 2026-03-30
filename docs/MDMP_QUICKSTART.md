# Data Certification Quickstart

Data certification is the IINTS trust layer for validating data quality before modeling or evaluation.

!!! important "Use a Virtual Environment"
    Run all commands from an active `.venv`:
    `python3 -m venv .venv && source .venv/bin/activate`

## What Certification Produces

- Contract validation results (pass/fail per rule)
- Compliance score
- Deterministic dataset + contract fingerprints
- Trust grade: `draft`, `research_grade`, or `clinical_grade`

## 1) Generate a Contract Template

```bash
iints data certify-template --output-path data_contract.yaml
```

Edit the contract to match your dataset schema and bounds.

## 2) Validate a Dataset

```bash
iints data certify data_contract.yaml data/my_cgm.csv --output-json results/certification.json
```

Strict mode for pipelines:

```bash
iints data certify data_contract.yaml data/my_cgm.csv \
  --min-mdmp-grade research_grade \
  --fail-on-noncompliant \
  --output-json results/certification.json
```

## 3) Generate an Audit Dashboard

```bash
iints data certify-visualizer results/certification.json --output-html results/mdmp_dashboard.html
```

This creates a single-file HTML report that can be shared offline.

## 4) Create Synthetic Mirror Data

```bash
iints data synthetic-mirror data/my_cgm.csv data_contract.yaml \
  --output-csv data/synthetic_mirror.csv \
  --output-json results/synthetic_mirror_report.json
```

Use this for safe development/testing when real data access is restricted.

## Grade Interpretation

- `draft`: schema partially valid, not ready for rigorous research workflows.
- `research_grade`: acceptable quality for research experiments.
- `clinical_grade`: strongest validation profile available in the SDK.

## Python Gate (Optional)

```python
import pandas as pd
from iints import mdmp_gate

@mdmp_gate("contracts/clinical_mdmp_contract.yaml", min_grade="research_grade")
def process(df: pd.DataFrame) -> int:
    return len(df)
```

This blocks or warns if input data does not meet required quality.

## Important Scope

MDMP improves data quality and traceability. It does not provide clinical approval.
