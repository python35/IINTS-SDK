# Data Certification Quickstart

Use this page when you need a quick, defensible answer to one question: “is this dataset good enough to use?”

By the end, you should have a contract, a certification report, and a shareable audit dashboard.

**Read before:** [Getting Started](GETTING_STARTED.md) if you have not produced a run or dataset yet.

**Read next:** [Data Certification Full Guide](MDMP_FULL_GUIDE.md) when you need the full protocol details.

!!! important "Use a virtual environment"
    Run all commands from an active `.venv`.

## What Certification Produces

- contract validation results
- compliance score
- deterministic dataset and contract fingerprints
- trust grade: `draft`, `research_grade`, or `clinical_grade`

## Fastest Working Path

### 1. Create a contract

```bash
iints data certify-template --output-path data_contract.yaml
```

Edit the schema, units, and value ranges so they match your dataset.

### 2. Validate the dataset

```bash
iints data certify data_contract.yaml data/my_cgm.csv --output-json results/certification.json
```

For pipelines, use a strict gate:

```bash
iints data certify data_contract.yaml data/my_cgm.csv \
  --min-mdmp-grade research_grade \
  --fail-on-noncompliant \
  --output-json results/certification.json
```

### 3. Generate a dashboard

```bash
iints data certify-visualizer results/certification.json --output-html results/mdmp_dashboard.html
```

The dashboard is a single-file HTML artifact you can share offline.

## Optional: Create Synthetic Mirror Data

```bash
iints data synthetic-mirror data/my_cgm.csv data_contract.yaml \
  --output-csv data/synthetic_mirror.csv \
  --output-json results/synthetic_mirror_report.json
```

Use this when you need schema-realistic development data without distributing raw sensitive rows.

## Grade Meaning

| Grade | Interpretation |
| --- | --- |
| `draft` | useful for iteration, not ready for rigorous research claims |
| `research_grade` | acceptable for research workflows |
| `clinical_grade` | strongest validation level currently exposed by the SDK |

## Optional Python Gate

```python
import pandas as pd
from iints import mdmp_gate

@mdmp_gate("contracts/clinical_mdmp_contract.yaml", min_grade="research_grade")
def process(df: pd.DataFrame) -> int:
    return len(df)
```

This blocks or warns when incoming data does not meet the required quality level.

## Where To Go Next

| If you want to... | Continue with |
| --- | --- |
| understand every rule and fingerprint | [Data Certification Full Guide](MDMP_FULL_GUIDE.md) |
| use certified data in a benchmark | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| inspect the science behind realism claims | [Evidence Base](EVIDENCE_BASE.md) |
