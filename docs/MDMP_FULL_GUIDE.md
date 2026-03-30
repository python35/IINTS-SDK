# Data Certification Full Guide

This page is the complete implementation guide for the IINTS data-certification layer.

## Environment Requirement

Use an active virtual environment for all commands in this guide:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

## Who This Page Is For

- Researchers who need reproducible data validation before model training/evaluation.
- Engineers implementing quality gates in scripts, CI, and services.
- Reviewers who need traceable evidence for dataset integrity.

## What Data Certification Is (And Is Not)

Data certification is the data-quality protocol layer in IINTS-AF.

It does:
- Validate schema, types, value ranges, and explicit data rules.
- Generate deterministic fingerprints for contract and dataset.
- Assign a quality grade (`draft`, `research_grade`, `clinical_grade`).
- Produce machine-readable reports and optional HTML dashboards.

It does not:
- Grant clinical approval.
- Convert this SDK into a medical device.
- Replace model evaluation or clinical study design.

## Protocol Surface

Preferred public namespace:
- CLI: `iints data ...`
- Python: `iints.data`

Legacy compatibility namespace (hidden from public help):
- CLI: `iints mdmp ...`
- Python: `iints.mdmp`

## MDMP Contract Model

MDMP uses a YAML contract. Conceptually it has:
- `streams`: source + metadata + required columns + type/range expectations.
- `processes`: input stream, feature declarations, labels, and validation expressions.

Minimal example:

```yaml
version: 1
streams:
  - name: cgm
    source: csv
    security: PII_MINIMIZED
    metadata:
      required_columns: [timestamp, glucose, carbs, insulin]
      column_types:
        timestamp: datetime
        glucose: float
        carbs: float
        insulin: float
      ranges:
        glucose:
          min: 20
          max: 450
        carbs:
          min: 0
          max: 250
        insulin:
          min: 0
          max: 25
      unit_conversions:
        glucose:
          from: mmol/L
          to: mg/dL
processes:
  - name: glucose_quality
    input_stream: cgm.glucose
    validations:
      - expression: glucose is not null and glucose > 20
        on_fail: DISCARD_AND_LOG
```

## What The Validator Checks

When you run MDMP validation, the engine computes these checks:

1. `schema_columns`
- Verifies required columns exist.

2. `schema_types`
- Verifies column dtypes against contract (`float`, `int`, `string`, `bool`, `datetime`).

3. `value_ranges`
- Verifies numeric values are within `min`/`max` bounds from contract metadata.

4. `rule_validations`
- Evaluates contract rules from `processes[].validations[]`.

### Supported Rule Grammar

Current rule evaluator supports:
- `column is not null`
- `column is null`
- numeric comparisons: `column > 70`, `column <= 180`, etc.
- boolean composition using `and` / `or`

Example:

```text
glucose is not null and glucose > 20 and glucose < 450
```

## Scoring and Grades

MDMP computes:
- `compliance_score` = `(passed_checks / total_checks) * 100`
- `is_compliant` = all checks pass

Grade logic:
- `clinical_grade` if compliant and score >= 90
- `research_grade` if score >= 75
- `draft` otherwise

Protocol also returns:
- `certified_for_medical_research` = `true` for `research_grade` and `clinical_grade`
- `mdmp_protocol_version` (currently `1.0-draft`)

## Fingerprints and Reproducibility

Each validation result includes:
- `contract_fingerprint_sha256`
- `dataset_fingerprint_sha256`

Practical meaning:
- If data changes, dataset fingerprint changes.
- If contract changes, contract fingerprint changes.
- This allows exact audit traceability of what was validated.

## CLI Workflows

### 1) Create a contract template

```bash
iints data certify-template --output-path data_contract.yaml
```

### 2) Validate dataset

```bash
iints data certify data_contract.yaml data/my_cgm.csv --output-json results/certification.json
```

Strict gate example:

```bash
iints data certify data_contract.yaml data/my_cgm.csv \
  --min-mdmp-grade research_grade \
  --fail-on-noncompliant \
  --output-json results/certification.json
```

### 3) Build HTML dashboard

```bash
iints data certify-visualizer results/certification.json --output-html results/mdmp_dashboard.html
```

### 4) Generate synthetic mirror data

```bash
iints data synthetic-mirror data/my_cgm.csv data_contract.yaml \
  --output-csv data/synthetic_mirror.csv \
  --output-json results/synthetic_mirror_report.json
```

## `iints data certify` vs legacy commands

The public command is now `iints data certify`.

- `iints data certify`
  - public, unified certification flow
  - recommended for docs, demos, and new automation

- `iints data contract-run`
  - low-level compatibility alias

- `iints mdmp validate`
  - hidden legacy alias kept so older scripts do not break immediately

## Built-in Unit Transforms

When built-in transforms are enabled, current supported conversion is:
- `mmol/L <-> mg/dL` for numeric columns configured under `metadata.unit_conversions`

## Python API Usage

### Parse/compile contract

```python
from pathlib import Path
from iints.data import load_contract_yaml, compile_contract

contract = load_contract_yaml(Path("mdmp_contract.yaml"))
compiled = compile_contract(contract.to_dict())
print(compiled["fingerprint_sha256"])
```

### Run validation directly

```python
import pandas as pd
from iints.data import ContractRunner, load_contract_yaml

df = pd.read_csv("data/my_cgm.csv")
contract = load_contract_yaml(Path("mdmp_contract.yaml"))
result = ContractRunner(contract).run(df)
print(result.mdmp_grade, result.compliance_score)
```

### Enforce runtime gate with decorator

```python
import pandas as pd
from iints.data import mdmp_gate

@mdmp_gate("mdmp_contract.yaml", min_grade="research_grade", fail_mode="raise")
def process(df: pd.DataFrame) -> int:
    return len(df)
```

Fail modes:
- `raise`: block execution with `MDMPGateError`
- `warn`: continue and raise warning
- `log`: continue and log warning

## Synthetic Mirror: What It Preserves

`synthetic-mirror` creates a privacy-safe dataset by:
- sampling source rows with deterministic seed,
- adding controlled numeric noise,
- preserving required schema columns,
- clipping values to contract ranges,
- validating output again with MDMP.

Use it for:
- pipeline development without distributing raw sensitive rows,
- demo/test scenarios where schema realism matters.

## Recommended Audit Bundle

For each experiment, store together:
- `mdmp_report.json` (or `contract_data_report.json`)
- `mdmp_dashboard.html`
- `run_manifest.json`
- `run_metadata.json`
- model artifact metadata (seed, split, commit, config hash)

## Common Pitfalls

1. Missing required columns
- Fix contract vs dataset schema mismatch first.

2. Type mismatch (e.g., strings in numeric columns)
- Normalize upstream import/parsing before validation.

3. Rule expressions too complex
- Keep current grammar simple (`is null`, `is not null`, numeric comparisons, `and/or`).

4. Grade gate fails unexpectedly
- Check both compliance and score; `clinical_grade` requires compliance + >=90.

## Quick Decision Matrix

- Want fastest onboarding: use [MDMP Quickstart](MDMP_QUICKSTART.md)
- Want full semantics and edge cases: use this page
- Want concise protocol definition: use [MDMP Specification (Draft)](MDMP.md)

## Scope Reminder

MDMP strengthens data quality and reproducibility.
It is a research protocol component, not a clinical approval framework.
