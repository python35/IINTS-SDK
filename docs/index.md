# IINTS-AF SDK Documentation

IINTS-AF is a safety-first research SDK for insulin algorithm simulation, validation, and audit-ready reporting.

## Start Here

- [Quick Start](GETTING_STARTED.md)
- [Plain Language Overview](PLAIN_LANGUAGE_GUIDE.md)
- [Documentation Map](DOCUMENTATION_INDEX.md)
- [MDMP Quickstart](MDMP_QUICKSTART.md)
- [Demos (GitHub)](https://github.com/python35/IINTS-SDK/tree/main/examples/demos)

## 10-Minute Quick Start

```bash
pip install iints-sdk-python35

iints doctor --smoke-run
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

Expected outputs:
- `results.csv`
- `clinical_report.pdf`
- `audit/`
- `run_manifest.json`

## MDMP in 60 Seconds

MDMP is the IINTS data-quality protocol.

- `Contract`: required columns, types, units, and value bounds.
- `Validation`: dataset is checked against contract rules.
- `Grading`: output receives `draft`, `research_grade`, or `clinical_grade`.
- `Fingerprinting`: deterministic hashes support reproducibility and audits.

Use MDMP with:

```bash
iints mdmp template --output-path mdmp_contract.yaml
iints mdmp validate mdmp_contract.yaml data/my_cgm.csv --output-json results/mdmp_report.json
iints mdmp visualizer results/mdmp_report.json --output-html results/mdmp_dashboard.html
```

## Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
