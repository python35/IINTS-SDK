# IINTS-AF SDK Documentation

IINTS-AF is a safety-first research SDK for insulin algorithm simulation, validation, and audit-ready reporting.

!!! important "Use a Virtual Environment"
    Always run SDK commands from an active Python virtual environment (`.venv`).
    This avoids package conflicts and missing dependency issues.

## Start Here

- [Quick Start](GETTING_STARTED.md)
- [Installation And Paths](INSTALLATION.md)
- [Plain Language Overview](PLAIN_LANGUAGE_GUIDE.md)
- [Documentation Map](DOCUMENTATION_INDEX.md)
- [AI Assistant Guide](AI_ASSISTANT.md)
- [MDMP Quickstart](MDMP_QUICKSTART.md)
- [MDMP Full Guide](MDMP_FULL_GUIDE.md)
- [Demos (GitHub)](https://github.com/python35/IINTS-SDK/tree/main/examples/demos)

## Choose Your Path

| New to IINTS | Install Correctly | Build First Run | Engineering Reference |
|---|---|---|---|
| [Plain Language Overview](PLAIN_LANGUAGE_GUIDE.md) | [Installation And Paths](INSTALLATION.md) | [Quick Start](GETTING_STARTED.md) | [Developer CLI Guide](TECHNICAL_README.md) |
| Understand what the SDK does and does not do. | Know which folder to use and which install path fits your setup. | Install, run baseline, inspect outputs. | Full command reference and technical integration details. |

For the local research assistant:
- [AI Assistant Guide](AI_ASSISTANT.md)

## 10-Minute Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[mdmp]"

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

If you are unsure which folder these commands should run from, start with [Installation And Paths](INSTALLATION.md).

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
