# IINTS-AF SDK Documentation

IINTS-AF is one platform for insulin-algorithm research.

It combines three layers:
- **Simulate** closed-loop behavior on virtual patients
- **Certify** datasets and outputs with built-in trust grading
- **Understand** results with reports, posters, and local AI review

!!! important "Use a Virtual Environment"
    Always run SDK commands from an active Python virtual environment (`.venv`).
    This avoids package conflicts and missing dependency issues.

## Start Here

- [Quick Start](GETTING_STARTED.md)
- [Installation And Paths](INSTALLATION.md)
- [Edge Hardware Profiles](EDGE_HARDWARE.md)
- [Digital Patient On Raspberry Pi](DIGITAL_PATIENT_PI.md)
- [Study Analysis](STUDY_ANALYSIS.md)
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
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

For a live Raspberry Pi installation:
- [Digital Patient On Raspberry Pi](DIGITAL_PATIENT_PI.md)
- [Edge Hardware Profiles](EDGE_HARDWARE.md)

Core edge commands:
- `iints edge setup`
- `iints edge status`
- `iints edge bundle`

For the local research assistant:
- [AI Assistant Guide](AI_ASSISTANT.md)
- [Study Analysis](STUDY_ANALYSIS.md)
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)

## 10-Minute Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"

iints doctor --smoke-run
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
iints data certify contracts/clinical_mdmp_contract.yaml results/<run_id>/results.csv --output-json results/<run_id>/certification.json
iints ai report results/<run_id>
```

Expected outputs:
- `results.csv`
- `clinical_report.pdf`
- `audit/`
- `run_manifest.json`

If you are unsure which folder these commands should run from, start with [Installation And Paths](INSTALLATION.md).

## Data Certification in 60 Seconds

Data certification is the trust layer inside IINTS.

- `Contract`: required columns, types, units, and value bounds.
- `Validation`: dataset is checked against contract rules.
- `Grading`: output receives `draft`, `research_grade`, or `clinical_grade`.
- `Fingerprinting`: deterministic hashes support reproducibility and audits.

Use it with:

```bash
iints data certify-template --output-path data_contract.yaml
iints data certify data_contract.yaml data/my_cgm.csv --output-json results/certification.json
iints data certify-visualizer results/certification.json --output-html results/mdmp_dashboard.html
```

## Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
