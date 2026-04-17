# IINTS-AF SDK Documentation

IINTS-AF is a research SDK for testing insulin algorithms on virtual patients, validating data quality, and reviewing outputs with reproducible reports.

It combines three core workflows:
- **Simulate** closed-loop behavior on virtual patients
- **Certify** datasets and outputs with built-in trust grading
- **Review** runs with reports, posters, and optional local AI tooling

!!! important "Use a Virtual Environment"
    Run SDK commands from an active Python virtual environment such as `.venv`.
    This avoids package conflicts and missing dependency issues.

## Start Here

If you are new to the SDK, read these pages in order:

1. [Overview In Plain Language](PLAIN_LANGUAGE_GUIDE.md)
2. [Quick Start](GETTING_STARTED.md)
3. [Installation](INSTALLATION.md)
4. [CLI & Advanced Reference](TECHNICAL_README.md)

## Choose Your Path

| Goal | Best page | What you get |
|---|---|---|
| Understand the scope and terminology | [Overview In Plain Language](PLAIN_LANGUAGE_GUIDE.md) | A simple explanation of what the SDK does and does not do |
| Install and complete a first run | [Quick Start](GETTING_STARTED.md) | A reliable first workflow from install to outputs |
| Pick the right install profile | [Installation](INSTALLATION.md) | Package options, paths, extras, and environment checks |
| Run on Raspberry Pi or another SBC | [Edge Hardware & SBC Matrix](EDGE_HARDWARE.md) | Hardware guidance, edge install profile, and deployment choices |
| Operate a persistent digital patient | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) | Runtime control, kiosk view, export, and service setup |
| Bring a Pi to Maker Faire | [Maker Faire Pi Mode](MAKERFAIRE_PI.md) | One-command startup, reset routine, kiosk flow, and booth-safe recovery |
| Build reproducible experiments | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) | Protocols, corruption modes, study comparisons, and summaries |
| Explore the full CLI surface | [CLI & Advanced Reference](TECHNICAL_README.md) | Command reference and advanced workflows |

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

Typical outputs:
- `results.csv`
- `clinical_report.pdf`
- `audit/`
- `run_manifest.json`
- `certification.json`

If you are unsure which directory to use, start with [Installation](INSTALLATION.md).

## Common Workflows

### Simulation + Certification + AI Review

Use this when you want a complete baseline workflow:

1. `iints presets run`
2. `iints data certify`
3. `iints ai report`

### Study And Evidence Workflows

Use these pages when you want reproducible multi-run comparisons:

- [Study Analysis](STUDY_ANALYSIS.md)
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
- [Evidence Base](EVIDENCE_BASE.md)

### Edge And SBC Workflows

Use these pages when the SDK needs to stay running on a device:

- [Edge Hardware & SBC Matrix](EDGE_HARDWARE.md)
- [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md)

Core edge commands:
- `iints edge setup`
- `iints edge status`
- `iints edge bundle`

## Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
