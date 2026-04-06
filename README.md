# IINTS-AF SDK
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![Python Package CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-IINTS--AF-0a66c2?style=flat&logo=firefox-browser&logoColor=white)](https://python35.github.io/IINTS-SDK/)

One platform for insulin-algorithm research.

IINTS-AF combines three layers in one SDK:
- **Simulate** insulin algorithms on virtual patients
- **Certify** the data and outputs with built-in trust grading
- **Understand** results with reports, posters, and local AI review

Docs: [python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"
```

For Raspberry Pi or UNO Q edge rigs, use `iints-sdk-python35[edge,mdmp]` and follow `docs/EDGE_HARDWARE.md`.

Edge workflow:
```bash
iints edge setup --output-dir iints_edge_demo --board raspberry_pi
iints edge status --workspace iints_edge_demo/patient_runtime
iints edge bundle --workspace iints_edge_demo/patient_runtime --output results/edge_runtime_bundle.zip
```

Sanity check:
```bash
iints doctor --smoke-run
```

## Quick Flow
```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
iints data certify contracts/clinical_mdmp_contract.yaml results/<run_id>/results.csv --output-json results/<run_id>/certification.json
iints ai report results/<run_id>
```

## Read Next
- Start here: `docs/GETTING_STARTED.md`
- Installation and paths: `docs/INSTALLATION.md`
- Edge hardware profiles: `docs/EDGE_HARDWARE.md`
- Raspberry Pi digital patient: `docs/DIGITAL_PATIENT_PI.md`
- Study analysis: `docs/STUDY_ANALYSIS.md`
- AI assistant: `docs/AI_ASSISTANT.md`
- Data certification: `docs/MDMP_QUICKSTART.md`
- Full manual: `docs/manuals/IINTS-AF_SDK_Manual.md`

## Important
IINTS-AF is research software. It is not a medical device and does not provide clinical treatment advice.
