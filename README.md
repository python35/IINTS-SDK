# IINTS-AF SDK
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![Python Package CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Site](https://img.shields.io/badge/site-IINTS--AF-0a66c2?style=flat&logo=firefox-browser&logoColor=white)](https://python35.github.io/IINTS-Site/index.html)

IINTS-AF is a safety-first simulation SDK for insulin algorithm research.
It helps you test controllers, validate safety behavior, and generate audit-ready artifacts.

New here? Start with: [docs/PLAIN_LANGUAGE_GUIDE.md](https://github.com/python35/IINTS-SDK/blob/main/docs/PLAIN_LANGUAGE_GUIDE.md)

## What You Get
- Virtual patient simulation for insulin strategies.
- Deterministic safety supervision (hard rules).
- Optional AI glucose predictor as advisory signal.
- Audit outputs: CSV, JSON, PDF reports.
- Validation and scorecard tooling for repeatable benchmarks.
- Transparent evidence manifest linked to peer-reviewed sources (`iints sources`).

## Open Logic (Dual-Guard)
IINTS-AF is built as layered logic, not black-box control:

1. `InputValidator`  
   Filters biologically implausible glucose input and applies fail-soft fallback.
2. `LSTM Predictor` (optional)  
   Forecasts future glucose; advisory only.
3. `IndependentSupervisor`  
   Deterministically validates every dose request and overrides unsafe actions.

## Install
```bash
pip install iints-sdk-python35
```

Research extras:
```bash
pip install iints-sdk-python35[research]
```

## Quick Start (CLI)
```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

One full run with artifacts:
```bash
iints run-full \
  --algo algorithms/example_algorithm.py \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --output-dir results/run_full
```

Dual-Guard run with predictor:
```bash
iints run-full \
  --algo algorithms/example_algorithm.py \
  --predictor models/hupa_finetuned_v2/predictor.pt \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --output-dir results/dual_guard
```

## Quick Start (Python)
```python
import iints
from iints.core.algorithms.pid_controller import PIDController

outputs = iints.run_simulation(
    algorithm=PIDController(),
    scenario="scenarios/example_scenario.json",
    patient_config="default_patient",
    duration_minutes=720,
    seed=42,
    output_dir="results/quick_run",
)
```

## Most Used Commands
```bash
# Environment and validation checks
iints doctor --smoke-run
iints study-ready --algo algorithms/example_algorithm.py --output-dir results/study_ready
iints validation-profiles
iints validate-run --results-csv results/run_full/results.csv --profile research_default
iints contract-verify --output-json results/contract_report.json
iints data contract-template --output-path data_contract.yaml
iints data contract-run data_contract.yaml data/my_cgm.csv --output-json results/contract_data_report.json
iints data contract-run data_contract.yaml data/my_cgm.csv --min-mdmp-grade research_grade --fail-on-noncompliant
iints data mdmp-visualizer results/contract_data_report.json --output-html results/mdmp_dashboard.html
iints replay-check --algo algorithms/example_algorithm.py --output-json results/replay_check.json
iints golden-benchmark --algo algorithms/example_algorithm.py --output-json results/golden_benchmark.json

# Scenario bank benchmark
iints scorecard --algo algorithms/example_algorithm.py --profile research_default --output-dir results/scorecard

# Research diagnostics
iints research audit-split --data data_packs/public/OhioT1DM/processed/ohio_merged.csv
iints research evaluate-forecast --input-csv results/dual_guard/results.csv
iints research evaluate-forecast --input-csv results/dual_guard/results.csv --gate-profile research_default
iints research registry-list --registry models/registry.json
iints research registry-promote --registry models/registry.json --run-id <run-id> --stage validated
```

`iints study-ready` produces a complete bundle by default:
- simulation outputs
- `validation_report.json`
- `sources_manifest.json`
- `SUMMARY.md` (human-readable quick review)

`iints data contract-run` writes deterministic fingerprints plus an MDMP grade:
- `draft`
- `research_grade`
- `clinical_grade`

`iints data mdmp-visualizer` turns a contract report JSON into a shareable single-file HTML dashboard for audit review.

## Demos and Notebooks
- Demo scripts: [examples/demos](https://github.com/python35/IINTS-SDK/tree/main/examples/demos)
- Notebook tutorials: [examples/notebooks](https://github.com/python35/IINTS-SDK/tree/main/examples/notebooks)

## Documentation
- Manual (PDF): [docs/manuals/IINTS-AF_SDK_Manual.pdf](https://github.com/python35/IINTS-SDK/blob/main/docs/manuals/IINTS-AF_SDK_Manual.pdf)
- Comprehensive guide: [docs/COMPREHENSIVE_GUIDE.md](https://github.com/python35/IINTS-SDK/blob/main/docs/COMPREHENSIVE_GUIDE.md)
- Technical README: [docs/TECHNICAL_README.md](https://github.com/python35/IINTS-SDK/blob/main/docs/TECHNICAL_README.md)
- Plain-language guide: [docs/PLAIN_LANGUAGE_GUIDE.md](https://github.com/python35/IINTS-SDK/blob/main/docs/PLAIN_LANGUAGE_GUIDE.md)
- MDMP (draft): [docs/MDMP.md](https://github.com/python35/IINTS-SDK/blob/main/docs/MDMP.md)
- Evidence base (peer-reviewed sources): [docs/EVIDENCE_BASE.md](https://github.com/python35/IINTS-SDK/blob/main/docs/EVIDENCE_BASE.md)
- Research track: [research/README.md](https://github.com/python35/IINTS-SDK/blob/main/research/README.md)

## Evidence and Provenance
The SDK ships a source manifest grounded in peer-reviewed diabetes literature and standards.

```bash
iints sources
iints sources --category trial --output-json results/source_manifest.json
```

## Safety Notice
This SDK is for research and validation only.  
It is not a medical device and does not provide clinical dosing advice.
