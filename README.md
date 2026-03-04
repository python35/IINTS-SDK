# IINTS-AF SDK
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/00_Quickstart.ipynb)
[![Python Package CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Coverage](https://raw.githubusercontent.com/python35/IINTS-SDK/main/badges/coverage.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/health-badges.yml)
[![Docs Coverage](https://raw.githubusercontent.com/python35/IINTS-SDK/main/badges/docs.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/health-badges.yml)
[![Site](https://img.shields.io/badge/site-IINTS--AF-0a66c2?style=flat&logo=firefox-browser&logoColor=white)](https://python35.github.io/IINTS-Site/index.html)

<div style="text-align:center;">
  <img src="Ontwerp zonder titel.png" alt="" style="display:block; margin:0 auto;">
</div>

## Intelligent Insulin Titration System for Artificial Pancreas

IINTS-AF is a **safety-first simulation and validation platform** for insulin dosing algorithms. It lets you test AI or classical controllers on virtual patients, enforce deterministic safety constraints, and generate audit-ready clinical reports before anything touches a real patient.

**In one session you can**:
* Run a clinic-safe preset and compare against PID and standard pump baselines
* Import real-world CGM CSV into a standard schema + scenario JSON
* Use the bundled demo CGM data pack (zero setup)
* Export a clean PDF report plus full audit trail (JSONL/CSV)
* Stress-test sensor noise, pump limits, and human-in-the-loop interventions
* Generate patient profiles with ISF/ICR + dawn phenomenon

**Who it’s for**:
* Diabetes researchers and clinicians validating new control strategies
* ML engineers benchmarking AI controllers with medical safety rails
* Developers building decision-support systems for closed-loop insulin delivery

## From Black Box To Open Logic

IINTS-AF uses a **Dual-Guard Security Architecture**: AI is advisory, deterministic safety logic is authoritative.

**Layer 1 — InputValidator (Validation)**  
Biological plausibility filter on incoming glucose signals.  
It rejects or fail-soft clamps implausible values/rates so sensor artifacts do not propagate into control logic.

**Layer 2 — Intelligence (Forecasting)**  
LSTM predictor models hidden glucose/insulin dynamics and provides future glucose forecasts (e.g., 30-120 min horizon).  
These outputs are advisory signals for safety assessment, not direct therapy commands.

**Layer 3 — Independent Supervisor (Supervision)**  
Deterministic safety layer validates every proposed dose against hard constraints (IOB caps, hypo prevention, trend checks, contract rules).  
Supervisor decisions override algorithm outputs when risk is detected and are recorded in the audit trail.

This architecture is designed to be auditable and explainable for research reviews, clinical discussions, and edge deployment demonstrations.

## Installation

Install the SDK directly via PyPI:

```bash
pip install iints-sdk-python35
```

### Quick Start (CLI)
```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

One-line full run (CSV + audit + PDF + baseline):
```bash
iints run-full --algo algorithms/example_algorithm.py \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --output-dir results/run_full
```
By default, runs write to `results/<run_id>/` and include `config.json`, `run_metadata.json`, and `run_manifest.json`.

Dual-Guard AI run (paper architecture: AI proposal + deterministic supervisor):
```bash
iints run-full \
  --algo algorithms/example_algorithm.py \
  --predictor models/hupa_finetuned_v2/predictor.pt \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --output-dir results/dual_guard
```
This keeps control deterministic: predictor signals are advisory only, final dosing remains supervisor-constrained.

Import real-world CGM data:
```bash
iints import-data --input-csv data/my_cgm.csv --output-dir results/imported
```

Try the bundled demo data pack:
```bash
iints import-demo --output-dir results/demo_import
```

Official real-world datasets (download on demand):
```bash
iints data list
iints data info aide_t1d
iints data fetch aide_t1d
iints data cite aide_t1d
```
Some datasets require approval and are marked as `request` in the registry.
`iints data info` prints BibTeX + citation text for easy referencing.

Offline sample dataset (no download required):
```bash
iints data fetch sample --output-dir data_packs/sample
```

Nightscout import (optional dependency):
```bash
pip install iints-sdk-python35[nightscout]
iints import-nightscout --url https://your-nightscout.example --output-dir results/nightscout_import
```

Scenario generator:
```bash
iints scenarios generate --name "Random Stress Test" --output-path scenarios/generated_scenario.json
iints scenarios migrate --input-path scenarios/legacy.json
```

Parallel batch runs:
```bash
iints run-parallel --algo algorithms/example_algorithm.py --scenarios-dir scenarios --output-dir results/batch
```

Interactive run wizard:
```bash
iints run-wizard
```

Developer health + validation gates:
```bash
iints doctor --smoke-run
iints validation-profiles
iints validate-run --results-csv results/run_full/results.csv --profile research_default
iints contract-verify --output-json results/contract_report.json
iints certify-run --algo algorithms/example_algorithm.py --profile strict_safety --output-dir results/certified
iints scorecard --algo algorithms/example_algorithm.py --profile research_default --output-dir results/scorecard
```

Algorithm registry:
```bash
iints algorithms list
iints algorithms info "PID Controller"
```

Or run the full demo workflow (import + run + report) in one script:
```bash
python3 examples/demo_quickstart_flow.py
```

### Quick Start (Python)
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

### Notebook Guide
Hands-on Jupyter notebooks live in [`examples/notebooks/`](examples/notebooks/)  


* Quickstart end-to-end run
* Presets + scenario validation
* Safety supervisor behavior
* Audit trail + PDF report export
* Baseline comparison + clinical metrics
* Sensor/pump models + human-in-the-loop
* Optional Torch/LSTM usage
* Ablation study (with/without Supervisor)

### AI Research Track (Predictor)
IINTS-AF includes an optional research pipeline to train a glucose **predictor** that feeds the Safety Supervisor with a 30-120 minute forecast. The predictor never doses insulin; it only provides a forecast signal.

Install research extras:
```bash
pip install iints-sdk-python35[research]
```

Train a starter predictor:
```bash
python research/synthesize_dataset.py --runs 25 --output data/synthetic.parquet
python research/train_predictor.py --data data/synthetic.parquet --config research/configs/predictor.yaml --out models
```

Audit data leakage and forecast calibration:
```bash
iints research audit-split --data data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv
iints research evaluate-forecast --input-csv results/dual_guard/results.csv
iints research parity-check --model models/hupa_finetuned_v2/predictor.pt --onnx models/hupa_finetuned_v2/predictor.onnx
```

Integrate:
```python
from iints.research import load_predictor_service
predictor = load_predictor_service("models/predictor.pt")
outputs = iints.run_simulation(..., predictor=predictor)
```

### Documentation
* PDF manual: `docs/manuals/IINTS-AF_SDK_Manual.pdf`
* Manual source: `docs/manuals/IINTS-AF_SDK_Manual.md`
* Comprehensive guide: `docs/COMPREHENSIVE_GUIDE.md`
* Notebook index: `examples/notebooks/README.md`
* Technical README: `docs/TECHNICAL_README.md`
* API Stability: `API_STABILITY.md`
* Research track: `research/README.md`

### Related Work & Inspiration
We borrow techniques from the broader CGM/APS ecosystem, while differentiating with a safety‑first, audit‑ready workflow:
* [simglucose (UVA/Padova)](https://github.com/jxx123/simglucose): gymnasium‑style interfaces and parallel batch execution concepts.
* [OpenAPS / oref0](https://github.com/openaps/oref0): gold‑standard IOB logic and safety‑oriented control patterns.
* [Nightscout](https://github.com/nightscout/cgm-remote-monitor) + [py-nightscout](https://pypi.org/project/py-nightscout/): reference for human‑in‑the‑loop CGM ingest (planned connector).
* [Tidepool OpenAPI](https://developer.tidepool.org/TidepoolApi/): basis for a future cloud import client skeleton.

### Ethics & Safety
This SDK is for **research and validation**. It is not a medical device and does not provide clinical dosing advice.

> “Code shouldn’t be a secret when it’s managing a life.” — Bobbaers Rune
