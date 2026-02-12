# IINTS-AF SDK
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/00_Quickstart.ipynb)
[![Python Package CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)

<div style="text-align: center; margin-top: -10px; margin-bottom: -10px;">
  <img src="Ontwerp zonder titel.png" alt="" style="display: block; margin: 0 auto;">
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

Import real-world CGM data:
```bash
iints import-data --input-csv data/my_cgm.csv --output-dir results/imported
```

Try the bundled demo data pack:
```bash
iints import-demo --output-dir results/demo_import
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
Hands-on Jupyter notebooks live in `examples/notebooks/` and cover the full workflow (each is Colab-ready):

* Quickstart end-to-end run
* Presets + scenario validation
* Safety supervisor behavior
* Audit trail + PDF report export
* Baseline comparison + clinical metrics
* Sensor/pump models + human-in-the-loop
* Optional Torch/LSTM usage
* Ablation study (with/without Supervisor)

### Documentation
* Product manual: `SDK_COMPREHENSIVE_GUIDE.md`
* Notebook index: `examples/notebooks/README.md`
* Technical README: `TECHNICAL_README.md`
* API Stability: `API_STABILITY.md`

### Ethics & Safety
This SDK is for **research and validation**. It is not a medical device and does not provide clinical dosing advice.

> “Code shouldn’t be a secret when it’s managing a life.” — Bobbaers Rune
