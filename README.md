# IINTS-AF SDK (v0.1.3)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/quickstart_benchmark.ipynb)
[![Python Package CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
## Intelligent Insulin Titration System for Artificial Pancreas

### Overview
The IINTS-AF SDK is a hardware-agnostic Python framework designed for researchers and developers in the field of diabetes technology. It provides a standardized environment to design, simulate, and validate insulin delivery algorithms.

The framework is built around the **Dual-Guard Architecture**, ensuring that any controller—ranging from simple PID loops to complex neural networks—operates within a deterministic safety environment.

### Key Features
*   **Universal Compatibility**: Designed to run on any environment supporting Python 3.8+, from high-performance workstations to resource-constrained edge devices.
*   **Dual-Guard Safety**: Native integration of an `InputValidator` and `IndependentSupervisor` to mitigate AI hallucinations and prevent dangerous insulin dosing.
*   **Algorithm Flexibility**: Built-in support and templates for both classical control (PID) and modern AI (LSTM) architectures.
*   **Clinical Validation Tools**: Automated analysis of TIR (Time In Range) and other glycemic metrics using standardized research datasets like Ohio T1DM.
*   **Extensible Data Pipeline**: Tools for ingesting, cleaning, and normalizing diverse patient data formats.

### Installation

#### System Requirements
*   Python 3.8 or higher
*   Works on Windows, macOS, and Linux

#### From TestPyPI
```bash
pip install -i https://test.pypi.org/simple/ iints-sdk-python35
```

#### From Source (Development)
```bash
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python3 -m pip install -e .
python3 -m pip install -e ".[dev]"
```

### Full Manual
The complete SDK guide is available in `SDK_COMPREHENSIVE_GUIDE.md`.

### Quick Start

#### 1. Initialize a Project
Create a new research workspace with standard templates and data:

```bash
iints init --project-name my_research
cd my_research
```

#### 2. Execute a Basic Simulation
Run a standard PID-controller simulation on an Ohio T1DM patient profile:

```python
import iints

# Initialize simulator with independent supervisor
sim = iints.Simulator(algo="PID", patient="ohio_559")
sim.run(duration_minutes=1440) # Run for 24 hours

# Generate clinical report
sim.generate_report()
```

#### 2. Develop a Custom AI Algorithm
Use the CLI to generate a new template:

```bash
iints new-algo MySmartAlgo.py
```

Implement the logic within the `predict_insulin` method. The SDK automatically applies safety constraints such as IOB limits and maximum dose caps.

### Testing
Run the automated test suite:

```bash
pytest
```

### One-Command Dev Workflow
```bash
make dev
make test
make lint
```

### Helper Scripts
```bash
./scripts/run_tests.sh
./scripts/run_lint.sh
./scripts/run_demo.sh
```

### Performance Profiling
Enable high-precision timing for algorithm and supervisor latency:

```python
from iints.core.simulator import Simulator

sim = Simulator(patient_model=patient, algorithm=algo, enable_profiling=True)
results_df, safety_report = sim.run_batch(duration_minutes=1440)
print(safety_report["performance_report"])
```

### Audit Trail + PDF Report
Generate an audit trail (JSONL/CSV + summary) and a clean clinical PDF in one run:

```bash
python3 examples/audit_and_report.py
```

Notes:
* The PDF includes a safety summary plus top intervention reasons for explainability.
* The simulator stops automatically on sustained critical hypoglycemia (default: <40 mg/dL for 30 minutes).

### Clinic-Safe Presets (Quickstart)
Run a clinically safe preset with any algorithm:

```bash
iints presets list
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

Quickstart project in one command:

```bash
iints quickstart --project-name iints_quickstart
```

New presets:
* `hypo_prone_night`
* `hyper_challenge`
* `pizza_paradox`
* `midnight_crash`

Generate your own clinic-safe preset scaffold:

```bash
iints presets create --name custom_safe --output-dir ./presets
```
### Report + Validate CLI
Generate a report from a results CSV and validate scenarios before running:

```bash
iints report --results-csv results/data/sim_results_example.csv --output-path results/clinical_report.pdf
iints validate --scenario-path scenarios/example_scenario.json
```

Scenario files must include a `scenario_version` field for reproducibility.

Full report bundle (PDF + plots + audit):

```bash
iints report --results-csv results/data/sim_results_example.csv --bundle-dir results/report_bundle
```

Baseline comparison (auto-run PID + standard pump) is enabled by default for `iints run` and `iints presets run`.

Validate a patient config alongside a scenario:

```bash
iints validate --scenario-path scenarios/example_scenario.json --patient-config-path src/iints/data/virtual_patients/clinic_safe_baseline.yaml
```

Deterministic runs (seeded):

```bash
iints run --algo algorithms/example_algorithm.py --scenario-path scenarios/example_scenario.json --seed 42
```

One-line Python API:

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

### Core Concepts
*   **IndependentSupervisor**: A deterministic "digital cage" that validates every suggested dose against physiological safety limits.
*   **InputValidator**: Filters CGM noise and blocks physiologically impossible glucose excursions (e.g., sensor compression lows).
*   **Deterministic Audit**: Every decision is logged for full accountability and explainability (XAI).

### Strategic Roadmap
*   **February 2026**: Hardening of the Safety Engine and Documentation sprint.
*   **March 2026**: Monte Carlo population studies and Edge AI benchmarking.
*   **March 27, 2026**: Official Launch & Live Expo Demo.

### License & Ethics
This project is licensed under the MIT License.

> "Code shouldn't be a secret when it's managing a life." — Bobbaers Rune
