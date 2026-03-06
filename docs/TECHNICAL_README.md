# Developer CLI Guide

This document contains the full technical usage for the IINTS-AF SDK.

If you are new to the project, start with `docs/PLAIN_LANGUAGE_GUIDE.md` and `README.md` first.

## Who This Page Is For

- Engineers integrating SDK runs into applications or CI pipelines.
- Researchers needing exact commands and reproducible artifact expectations.
- Technical reviewers validating run traceability and data-quality gates.

## Terminology Used Consistently In This Page

- `Algorithm`: insulin-dosing logic under test.
- `Forecast model`: optional AI predictor signal (advisory only).
- `Safety Supervisor`: deterministic safety gate enforcing hard rules.
- `Run bundle`: output folder containing result traces + metadata + reports.
- `MDMP`: data contract validation protocol and grading system.

## Reading Structure

Workflow chapters are organized with:
- `Purpose`
- `When to use`
- `Commands`
- `Output`

## Documentation Site

Local preview:

```bash
python3 -m pip install mkdocs mkdocs-material
mkdocs serve
```

Static build:

```bash
mkdocs build
```

GitHub Actions deployment notes:
- Set repository Pages source to **GitHub Actions** once in repository settings.
- Set repository variable `ENABLE_PAGES_DEPLOY=true` to enable deploy job.

## What This File Is For
- Exact CLI commands.
- Integration and development workflows.
- Reproducible run artifacts and technical options.

## Installation

### System Requirements
* Python 3.10+
* Works on Windows, macOS, and Linux

### From TestPyPI
```bash
pip install -i https://test.pypi.org/simple/ iints-sdk-python35
```

### From Source (Development)
```bash
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python3 -m pip install -e .
python3 -m pip install -e ".[dev]"
```

## CLI Workflow

### Core Workflow Chapter A: Initialize a Project

**Purpose**
- Create a standard SDK workspace with expected folder structure.

**When to use**
- At the start of a new study, benchmark, or algorithm experiment.

**Commands**
```bash
iints init --project-name my_research
cd my_research
```

**Output**
- Project folders for algorithms, scenarios, and results.

### Core Workflow Chapter B: Baseline Simulation

**Purpose**
- Run a known-good baseline to verify end-to-end simulation behavior.

**When to use**
- After setup, before introducing custom algorithms.

**Commands**
```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

**Output**
- Initial run bundle with `results.csv`, report PDF, and audit logs.

### Core Workflow Chapter C: Study-Ready Bundle

**Purpose**
- Generate one reproducible package for review and validation.

**When to use**
- Before internal review, external sharing, or paper-support artifacts.

**Commands**
```bash
iints study-ready \
  --algo algorithms/example_algorithm.py \
  --output-dir results/study_ready
```

**Output**
- `results.csv`, `clinical_report.pdf`, `audit/`, `run_manifest.json`
- `validation_report.json`, `sources_manifest.json`, `SUMMARY.md`

### Core Workflow Chapter D: MDMP Data Validation

**Purpose**
- Validate dataset quality before training or evaluation.

**When to use**
- Whenever new raw CGM data is introduced into your pipeline.

**Commands**
```bash
iints mdmp template --output-path mdmp_contract.yaml
iints mdmp validate mdmp_contract.yaml data/my_cgm.csv \
  --output-json results/mdmp_report.json
iints mdmp visualizer results/mdmp_report.json \
  --output-html results/mdmp_dashboard.html
```

**Output**
- Contract validation report, MDMP grade, fingerprints, and HTML dashboard.

### Detailed Command Reference

### Initialize a Project
```bash
iints init --project-name my_research
cd my_research
```

### Quickstart Project
```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

### Run a Simulation
```bash
iints run --algo algorithms/example_algorithm.py \
  --scenario-path scenarios/example_scenario.json \
  --patient-config-name default_patient \
  --seed 42
```

Each run writes a reproducible bundle to `results/<run_id>/` by default:
- `config.json`
- `run_metadata.json`
- `run_manifest.json` (SHA-256 hashes for provenance)
- `results.csv`
- `report.pdf`
- `audit/` and `baseline/` (when enabled)

### One-Line Runner (CSV + audit + PDF + baseline + profiling)
```bash
iints run-full --algo algorithms/example_algorithm.py \
  --scenario-path scenarios/example_scenario.json \
  --patient-config-name default_patient \
  --output-dir results/run_full
```

### One-Line Research Bundle (run + validate + sources + summary)
```bash
iints study-ready \
  --algo algorithms/example_algorithm.py \
  --output-dir results/study_ready
```

Creates:
- `results.csv`, `clinical_report.pdf`, `audit/`, `run_manifest.json`
- `validation_report.json`
- `sources_manifest.json`
- `SUMMARY.md`

### Parallel Batch Runner
```bash
iints run-parallel --algo algorithms/example_algorithm.py \
  --scenarios-dir scenarios \
  --output-dir results/batch
```

### Scenario Generator
```bash
iints scenarios generate --name "Random Stress Test" \
  --output-path scenarios/generated_scenario.json
```

### Validate Scenario + Patient Config
```bash
iints validate --scenario-path scenarios/example_scenario.json \
  --patient-config-path src/iints/data/virtual_patients/clinic_safe_baseline.yaml
```

### Show Scientific Sources Used by the SDK
```bash
iints sources
iints sources --category guideline
iints sources --output-json results/source_manifest.json
```

### Import Real-World CGM Data
```bash
iints import-data --input-csv data/my_cgm.csv --output-dir results/imported
```

### Data Contract Runner (Model-Ready Gate)
```bash
iints data contract-template --output-path data_contract.yaml
iints data contract-run data_contract.yaml data/my_cgm.csv \
  --output-json results/contract_data_report.json
iints data contract-run data_contract.yaml data/my_cgm.csv \
  --min-mdmp-grade research_grade --fail-on-noncompliant
iints data synthetic-mirror data/my_cgm.csv data_contract.yaml \
  --output-csv data/synthetic_mirror.csv \
  --output-json results/synthetic_mirror_report.json
iints data mdmp-visualizer results/contract_data_report.json \
  --output-html results/mdmp_dashboard.html
iints mdmp template --output-path mdmp_contract.yaml
iints mdmp validate mdmp_contract.yaml data/my_cgm.csv \
  --output-json results/mdmp_report.json
iints mdmp synthetic-mirror data/my_cgm.csv mdmp_contract.yaml \
  --output-csv data/synthetic_mirror.csv \
  --output-json results/synthetic_mirror_report.json
iints mdmp visualizer results/mdmp_report.json \
  --output-html results/mdmp_dashboard.html
```
`contract-run` reports:
- `compliance_score`
- `contract_fingerprint_sha256`
- `dataset_fingerprint_sha256`
- `mdmp_grade` (`draft`, `research_grade`, `clinical_grade`)
- `certified_for_medical_research`

`mdmp-visualizer` generates a single self-contained HTML dashboard that can be reviewed offline by auditors and collaborators.

`synthetic-mirror` generates a synthetic dataset from a validated source CSV, preserving schema and broad numeric behavior, then validates the synthetic output against the same contract.

`iints mdmp ...` is the preferred protocol namespace for MDMP. `iints data ...` MDMP commands remain available for compatibility.

### MDMP Auto-Guardian Decorator
```python
import pandas as pd
from iints import mdmp_gate

@mdmp_gate("contracts/clinical_mdmp_contract.yaml", min_grade="clinical_grade")
def train_step(df: pd.DataFrame) -> int:
    return len(df)
```
You can also import from `iints.mdmp` for protocol-specific code boundaries.
Behavior:
- `fail_mode="raise"` (default): blocks execution with `MDMPGateError`
- `fail_mode="warn"`: continues with `RuntimeWarning`
- `fail_mode="log"`: continues and logs warning

### Clinical-Trial Scaffold
```bash
iints init --project-name iints_trial --template clinical-trial
```
This template creates:
- `contracts/clinical_mdmp_contract.yaml`
- `data/demo/diabetes_cgm.csv`
- `audit/`, `reports/`, `notebooks/`, `results/`

### Import Wizard (Interactive)
```bash
iints import-wizard
```

### Use the Demo Data Pack
```bash
iints import-demo --output-dir results/demo_import
```

### Nightscout Import (Optional Dependency)
```bash
pip install iints-sdk-python35[nightscout]
iints import-nightscout --url https://your-nightscout.example \
  --output-dir results/nightscout_import
```

### Tidepool Client Skeleton (Future Cloud Imports)
```bash
iints import-tidepool --base-url https://api.tidepool.org --token YOUR_TOKEN
```

### Demo Quickstart Workflow (Script)
```bash
python3 examples/demo_quickstart_flow.py
```

### Create a Patient Profile (YAML)
```bash
iints profiles create --name patient_john \
  --isf 45 --icr 11 --basal-rate 0.9 --initial-glucose 130 \
  --dawn-strength 8 --dawn-start 4 --dawn-end 8

# Use it in a run:
iints run --algo algorithms/example_algorithm.py \
  --patient-config-path patient_profiles/patient_john.yaml
```

### Generate a Report from Results CSV
```bash
iints report --results-csv results/data/sim_results_example.csv \
  --output-path results/clinical_report.pdf
```

## Research Track (AI Predictor)
See `research/README.md` for training and evaluation scripts. The predictor is not a dosing controller; it only provides a 30-120 minute forecast signal to the Safety Supervisor.

Quick start:
```bash
pip install iints-sdk-python35[research]
python research/synthesize_dataset.py --runs 10 --output data/synthetic.parquet
python research/train_predictor.py --data data/synthetic.parquet --config research/configs/predictor.yaml --out models
```

Integration:
```python
from iints.research import load_predictor_service
predictor = load_predictor_service("models/predictor.pt")
outputs = iints.run_simulation(
    algorithm=PIDController(),
    scenario="scenarios/example_scenario.json",
    predictor=predictor,
    duration_minutes=720,
)
```

### Dependency Check (Optional Torch)
```bash
pip install "iints[torch]"
iints check-deps
```

## Python API

### One-Line Runner
```python
import iints
from iints.core.algorithms.pid_controller import PIDController
from iints.core.patient.profile import PatientProfile

outputs = iints.run_simulation(
    algorithm=PIDController(),
    scenario="scenarios/example_scenario.json",
    patient_config="default_patient",
    duration_minutes=720,
    seed=42,
    output_dir="results/quick_run",
)

# Full bundle in one call
outputs = iints.run_full(
    algorithm=PIDController(),
    scenario="scenarios/example_scenario.json",
    patient_config="default_patient",
    duration_minutes=720,
    seed=42,
    output_dir="results/run_full",
)

# Patient profile shortcut
profile = PatientProfile(isf=45, icr=11, basal_rate=0.9, initial_glucose=130)
outputs = iints.run_simulation(
    algorithm=PIDController(),
    scenario="scenarios/example_scenario.json",
    patient_config=profile,
    duration_minutes=720,
    seed=42,
    output_dir="results/profile_run",
)

# SafetyConfig override
from iints.core.safety import SafetyConfig
safe = SafetyConfig(max_insulin_per_bolus=2.0, hypo_cutoff=80.0)
outputs = iints.run_full(
    algorithm=PIDController(),
    scenario="scenarios/example_scenario.json",
    patient_config="default_patient",
    duration_minutes=720,
    seed=42,
    output_dir="results/safe_run",
    safety_config=safe,
)
```

### Quickstart & Demo PDF Exports
```python
quickstart_pdf = iints.generate_quickstart_report(
    outputs["results"],
    "results/quickstart/quickstart_report.pdf",
    outputs["safety_report"],
)

demo_pdf = iints.generate_demo_report(
    outputs["results"],
    "results/quickstart/demo_report.pdf",
    outputs["safety_report"],
)
```

### Real-World Import (Python)
```python
import iints

result = iints.scenario_from_csv(
    "data/my_cgm.csv",
    data_format="dexcom",
    scenario_name="Patient A - Week 1",
)

result.dataframe.head()
scenario = result.scenario
```

Demo data in Python:
```python
import iints

demo_df = iints.load_demo_dataframe()
```

## Clinic-Safe Presets
```bash
iints presets list
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

New presets:
* `hypo_prone_night`
* `hyper_challenge`
* `pizza_paradox`
* `midnight_crash`

Create a scaffold:
```bash
iints presets create --name custom_safe --output-dir ./presets
```

## Audit Trail + Report Bundle
```bash
python3 examples/audit_and_report.py
```

Notes:
* The PDF includes top intervention reasons for explainability.
* The simulator stops on sustained critical hypoglycemia (default: <40 mg/dL for 30 minutes).
* When the limit is exceeded, `SimulationLimitError` is raised and the safety report marks `terminated_early`.

## Metrics
```python
import iints.metrics as metrics

gmi = metrics.calculate_gmi(results_df["glucose_actual_mgdl"])
lbgi = metrics.calculate_lbgi(results_df["glucose_actual_mgdl"])
```

## Human-in-the-loop + Sensor/Pump Models
```python
from iints import SensorModel, PumpModel
from iints.core.simulator import Simulator

sensor = SensorModel(noise_std=8.0, lag_minutes=5, dropout_prob=0.02, seed=42)
pump = PumpModel(max_units_per_step=0.25, quantization_units=0.05, dropout_prob=0.01, seed=42)

def rescue_callback(ctx):
    if ctx["glucose_actual_mgdl"] < 65:
        return {"additional_carbs": 15, "note": "rescue carbs"}
    return None

sim = Simulator(patient_model=patient, algorithm=algo, sensor_model=sensor, pump_model=pump, on_step=rescue_callback)
```

## State Serialization (Time‑travel Debugging)
```python
state = sim.save_state()
sim.load_state(state)
```

## Performance Profiling
```python
from iints.core.simulator import Simulator

sim = Simulator(patient_model=patient, algorithm=algo, enable_profiling=True)
results_df, safety_report = sim.run_batch(duration_minutes=1440)
print(safety_report["performance_report"])
```

## Mock Algorithms (CI-Safe)
```python
from iints import ConstantDoseAlgorithm, RandomDoseAlgorithm
```

## Testing
```bash
pytest
```

## One-Command Dev Workflow
```bash
make dev
make test
make lint
```

## Helper Scripts
```bash
./scripts/run_tests.sh
./scripts/run_lint.sh
./scripts/run_demo.sh
```

## Safety Architecture
* **IndependentSupervisor**: deterministic safety layer that caps insulin, blocks dangerous doses, and logs interventions.
* **InputValidator**: filters CGM noise and blocks physiologically impossible glucose values.
* **Deterministic Audit**: every decision is logged for accountability and explainability.

## Roadmap
* February 2026: Safety Engine hardening + documentation sprint
* March 2026: Monte Carlo population studies + edge AI benchmarking
* March 27, 2026: Official Launch & Live Expo Demo

## API Stability
See `API_STABILITY.md` for semver and deprecation policy.
