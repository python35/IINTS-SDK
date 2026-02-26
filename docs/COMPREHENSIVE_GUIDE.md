# IINTS-AF SDK: Comprehensive Guide

This document provides a complete, end-to-end manual for the IINTS-AF (Intelligent Insulin Titration System for Artificial Pancreas) SDK. It covers installation, project structure, key APIs, command-line interface usage, data formats, safety features, profiling, examples, and development workflows.

## 1. Introduction to IINTS-AF SDK

The IINTS-AF SDK is designed for researchers and developers to create, simulate, and benchmark insulin delivery algorithms for artificial pancreas systems. It provides a robust framework for patient simulation, data handling, and algorithm evaluation, alongside tools for documentation and continuous integration.

## 2. Installation

### 2.1 System Requirements

*   Python 3.10 or higher.
*   `pip` package manager.

### 2.2 Installing the SDK

To install the IINTS-AF SDK, you typically use `pip`. If you have received a distributable package (a `.whl` file), you can install it locally:

```bash
python3 -m pip install path/to/iints-0.1.0-py3-none-any.whl
```

If you wish to install from the project source in editable mode (for development):

```bash
python3 -m pip install -e .
```

This will install the `iints` package and its dependencies.

### 2.3 Dev Install (Recommended for Researchers)

```bash
python3 -m pip install -e ".[dev]"
```

This installs PyTest, Flake8, MyPy, and type stubs for a smooth developer workflow.

## 3. Project Structure

The core of the SDK's source code is located within the `src/iints/` directory. This structure ensures that all modules are properly packaged and discoverable when the SDK is installed.

```
src/
└── iints/
    ├── __init__.py             # Package initialization, re-exports key APIs
    ├── analysis/               # Clinical metrics, algorithm analysis, validation
    ├── api/                    # Core API for algorithms (InsulinAlgorithm, AlgorithmInput, etc.)
    ├── cli/                    # Command-Line Interface for the `iints` command
    ├── core/                   # Core simulation and patient logic
    │   ├── algorithms/         # PID, LSTM, hybrid, standard pump implementations
    │   ├── patient/            # Patient models and patient factory
    │   ├── safety/             # Safety supervisor and input validation
    │   └── simulator.py        # Main simulator engine
    ├── data/                   # Parsers, ingestors, data adapters
    ├── emulation/              # Commercial pump emulators
    ├── learning/               # Autonomous optimization and learning systems
    └── visualization/          # Visual dashboards and plots

examples/                        # End-to-end scripts and demos
tests/                           # PyTest suite
scenarios/                       # JSON scenario definitions
data_packs/                      # Data packs and schema docs
models/                          # Saved model artifacts (if any)
scripts/                         # Helper scripts (tests, lint, demo)
```

## 4. Key API Components

The SDK provides several key classes and functions for interaction. The most important ones are often re-exported directly in `iints/__init__.py` for easy access.

### `iints.api.base_algorithm.InsulinAlgorithm`

This is the abstract base class for all insulin delivery algorithms. Any custom algorithm you develop must inherit from this class and implement its abstract methods.

**Core Methods:**

*   `get_algorithm_metadata() -> AlgorithmMetadata`: Provides descriptive metadata for your algorithm.
*   `predict_insulin(data: AlgorithmInput) -> Dict[str, float]`: The main method where your algorithm calculates and returns insulin doses based on current patient data.
*   `reset()`: Resets the algorithm's internal state for a new simulation.

### `iints.core.simulator.Simulator`

The central class for running simulations. It orchestrates the patient model, algorithm, and safety supervisor over time.

**Core Methods:**

*   `run()`: Executes a single simulation run.
*   `run_batch()`: Executes a full batch simulation and returns results + safety report.
*   `add_stress_event()`: Adds predefined stress events (e.g., missed meal, exercise).
*   `enable_profiling=True`: Records algorithm, supervisor, and step latency.
*   `critical_glucose_threshold` / `critical_glucose_duration_minutes`: Automatically stops the simulation if glucose is critically low for too long (default: <40 mg/dL for 30 minutes).

### `iints.data.ingestor.DataIngestor`

Handles the loading and processing of various patient data formats.

### `iints.analysis.metrics.generate_benchmark_metrics`

A function to compute various clinical and performance metrics from simulation results.

## 5. Command-Line Interface (CLI)

The SDK provides a `iints` command-line tool for common tasks.

```bash
iints --help
```

**Commands (current CLI)**

1. `init`  
Creates a new research workspace with standard folders and an example algorithm + scenario.  

```bash
iints init --project-name my_iints_project
```

2. `new-algo`  
Creates a new algorithm template file from `iints.templates/default_algorithm.py`.  

```bash
iints new-algo --name MyAlgo --author "Your Name" --output-dir algorithms/
```

3. `run`  
Runs a simulation using a specific algorithm file and optional scenario.  

```bash
iints run \
  --algo algorithms/my_algo.py \
  --patient-config-name default \
  --scenario-path scenarios/example_scenario.json \
  --duration 720 \
  --time-step 5 \
  --output-dir ./results/data
```

4. `run-full`  
One-line runner that always exports results + audit + baseline + PDF.  
```bash
iints run-full --algo algorithms/my_algo.py \
  --scenario-path scenarios/example_scenario.json \
  --output-dir ./results/run_full
```

5. `run-parallel`  
Parallel batch runner for large scenario suites.  
```bash
iints run-parallel --algo algorithms/my_algo.py \
  --scenarios-dir scenarios \
  --output-dir ./results/batch
```

6. `scenarios generate` / `scenarios wizard`  
Random/custom scenario generator.  
```bash
iints scenarios generate --name "Random Stress Test" \
  --output-path scenarios/generated_scenario.json
```

7. `import-nightscout`  
Pull CGM data from Nightscout into a scenario + standard CSV (optional dependency).  
```bash
pip install iints-sdk-python35[nightscout]
iints import-nightscout --url https://your-nightscout.example \
  --output-dir results/nightscout_import
```

8. `import-tidepool`  
Initial Tidepool client skeleton (auth flow is TODO).  
```bash
iints import-tidepool --base-url https://api.tidepool.org --token YOUR_TOKEN
```

9. `benchmark`  
Benchmarks one AI algorithm against the standard pump across patient configs and scenarios.  

```bash
iints benchmark \
  --algo-to-benchmark algorithms/my_algo.py \
  --patient-configs-dir src/iints/data/virtual_patients \
  --scenarios-dir scenarios \
  --duration 720 \
  --time-step 5 \
  --output-dir ./results/benchmarks
```

5. `docs algo`  
Generates an auto-documentation panel for a specific algorithm file.  

```bash
iints docs algo --algo-path algorithms/my_algo.py
```

## 6. AI Research Track (Predictor)

The SDK includes an optional AI research pipeline for training a glucose **predictor**. The predictor is not a dosing controller; it provides a forecast that the Safety Supervisor can use as an early warning signal.

Key goals:
* Forecast glucose 30-120 minutes ahead
* Keep the Safety Supervisor deterministic
* Support synthetic bootstrap + real-world fine-tuning

Install research extras:
```bash
pip install iints-sdk-python35[research]
```

Training pipeline:
```bash
python research/synthesize_dataset.py --runs 25 --output data/synthetic.parquet
python research/train_predictor.py --data data/synthetic.parquet --config research/configs/predictor.yaml --out models
python research/evaluate_predictor.py --data data/synthetic.parquet --model models/predictor.pt
```

OhioT1DM preparation + training:
```bash
python research/prepare_ohio_t1dm.py \
  --input data_packs/public/ohio_t1dm \
  --output data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv

python research/train_predictor.py \
  --data data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv \
  --config research/configs/predictor_ohio_dual_guard.yaml \
  --out models/ohio_dual_guard
```
Tip: the bundled Ohio pack has only a few subjects. For a stronger model,
add more subjects and/or pretrain on synthetic/AZT1D, then fine‑tune on Ohio.

HUPA-UCM preparation:
```bash
python research/prepare_hupa_ucm.py \
  --input data_packs/public/hupa_ucm \
  --output data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv
```

AZT1D → HUPA fine‑tuning:
```bash
python research/train_predictor.py \
  --data data_packs/public/azt1d/processed/azt1d_merged.csv \
  --config research/configs/predictor_multimodal_dual_guard.yaml \
  --out models/pretrain_azt1d

python research/train_predictor.py \
  --data data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv \
  --config research/configs/predictor_multimodal_dual_guard_finetune.yaml \
  --warm-start models/pretrain_azt1d/predictor.pt \
  --out models/hupa_finetuned
```

ONNX export (edge/Jetson):
```bash
iints research export-onnx --model models/hupa_finetuned_v2/predictor.pt --out models/hupa_finetuned_v2/predictor.onnx
```

Integration:
```python
from iints.research import load_predictor_service
predictor = load_predictor_service("models/predictor.pt")
outputs = iints.run_simulation(..., predictor=predictor)
```

Documentation:
* `research/README.md`
* `research/model_card.md`
* `research/datasheet.md`

## 7. Quick Start (End-to-End)

```python
from iints.core.simulator import Simulator, StressEvent
from iints.core.patient.models import PatientModel
from iints.core.algorithms.pid_controller import PIDController

patient = PatientModel(initial_glucose=120)
algo = PIDController()
sim = Simulator(patient_model=patient, algorithm=algo, time_step=5, enable_profiling=True)

# Add a meal at 8:00 (in minutes)
sim.add_stress_event(StressEvent(start_time=8 * 60, event_type='meal', value=60))

results_df, safety_report = sim.run_batch(duration_minutes=24 * 60)
print(results_df.head())
print(safety_report.get("performance_report", {}))
```

## 8. Creating Custom Algorithms

1.  **Generate Template**: Use `iints new-algo YourAlgorithmName.py` to create a template.
2.  **Implement Logic**: Fill in your insulin delivery logic within the `predict_insulin` method of your new algorithm class, inheriting from `InsulinAlgorithm`.
3.  **Run**: Use `iints run --algorithm YourAlgorithmName ...` to test your algorithm in simulations.

## 9. Data Formats

The SDK expects patient data in a standardized format, often managed through `iints.data.ingestor.DataIngestor` and `iints.data.universal_parser.UniversalParser`. Key data points typically include timestamps, glucose readings, insulin doses, and carbohydrate intake. Specific details can be found in `data_packs/DATA_SCHEMA.md`.

Nightscout imports are supported via `iints import-nightscout` (requires the optional `nightscout` extra).

## 10. Safety & Clinical Guardrails

The SDK enforces safety constraints through two layers:

1. **InputValidator**: Filters biologically implausible glucose values and unsafe insulin requests.
2. **IndependentSupervisor**: Applies deterministic caps and overrides based on IOB and glucose state.

All safety limits are now centralized in `SafetyConfig` and can be set via CLI flags or Python API.

The safety report includes:
- Violation counts and breakdown
- Bolus interventions
- Recent safety events

### 10.1 Formal Safety Contract (Logic Validation)

Because the supervisor is deterministic, the SDK ships with a **formal safety contract** you can test exhaustively.
By default, insulin is inhibited when:

```
glucose < 90 mg/dL AND trend <= -5 mg/dL per 5 minutes
```

These thresholds are configurable in `SafetyConfig`:

```python
from iints.core.safety import SafetyConfig

config = SafetyConfig(
    contract_enabled=True,
    contract_glucose_threshold=90.0,
    contract_trend_threshold_mgdl_min=-1.0,  # -5 mg/dL per 5 minutes
)
```

Unit tests iterate across glucose/trend grids to prove no unsafe dose can pass this contract.

### 10.2 Red‑Team Scenarios (Chaos Testing)

For robust safety validation, the SDK includes chaos scenarios and stress algorithms:

* `StackingAIAlgorithm` + `chaos_insulin_stacking.json`
* `RunawayAIAlgorithm` + `chaos_runaway_ai.json`

Example CLI run:

```bash
iints run --algo algorithms/stacking_ai.py --scenario-path scenarios/chaos_insulin_stacking.json
```

### 10.3 SafetyEvent Callback (Trust API)

When the supervisor intervenes, you can capture the reason and display it to the user:

```python
def on_safety_event(event):
    print(
        f"AI requested {event['ai_requested_units']:.2f}U, "
        f"approved {event['supervisor_approved_units']:.2f}U "
        f"because: {event['safety_reason']}"
    )

sim = Simulator(patient_model=patient, algorithm=algo, on_safety_event=on_safety_event)
```

This makes the safety system **auditable and explainable** in demos or UIs.

### 10.4 Edge Efficiency (Battery & Resource Impact)

Safety checks are extremely lightweight. You can estimate energy per decision using:

```
Energy (J) = Power (W) × Time (s)
```

Example: at 5W and 0.002 ms per safety decision,
```
Energy ≈ 5 × 0.000002 = 0.00001 J (10 µJ)
```

This makes on‑device deployment feasible even on low‑power hardware. Use your device’s real power budget for a precise estimate.
The SDK is pure Python and runs on any device that can run Python 3.10+ (optional GPU for training only).

## 11. Precision Telemetry (Profiling)

Enable latency profiling on the simulator to measure:
- Algorithm inference latency
- Supervisor latency
- Full step latency

```python
sim = Simulator(patient_model=patient, algorithm=algo, enable_profiling=True)
results_df, safety_report = sim.run_batch(duration_minutes=1440)
print(safety_report["performance_report"])
```

## 12. Examples and Demos

All examples use `from iints...` imports. Run them directly:

```bash
python3 examples/main.py
python3 examples/run_final_analysis.py
```

Audit trail + PDF report example:

```bash
python3 examples/audit_and_report.py
```

Demo showcase script (researcher/clinician/company demo):

```bash
python3 examples/demo_showcase.py --output-dir results/demo_showcase
```

Paper-aligned Dual-Guard predictor configuration:

```bash
PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/public/azt1d/processed/azt1d_merged.csv \
  --config research/configs/predictor_paper_dual_guard.yaml \
  --out models/paper_dual_guard
```

## 13. Development Workflow

### 12.1 Versioning

The SDK uses semantic versioning. The current version is defined in `pyproject.toml`.

*   To update the version for a new release, edit the `version` field in `pyproject.toml` (e.g., from `0.1.0` to `0.1.1`).
*   It is good practice to use `git tag vX.Y.Z` to mark releases in your version control history.

### 12.2 Continuous Integration (CI) with GitHub Actions

A GitHub Actions workflow (`.github/workflows/python-package.yml`) has been set up to automate the build and testing process.

*   **Triggers**: The workflow runs automatically on `push` to the `main` branch and on every `pull_request`.
*   **Steps**: It checks out the code, sets up multiple Python versions, installs dependencies, builds the SDK, runs basic installation tests, performs linting with Flake8, and type checking with MyPy.
*   **Artifacts**: Built `.whl` and `.tar.gz` files are uploaded as artifacts, which you can download from the GitHub Actions run summary.

This ensures that every change to the codebase is automatically validated, preventing regressions and maintaining code quality.

### 12.3 Change Log

A `CHANGELOG.md` file has been created in the project root. It is recommended to update this file with a concise summary of changes for each new version, helping users understand what's new or fixed in each release.

## 14. Testing, Linting, Type Checking

```bash
python3 -m pytest
python3 -m flake8 .
python3 -m mypy src/iints
```

Or use the one-command flow:

```bash
make dev
make test
make lint
```

Helper scripts:

```bash
./scripts/run_tests.sh
./scripts/run_lint.sh
./scripts/run_demo.sh
```

## 15. Clinic-Safe Presets

List and run built-in presets:

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

## 16. Report + Validate CLI

Generate a report directly from a results CSV:

```bash
iints report --results-csv results/data/sim_results_example.csv --output-path results/clinical_report.pdf
```

Generate a full report bundle (PDF + plots + audit):

```bash
iints report --results-csv results/data/sim_results_example.csv --bundle-dir results/report_bundle
```

Baseline comparison (auto-run PID + standard pump) is enabled by default for `iints run` and `iints presets run`.

Validate scenario files before running:

```bash
iints validate --scenario-path scenarios/example_scenario.json
```

Scenario files must include a `scenario_version` field for reproducibility.

Validate scenario + patient config together:

```bash
iints validate --scenario-path scenarios/example_scenario.json --patient-config-path src/iints/data/virtual_patients/clinic_safe_baseline.yaml
```

Deterministic runs (seeded):

```bash
iints run --algo algorithms/example_algorithm.py --scenario-path scenarios/example_scenario.json --seed 42
```

Optional deep learning support:

```bash
pip install "iints[torch]"
iints check-deps
```

Mock algorithms (CI-safe, no Torch required):

```python
from iints import ConstantDoseAlgorithm, RandomDoseAlgorithm
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

Metrics module (GMI, CV, LBGI, HBGI):

```python
import iints.metrics as metrics

gmi = metrics.calculate_gmi(results_df["glucose_actual_mgdl"])
lbgi = metrics.calculate_lbgi(results_df["glucose_actual_mgdl"])
```

Human-in-the-loop + sensor/pump models:

```python
from iints.core.devices.models import SensorModel, PumpModel
from iints.core.simulator import Simulator

sensor = SensorModel(noise_std=8.0, lag_minutes=5, dropout_prob=0.02, seed=42)
pump = PumpModel(max_units_per_step=0.25, quantization_units=0.05, dropout_prob=0.01, seed=42)

def rescue_callback(ctx):
    if ctx["glucose_actual_mgdl"] < 65:
        return {"additional_carbs": 15, "note": "rescue carbs"}
    return None

sim = Simulator(patient_model=patient, algorithm=algo, sensor_model=sensor, pump_model=pump, on_step=rescue_callback)
```

State serialization (time-travel debugging):

```python
state = sim.save_state()
sim.load_state(state)
```

## 17. Audit Trail Export

Use the simulator to export audit trails and summaries:

```python
results_df, safety_report = sim.run_batch(duration_minutes=1440)
paths = sim.export_audit_trail(results_df, output_dir="results/audit")
print(paths)
```

## 18. Clinical PDF Reports

Generate a professional PDF report:

```python
from iints.analysis.reporting import ClinicalReportGenerator

generator = ClinicalReportGenerator()
generator.generate_pdf(results_df, safety_report, "results/clinical_report.pdf")
```

The report includes:
* Clinical metrics (TIR, time below/above range, CV, GMI)
* Safety summary and top intervention reasons
* Glucose and insulin delivery plots (insulin plot clamps at zero)

## 19. Troubleshooting

**ModuleNotFoundError: `src.*`**
- Use `from iints...` imports instead.
- Install in editable mode: `python3 -m pip install -e ".[dev]"`

**Flake8 fails on templates**
- The template directory is excluded in `.flake8` because it contains placeholders.

## 20. API Documentation

Comprehensive API documentation, generated using Sphinx, is available in HTML format.

*   **Location**: `docssphinx/_build/html/index.html`
*   **How to Build**: From the `docssphinx/` directory, run `make html`. Ensure your Python environment (or the `PYTHONPATH` if building manually) is correctly configured to include the project's `src` directory.

This documentation details all classes, methods, and functions within the `iints` package, extracted directly from their docstrings.
