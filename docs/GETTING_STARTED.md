# Getting Started

Use this page for the fastest reliable path from installation to a first complete SDK run.

**Before this page:** [Quickstart](QUICKSTART.md) if you have not run `iints demo` yet.

**After this page:** [Scientific Workflow](SCIENTIFIC_WORKFLOW.md), [MDMP Quickstart](MDMP_QUICKSTART.md), or [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md), depending on your goal.

The core workflow is:

1. run a simulation
2. certify the output data
3. review the run report

If you mainly need help choosing folders, install extras, or repository vs package usage, read [Installation](INSTALLATION.md) first.

## 1) Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

All commands below assume this `.venv` is active.

## 2) Install The SDK

```bash
python -m pip install -U "iints-sdk-python35[full,mdmp]"
```

Optional extras:

```bash
pip install "iints-sdk-python35[research]"
pip install "iints-sdk-python35[nightscout]"
pip install "iints-sdk-python35[edge,mdmp]"
```

## 3) Verify The Environment

```bash
iints doctor --smoke-run
```

If this fails, fix the environment before starting longer runs.

## 4) Create A Project

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
```

The generated structure includes:
- `algorithms/`
- `scenarios/`
- `contracts/`
- `data/demo/`
- `audit/`
- `results/`

Important:
- before `iints quickstart`, commands can be run from any folder
- after `iints quickstart`, move into the generated project folder
- repository helper scripts such as `./scripts/run_live_stage_demo.sh` belong to the SDK repository, not the quickstart project
- custom algorithms can be installed later with `iints plugin install algorithms/my_algo.py`

## 5) Run A Baseline Simulation

```bash
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

## 6) Certify The Run Data

Use the generated results CSV plus the project contract:

```bash
iints data certify \
  contracts/clinical_mdmp_contract.yaml \
  results/<run_id>/results.csv \
  --output-json results/<run_id>/certification.json
```

Optional dashboard:

```bash
iints data certify-visualizer \
  results/<run_id>/certification.json \
  --output-html results/<run_id>/certification_dashboard.html
```

## 7) Review The Run With The Local AI Layer

```bash
iints ai report results/<run_id>
```

That gives the main SDK workflow in three commands:

1. `iints presets run`
2. `iints data certify`
3. `iints ai report`

## 8) Inspect The Outputs

A typical run writes:
- `results.csv`: time-series simulation output
- `clinical_report.pdf`: summary report for review
- `audit/`: safety and decision trail
- `run_manifest.json`: file hashes for reproducibility
- `run_metadata.json`: run configuration, environment details, SDK version, and data-format versions
- `certification.json`: trust grade and dataset checks after `iints data certify`

## 9) Build A Study-Ready Bundle

```bash
iints study-ready --algo algorithms/example_algorithm.py --output-dir results/study_ready
```

This adds:
- `validation_report.json`
- `sources_manifest.json`
- `SUMMARY.md`

## 10) Common Next Steps

### Import Personal Pump / CGM Data

```bash
iints import-carelink \
  --input-csv "/path/to/CareLink export.csv" \
  --output-dir results/imported_carelink
```

Or build the full personal-data workspace at once:

```bash
iints carelink-workbench \
  --input-csv "/path/to/CareLink export.csv" \
  --output-dir results/personal_carelink
```

### Enable The Optional Local AI Assistant

```bash
iints ai models
ollama pull ministral-3:8b
iints ai local-check --model ministral-3:8b
iints ai prepare results/<run_id>
iints ai report results/<run_id>
```

If `iints ai local-check` reports that Ollama closed the connection, the most likely causes are a restarting daemon or insufficient memory. In that case, try `ministral-3:3b`.

### Build A Poster From Existing Run Bundles

```bash
iints poster \
  --run-dir results/normal_run \
  --run-dir results/meal_stress \
  --run-dir results/supervisor_override \
  --label "Normal Run" \
  --label "Meal Stress Test" \
  --label "Supervisor Override" \
  --output-path results/posters/iints_results_poster.png
```

### Run A Prepared Presentation Demo

```bash
./scripts/run_live_stage_demo.sh
```

This is a convenient repository wrapper when you want a ready-made live walkthrough with code, outputs, and presentation assets.

If the machine only has the installed SDK and not the repository checkout, export the same demo code first:

```bash
iints demo-export --output-dir iints_demo
cd iints_demo
python 07_live_stage_demo.py
```

### Run A Persistent Digital Patient On Raspberry Pi

```bash
iints quickstart --project-name iints_pi_demo
cd iints_pi_demo
iints patient scenarios
iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile normal_day \
  --mode demo-time \
  --speed 60x
iints patient status --workspace patient_runtime
```

That flow gives you:
- a persistent SQLite-backed runtime
- a live dashboard on `http://127.0.0.1:8765/dashboard`
- a run-like bundle under `patient_runtime/live_bundle/`

Security default:

- the dashboard stays on loopback by default
- for remote presentation, prefer Raspberry Pi Connect instead of opening the API to the LAN
- if you truly need a non-loopback bind, use `--allow-remote-api` together with `--api-token-env` or `--api-token-file`

Full guide: [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md)

### Build An Edge-Ready SBC Project

```bash
iints edge setup --output-dir iints_edge_demo --board raspberry_pi
cd iints_edge_demo
./run_edge_patient.sh
```

Export the live runtime back to a workstation with:

```bash
iints patient kiosk --workspace patient_runtime
iints edge status --workspace patient_runtime
iints edge bundle --workspace patient_runtime --output results/edge_runtime_bundle.zip
```

That edge flow gives you:
- a generated edge project scaffold
- a service file and update script for the board
- a kiosk dashboard URL for Raspberry Pi Connect screen sharing
- a ZIP bundle for workstation-side analysis and reporting

## Related Guides

- [User Guide Map](USER_GUIDE_MAP.md)
- [Quickstart](QUICKSTART.md)
- [Updating An Existing Install](UPDATING.md)
- [Study Analysis](STUDY_ANALYSIS.md)
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
- [AI Assistant](AI_ASSISTANT.md)
- [Evidence Base](EVIDENCE_BASE.md)
- [MDMP Quickstart](MDMP_QUICKSTART.md)
- [CLI & Advanced Reference](TECHNICAL_README.md)

## Where To Go Next

| If you finished this page and want to... | Continue with |
|---|---|
| turn one run into a formal benchmark | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| aggregate several runs | [Study Analysis](STUDY_ANALYSIS.md) |
| certify datasets and outputs | [MDMP Quickstart](MDMP_QUICKSTART.md) |
| use local AI reporting | [AI Assistant](AI_ASSISTANT.md) |
| deploy to Raspberry Pi | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) |

## Safety Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
