# Getting Started

Use this page when you want the first complete SDK workflow, not only a demo.

By the end, you will have installed the SDK, created a project, run a baseline simulation, certified the output, and reviewed the resulting bundle.

**Read before:** [Quickstart](QUICKSTART.md) if you have not run `iints demo` yet.

**Read next:** [Workflow Hub](WORKFLOWS.md), [MDMP Quickstart](MDMP_QUICKSTART.md), or [Hardware Hub](HARDWARE.md), depending on your goal.

## The Workflow In One View

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
- `patients/`
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
iints run --algo algorithms/example_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440
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

## 10) Pick Your Next Workflow

The IINTS-AF SDK supports many advanced workflows that are outside the scope of a basic simulation, including:

- **Importing personal pump/CGM data** (CareLink, Tidepool, Nightscout).
- **Running local AI Models** (Ollama) to automatically audit and explain simulation runs.
- **Generating Poster Summaries** for academic conferences.
- **Deploying to Edge devices** like the Raspberry Pi or Arduino UNO Q for live hardware testing.

To keep this Getting Started guide clean, all these advanced flows have been moved to their respective guides. 
Please refer to the [Workflow Hub](WORKFLOWS.md), [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md), or the [AI Assistant](AI_ASSISTANT.md) guide for full commands on these topics.

## Where To Go Next

| If you finished this page and want to... | Continue with |
|---|---|
| turn one run into a formal benchmark | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| aggregate several runs | [Study Analysis](STUDY_ANALYSIS.md) |
| certify datasets and outputs | [MDMP Quickstart](MDMP_QUICKSTART.md) |
| use local AI reporting | [AI Assistant](AI_ASSISTANT.md) |
| deploy to Raspberry Pi | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) |

More references:

- [Choose Your Path](USER_GUIDE_MAP.md)
- [Updating The SDK](UPDATING.md)
- [Evidence Base](EVIDENCE_BASE.md)
- [CLI & Advanced Reference](TECHNICAL_README.md)

## Safety Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
