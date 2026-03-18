# Getting Started

This page gives the fastest reliable path from install to a validated run.

## 1) Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

All commands below assume this `.venv` is active.

## 2) Install

```bash
pip install iints-sdk-python35
```

Optional extras:

```bash
pip install "iints-sdk-python35[research]"
pip install "iints-sdk-python35[nightscout]"
```

## 3) Verify Environment

```bash
iints doctor --smoke-run
```

If this fails, fix environment issues before running long experiments.

## 4) Create a Project

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
```

Generated structure includes:
- `algorithms/`
- `scenarios/`
- `results/`

## 5) Run a Baseline Simulation

```bash
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

## 6) Check Outputs

A typical run writes:
- `results.csv`: time-series simulation output.
- `clinical_report.pdf`: report for review.
- `audit/`: decision and safety trail.
- `run_manifest.json`: file hashes for reproducibility.
- `run_metadata.json`: run config and environment details.

## 7) Build a Study-Ready Bundle

```bash
iints study-ready --algo algorithms/example_algorithm.py --output-dir results/study_ready
```

Adds:
- `validation_report.json`
- `sources_manifest.json`
- `SUMMARY.md`

## 8) Next Steps

- Optional AI assistant (Ministral 3 via Ollama):

```bash
python -m pip install -e ".[mdmp]"
iints ai models
ollama pull ministral-3:8b
iints ai local-check --model ministral-3:8b
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
iints ai prepare results/<run_id>
iints ai report results/<run_id>
```

`iints ai local-check` now runs a tiny smoke-test generation by default. If it reports that Ollama closed the generation connection, the most likely causes are a restarting daemon or insufficient memory; in that case, try `ministral-3:3b`.

- Data validation: [MDMP Quickstart](MDMP_QUICKSTART.md)
- Full command reference: [Technical README](TECHNICAL_README.md)
- End-to-end examples: [Demos](https://github.com/python35/IINTS-SDK/tree/main/examples/demos)

## Safety Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
