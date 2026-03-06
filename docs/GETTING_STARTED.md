# Getting Started

This page gives the fastest reliable path from install to a validated run.

## 1) Install

```bash
pip install iints-sdk-python35
```

Optional extras:

```bash
pip install "iints-sdk-python35[research]"
pip install "iints-sdk-python35[nightscout]"
```

## 2) Verify Environment

```bash
iints doctor --smoke-run
```

If this fails, fix environment issues before running long experiments.

## 3) Create a Project

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
```

Generated structure includes:
- `algorithms/`
- `scenarios/`
- `results/`

## 4) Run a Baseline Simulation

```bash
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

## 5) Check Outputs

A typical run writes:
- `results.csv`: time-series simulation output.
- `clinical_report.pdf`: report for review.
- `audit/`: decision and safety trail.
- `run_manifest.json`: file hashes for reproducibility.
- `run_metadata.json`: run config and environment details.

## 6) Build a Study-Ready Bundle

```bash
iints study-ready --algo algorithms/example_algorithm.py --output-dir results/study_ready
```

Adds:
- `validation_report.json`
- `sources_manifest.json`
- `SUMMARY.md`

## 7) Next Steps

- Data validation: [MDMP Quickstart](MDMP_QUICKSTART.md)
- Full command reference: [Technical README](TECHNICAL_README.md)
- End-to-end examples: [Demos](https://github.com/python35/IINTS-SDK/tree/main/examples/demos)

## Safety Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
