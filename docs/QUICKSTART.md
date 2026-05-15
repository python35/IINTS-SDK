# Quickstart

**This page is for:** people who want a first successful run fast.

**Fastest path:** run `iints start` first. It tells you the shortest next command for a demo, project, study, edge setup, or data workflow.

**Before this page:** [Choose Your Path](USER_GUIDE_MAP.md) if you want the whole route.

**After this page:** [Getting Started](GETTING_STARTED.md) for the complete first project workflow.

## Option 1: One-Command Demo

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"

iints doctor --suggest
iints start
iints demo
```

What you get:
- a working simulation run
- `results.csv`
- `report.pdf`
- `run_manifest.json`
- an editable example algorithm in `results/demo/demo_assets/`

Useful variants:

```bash
iints start --goal project
iints start --goal project --run
iints start --goal edge
iints demo --dry-run
iints demo --full
iints demo --output-dir results/my_first_demo
```

## Option 2: Guided First Run

If you want a bit more control without memorizing flags:

```bash
iints run --wizard
```

This asks for:
- preset or custom setup
- algorithm path or built-in baseline
- duration
- output directory
- seed

## Option 3: Your Own Starter Project

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

What you get:
- `algorithms/example_algorithm.py`
- clinic-safe starter scenarios
- demo data for certification
- contracts and audit folders

## If Something Fails

Start here:

```bash
iints doctor --full --suggest
iints run --dry-run --preset baseline_t1d
```

Then check:
- [Installation](INSTALLATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [CLI Command Reference](COMMAND_REFERENCE.md)

## Where To Go Next

| If you want to... | Continue with |
|---|---|
| understand the full first-run workflow | [Getting Started](GETTING_STARTED.md) |
| fix an install or path issue | [Installation](INSTALLATION.md) |
| see every beginner command | [Command Reference](COMMAND_REFERENCE.md) |
| run a real benchmark study | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| move to Raspberry Pi or UNO Q | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) |
