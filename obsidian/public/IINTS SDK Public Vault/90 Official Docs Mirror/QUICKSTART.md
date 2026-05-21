# Quickstart

Use this page when you want one successful run as quickly as possible.

By the end, you should have a working install, one demo bundle, and enough confidence to continue to the full workflow.

**Read before:** [Choose Your Path](USER_GUIDE_MAP.md) if you are still deciding what you need.

**Read next:** [Getting Started](GETTING_STARTED.md) when you want a complete project rather than only a demo.

## Fastest Working Path

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"

iints doctor --smoke-run
iints demo
```

## What Success Looks Like

The demo writes:

- a working simulation run
- `results.csv`
- `report.pdf`
- `run_manifest.json`
- an editable example algorithm under `results/demo/demo_assets/`

If those files exist, the SDK is installed and the core workflow is working.

## If You Want A Little More Control

### Let the CLI choose the route

```bash
iints start
iints start --goal project
iints start --goal edge
```

### Build a guided custom run

```bash
iints run --wizard
```

### Create a starter project

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints run --algo algorithms/example_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440
```

The generated project includes:

- `algorithms/`
- `scenarios/`
- `patients/`
- `contracts/`
- `data/demo/`
- `audit/`
- `results/`

## If Something Fails

Run the shortest recovery checks first:

```bash
iints doctor --full --suggest
iints run --dry-run --preset baseline_t1d
```

The dry run now includes a no-insulin/no-meal physiology preview so you can spot suspicious patient settings before spending time on a long simulation.

Then continue with:

- [Installation](INSTALLATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Command Reference](COMMAND_REFERENCE.md)

## Where To Go Next

| If you want to... | Continue with |
| --- | --- |
| complete the full first workflow | [Getting Started](GETTING_STARTED.md) |
| fix install or path issues | [Installation](INSTALLATION.md) |
| run a formal benchmark | [Workflow Hub](WORKFLOWS.md) |
| deploy to edge hardware | [Hardware Hub](HARDWARE.md) |
