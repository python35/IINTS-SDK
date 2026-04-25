# Command Reference

**This page is for:** people who want a quick map of the CLI without reading the full technical manual.

## Beginner-Friendly Entry Points

### `iints guide`
Use this when you are not sure where to start.

### `iints demo`
Zero-config first run.

Common forms:

```bash
iints demo
iints demo --full
iints demo --dry-run
```

### `iints quickstart`
Create a ready-to-run project folder.

```bash
iints quickstart --project-name iints_quickstart
```

### `iints run --wizard`
Interactive custom run builder.

## Core Simulation Commands

### `iints run`
Run one simulation.

Examples:

```bash
iints run --preset baseline_t1d
iints run --algo algorithms/example_algorithm.py --scenario scenarios/example_scenario.json
iints run --dry-run --preset baseline_t1d
```

### `iints run-full`
One-line run with full output bundle.

### `iints run-parallel`
Run a matrix of scenarios in parallel.

### `iints benchmark`
Compare algorithms across standard workloads.

## Study / Research Commands

### `iints study-protocol`
Write the official study protocol bundle.

### `iints run-study`
Run the scientific benchmark matrix.

### `iints analyze`
Aggregate a study directory.

### `iints compare-study`
Compare two study outputs.

### `iints poster-study`
Generate poster-ready figures from a study.

## Data Commands

### `iints data list`
Show public data packs.

### `iints data fetch`
Fetch a pack into a local directory.

### `iints data certify`
Run data certification.

### `iints import-carelink`, `iints import-nightscout`, `iints import-tidepool`
Import real-world CGM sources.

## Edge / Booth Commands

### `iints edge doctor`
Preflight for Raspberry Pi or UNO Q.

### `iints edge setup`
Generate an edge project scaffold.

### `iints edge deploy`
Scaffold, upload, install, and start a Raspberry Pi edge project in one command.

### `iints edge offline-bundle`
Build a USB-friendly offline install tarball for Raspberry Pi or UNO Q setups.

### `iints edge study`
Run a reproducible multi-seed study directly on the current edge machine.

### `iints edge long-study`
Run a multi-day or multi-week YAML-driven study directly on the Pi, with rolling day profiles and export-friendly nested outputs.
Use `--resume` to continue from the next incomplete day after a reboot.

### `iints edge study-snapshot`
Create a `.tar.gz` snapshot of a long-study folder for crash recovery or USB backup.

### `iints edge study-export`
Package a long-study folder into a transfer-ready zip archive for another device.

### `iints edge remote-status`, `iints edge remote-reset`, `iints edge remote-stop`
Run common Raspberry Pi maintenance commands remotely over SSH.

### `iints makerfaire up`
Start the Pi booth flow.

### `iints makerfaire autostart`
Prepare booth autostart files.

### `iints makerfaire watchdog`
Recover the booth runtime if it stops.

## Diagnostics

### `iints doctor`
Basic and full environment checks.

```bash
iints doctor
iints doctor --full --suggest
iints doctor --smoke-run
```

## Full Details

For every option and advanced workflow, continue to:
- [CLI & Advanced Reference](TECHNICAL_README.md)
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
- [Study Analysis](STUDY_ANALYSIS.md)
