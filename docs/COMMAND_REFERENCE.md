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
