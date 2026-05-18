# Command Reference

Use this page when you know the task and want the public command family quickly, without reading the full technical manual.

**Read before:** [Choose Your Path](USER_GUIDE_MAP.md) if you are not sure which command family you need.

**Read next:** [Technical Reference](TECHNICAL_README.md) for deeper integration details.

## Beginner-Friendly Entry Points

### `iints guide`
Use this when you are not sure where to start.

### `iints start`
Print a goal-based first-run plan, or run the safe starter action.

Common forms:

```bash
iints start
iints start --goal project --run
iints start --goal edge
iints start --goal data
```

### `iints onboard`
Show the one recommended path from machine check to first study bundle.

```bash
iints onboard
iints onboard --run-safe-steps
```

The safe mode runs `doctor`, a full demo, demo-data import, and a realism check,
then prints the two study commands you can run once you choose an algorithm.

### `iints demo`
Zero-config first run.

Common forms:

```bash
iints demo
iints demo --full
iints demo --dry-run
```

### `iints demo-live`
One-command live presentation flow for calls, juries, and sponsor walkthroughs.

It exports showable Python code, prints an audience-aware opening talk track, shows the key SDK calls in the terminal, runs the live demo, writes `PRESENTER_GUIDE.md`, and lists the poster plus proof artifacts to open next.

Common forms:

```bash
iints demo-live
iints demo-live --output-dir results/live_demo
iints demo-live --no-run
iints demo-live --prepare-ai
iints demo-live --audience clinical
iints demo-live --audience engineering
```

### `iints quickstart`
Create a ready-to-run project folder.

```bash
iints quickstart --project-name iints_quickstart
```

The generated project is self-contained: it includes `patients/stable_patient.yaml`, a scenario file, and an editable starter algorithm so you can run locally without depending on packaged patient assets.

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

## Extension Commands

### `iints plugin install`
Install a local algorithm plugin without editing SDK source code.

```bash
iints plugin install algorithms/my_algo.py
iints algorithms list
```

The SDK copies the file into the local plugin home and records it in
`~/.iints/plugins/registry.json`. For tests or portable environments, set
`IINTS_PLUGIN_HOME` to another folder.

### `iints plugin register`
Register extension files by kind.

```bash
iints plugin register algo algorithms/my_algo.py
iints plugin register patient-model patient_models/my_model.py --name "My Model"
iints plugin register data-source data_sources/my_importer.py
iints plugin register validator validators/my_check.py
```

Algorithm plugins become visible in `iints algorithms list`. Patient model,
data source, and validator plugins are registered for discovery/documentation
hooks so the SDK can grow without source-code edits.

### `iints plugin list`
Show local extension plugins.

```bash
iints plugin list
iints plugin list --kind algorithm
```

### `iints plugin uninstall`
Remove a local plugin registry entry.

```bash
iints plugin uninstall "My Algorithm"
iints plugin uninstall "My Algorithm" --remove-file
```

### `iints patientmodel list`
Show built-in and locally registered patient models.

```bash
iints patientmodel list
```

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

### `iints data realism-check`
Judge whether a glucose trace looks physiologically plausible for research or demo use.
Supports:
- `--reference free_living_t1d`, `--reference azt1d`, or `--reference hupa_ucm`
- `--output-json results/realism_report.json`
- `--output-html results/realism_dashboard.html`
- `--min-realism-verdict needs_review`

### `iints import-carelink`, `iints import-nightscout`, `iints import-tidepool`
Import real-world CGM sources.

## Edge / Booth Commands

### `iints edge doctor`
Preflight for Raspberry Pi or UNO Q.

### `iints edge quickstart`
Create the easiest Pi or UNO Q demo project and optionally start the Linux-side runtime.

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

## Jetson Endurance Commands

### `iints jetson doctor`
Check Jetson-like hardware probes, thermal zones, and NVIDIA tooling before a long headless run.

### `iints jetson endurance start`
Run a headless adversarial endurance study.

```bash
iints jetson endurance start \
  --algo algorithms/example_algorithm.py \
  --predictor models/lstm_predictor.pt \
  --duration 7d \
  --output-dir results/jetson_7day \
  --profile mixed_adversarial \
  --seed 42 \
  --checkpoint-interval 360 \
  --hardware-sample-interval 60
```

### `iints jetson endurance status`
Show progress, current glucose, TIR so far, interventions, critical events, the
latest checkpoint, resume count, and wall-clock ETA.

### `iints jetson endurance monitor`
Print the same status repeatedly with `--watch`.

### `iints jetson endurance stop`
Request a safe stop and optional report finalization.

### `iints jetson endurance export`
Package the complete endurance folder into a transfer-ready `.zip`.

### `iints jetson endurance install-service`
Write a systemd service file with automatic `--resume` for multi-day Jetson runs.

Full workflow: [Jetson Endurance Mode](JETSON_ENDURANCE.md).

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

`doctor` reports the installed SDK version, active Python executable, install path, and available command groups. This is the fastest way to catch an old Python interpreter that silently resolves only legacy SDK releases.

### `iints profiles`

```bash
iints profiles presets
iints profiles create --name stable_patient --preset stable-demo
iints profiles create --name endurance_patient --preset endurance
```

Starter presets are provided for `stable-demo`, `stress-test`, and `endurance`.

## Full Details

For every option and advanced workflow, continue to:
- [Choose Your Path](USER_GUIDE_MAP.md)
- [CLI & Advanced Reference](TECHNICAL_README.md)
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
- [Study Analysis](STUDY_ANALYSIS.md)
