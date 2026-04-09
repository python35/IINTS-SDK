# Raspberry Pi Digital Patient

Use this guide when you want the SDK to keep a virtual patient running continuously on a Raspberry Pi.

The Raspberry Pi digital-patient workflow gives you:
- a persistent virtual patient runtime
- SQLite-backed state on disk
- a lightweight dashboard for the live glucose view
- exportable run bundles for later workstation-side analysis
- optional auto-start through `systemd`

## Recommended Hardware

Recommended starting setup:
- `Raspberry Pi 5` with `8 GB` RAM
- active cooling
- Raspberry Pi OS `Bookworm` Desktop
- Raspberry Pi Connect enabled
- Python `3.10+`

Why this setup works well:
- Raspberry Pi 5 has enough headroom for the runtime and dashboard
- Desktop edition makes browser-based presentation and remote viewing simpler
- Raspberry Pi Connect makes it easy to present the Pi from another machine

## Runtime Command Set

The persistent runtime lives under the `iints patient ...` namespace.

Common commands:

```bash
iints patient start
iints patient status
iints patient kiosk
iints patient inject-meal
iints patient pause
iints patient resume
iints patient scenarios
iints patient expo-reset
iints patient stop
iints patient export-service
iints edge status
iints edge bundle
iints edge update
iints patient review
```

## Fastest Setup

Start from a clean project directory on the Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[edge,mdmp]"

iints quickstart --project-name iints_pi_demo
cd iints_pi_demo
```

Start the persistent patient runtime:

```bash
iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile normal_day \
  --mode demo-time \
  --speed 60x
```

That starts:
- the persistent simulation loop
- the SQLite runtime store
- the local dashboard API
- a run-like bundle under `patient_runtime/live_bundle/`
- a fullscreen-friendly kiosk view at `/kiosk`

## Day-To-Day Runtime Operations

Check status:

```bash
iints patient status --workspace patient_runtime
iints edge status --workspace patient_runtime
```

List built-in day profiles:

```bash
iints patient scenarios
```

Inject a meal during a live session:

```bash
iints patient inject-meal --carbs 60 --workspace patient_runtime
```

Pause and resume the runtime:

```bash
iints patient pause --workspace patient_runtime
iints patient resume --workspace patient_runtime
```

Reset to the prepared presentation profile:

```bash
iints patient expo-reset --workspace patient_runtime
```

By default, `expo-reset` switches to the `expo_hot_start` profile so the runtime resumes in an already active scenario instead of a flat baseline.

Jump to a specific profile instead:

```bash
iints patient expo-reset \
  --workspace patient_runtime \
  --scenario-profile sport_day
```

Stop the runtime:

```bash
iints patient stop --workspace patient_runtime
```

Export the live runtime for workstation-side analysis:

```bash
iints edge bundle --workspace patient_runtime --output results/edge_runtime_bundle.zip
```

## Dashboard URLs

Default dashboard URL:

```text
http://127.0.0.1:8765/dashboard
```

Fullscreen kiosk URL:

```text
http://127.0.0.1:8765/kiosk
```

The dashboard is intentionally lightweight and focuses on:
- current glucose
- simulated clock
- recent event history
- certification badge
- realism review status
- live glucose curve
- pause, resume, meal, and reset controls
- quick scenario shortcuts

## Scenario Profiles

Built-in profiles:
- `normal_day`
- `sport_day`
- `bad_carb_count`
- `night_hypo_risk`
- `expo_hot_start`

The first four represent general study or demo days.
`expo_hot_start` is the fast-start presentation profile.

Each profile has a fixed default seed, so the same profile and algorithm can be reproduced more easily across sessions.

Override the default seed if needed:

```bash
iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile bad_carb_count \
  --seed 7777 \
  --mode demo-time \
  --speed 60x
```

## Remote Presentation With Raspberry Pi Connect

A simple remote presentation flow is:

1. Start the patient runtime on the Pi.
2. Open the dashboard in a browser on the Pi.
3. Connect to the Pi from another machine with Raspberry Pi Connect.
4. Share the Pi browser window.
5. Use the dashboard controls or CLI commands during the session.

This keeps the actual runtime on the Pi while the laptop or workstation acts only as the remote presenter.

## systemd Auto-Start

If the device should come back automatically after a reboot, export a `systemd` unit after the runtime has started once:

```bash
iints patient export-service --workspace patient_runtime
```

This writes:
- `patient_runtime/iints-digital-patient.service`
- `patient_runtime/iints-digital-patient.INSTALL.txt`

Install it on the Pi:

```bash
sudo cp patient_runtime/iints-digital-patient.service /etc/systemd/system/iints-digital-patient.service
sudo systemctl daemon-reload
sudo systemctl enable iints-digital-patient.service
sudo systemctl start iints-digital-patient.service
systemctl status iints-digital-patient.service
```

## What Gets Stored

Inside `patient_runtime/` you will see:
- `patient_state.db`
- `patient_runtime_config.json`
- `patient.log`
- `simulator_snapshot.json`
- `live_bundle/results.csv`
- `live_bundle/run_manifest.json`
- `live_bundle/run_metadata.json`
- `live_bundle/audit/audit_summary.json`

This means the digital patient is not just a display layer. It keeps a traceable runtime history that can be analyzed later with the wider SDK workflow.

## Edge Setup And Update Helpers

Generate a Pi-friendly project scaffold with:

```bash
iints edge setup --output-dir iints_edge_demo --board raspberry_pi
```

That writes:
- `run_edge_patient.sh`
- `launch_kiosk.sh`
- `update_edge_runtime.sh`
- `patient_runtime/iints-digital-patient.service`
- `EDGE_SETUP.md`

Generate an update helper later with:

```bash
iints edge update --output-script update_edge_runtime.sh
```

## Optional AI Review

If Ollama and a local Ministral model are available, you can request a realism review of the live runtime:

```bash
iints patient review \
  --workspace patient_runtime \
  --model ministral-3:3b
```

Output goes to:
- `patient_runtime/live_bundle/ai/realism_review.md`

## Example Live Session

One simple live flow is:

1. `iints patient scenarios`
2. `iints patient start --algo algorithms/example_algorithm.py --workspace patient_runtime --scenario-profile normal_day --mode demo-time --speed 60x`
3. open `http://127.0.0.1:8765/dashboard`
4. explain the live glucose curve and current state
5. run `iints patient inject-meal --carbs 60 --workspace patient_runtime`
6. run `iints patient expo-reset --workspace patient_runtime`
7. optionally run `iints patient review --workspace patient_runtime --model ministral-3:3b`

## Scope

- built for Raspberry Pi demos, teaching, and long-running virtual patient studies
- research use only
- not a medical device
- not a clinical treatment controller
