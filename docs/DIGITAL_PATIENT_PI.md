# Raspberry Pi Digital Patient

Use this guide when you want the SDK to keep a virtual patient running continuously on a Raspberry Pi.

The Raspberry Pi digital-patient workflow gives you:
- a persistent virtual patient runtime
- SQLite-backed state on disk
- a lightweight dashboard for the live glucose view
- exportable run bundles for later workstation-side analysis
- optional auto-start through `systemd`

If your main goal is a booth-ready startup routine, jump to [Maker Faire Pi Mode](MAKERFAIRE_PI.md).

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

The recommended Raspberry Pi flow is now CLI-first through the `iints edge ...` namespace.

Common commands:

```bash
iints edge doctor --board raspberry_pi
iints edge setup --board raspberry_pi
iints edge up
iints edge status
iints edge kiosk
iints edge reset
iints edge stop
iints edge service
iints edge bundle
iints edge update
iints makerfaire up
iints patient inject-meal
iints patient pause
iints patient resume
iints patient scenarios
iints patient review
```

## Fastest Setup

Start from a clean project directory on the Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[edge,mdmp]"

iints edge doctor --board raspberry_pi
iints edge setup --output-dir iints_pi_demo --board raspberry_pi
cd iints_pi_demo
iints edge up --project-dir .
iints edge status --project-dir .
iints edge kiosk --project-dir .
```

`iints edge doctor --board raspberry_pi` is the friendly preflight check. It tells you what is ready, what is optional, and the exact next command to run.

That starts:
- the persistent simulation loop
- the SQLite runtime store
- the local dashboard API
- a run-like bundle under `patient_runtime/live_bundle/`
- a fullscreen-friendly kiosk view at `/kiosk`

Security defaults:

- the dashboard/API listens on `127.0.0.1` by default
- mutating dashboard actions require the internal IINTS control header
- if you enable token protection, dashboard reads also require auth

## Day-To-Day Runtime Operations

Check status:

```bash
iints edge status --project-dir .
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
iints edge reset --project-dir .
```

By default, `expo-reset` switches to the `expo_hot_start` profile so the runtime resumes in an already active scenario instead of a flat baseline.

Jump to a specific profile instead:

```bash
iints edge reset \
  --project-dir . \
  --scenario-profile sport_day
```

Stop the runtime:

```bash
iints edge stop --project-dir .
```

Export the live runtime for workstation-side analysis:

```bash
iints edge bundle --project-dir . --output results/edge_runtime_bundle.zip
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

## API Security Defaults

The safest Raspberry Pi presentation path is still:

- keep the API on `127.0.0.1`
- open the dashboard locally on the Pi
- use Raspberry Pi Connect to share that local browser window

That means you usually do **not** need a remotely reachable dashboard port at all.

If you do need another machine to talk directly to the dashboard API, opt in explicitly:

```bash
export IINTS_PATIENT_API_TOKEN="replace-this-with-a-random-secret"

iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile normal_day \
  --mode demo-time \
  --speed 60x \
  --api-host 0.0.0.0 \
  --allow-remote-api \
  --api-token-env IINTS_PATIENT_API_TOKEN
```

When token protection is enabled:

- the browser can use `http://<host>:8765/dashboard?token=<your-token>`
- scripts and tools should use `Authorization: Bearer <token>`
- `/dashboard`, `/kiosk`, `/status`, `/glucose/latest`, and `/glucose/history` are protected too
- the control endpoints still require the extra `X-IINTS-Control: 1` header, which the built-in dashboard sends automatically

Recommendation:

- use token-backed remote access only when you truly need it
- for booth demos, classrooms, and Raspberry Pi Connect, prefer loopback-only mode

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
iints edge setup --output-dir iints_pi_demo_seeded --board raspberry_pi --scenario-profile bad_carb_count --seed 7777
cd iints_pi_demo_seeded
iints edge up --project-dir . --reset
```

## Remote Presentation With Raspberry Pi Connect

A simple remote presentation flow is:

1. Start the patient runtime on the Pi.
2. Open the dashboard in a browser on the Pi.
3. Connect to the Pi from another machine with Raspberry Pi Connect.
4. Share the Pi browser window.
5. Use the dashboard controls or CLI commands during the session.

This keeps the actual runtime on the Pi while the laptop or workstation acts only as the remote presenter.

This is also the security-friendly route because it avoids opening the dashboard API to the LAN in the first place.

## systemd Auto-Start

If the device should come back automatically after a reboot, export a `systemd` unit after the runtime has started once:

```bash
iints edge service --project-dir .
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
2. `iints edge up --project-dir .`
3. open `http://127.0.0.1:8765/dashboard`
4. explain the live glucose curve and current state
5. run `iints patient inject-meal --carbs 60 --workspace patient_runtime`
6. run `iints edge reset --project-dir .`
7. optionally run `iints patient review --workspace patient_runtime --model ministral-3:3b`

## Scope

- built for Raspberry Pi demos, teaching, and long-running virtual patient studies
- research use only
- not a medical device
- not a clinical treatment controller
