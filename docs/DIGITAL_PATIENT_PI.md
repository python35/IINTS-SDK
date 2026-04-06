# Digital Patient On Raspberry Pi

This guide explains the new persistent `iints patient ...` flow for a Raspberry Pi 5 demo rig.

The idea is simple:

- the Raspberry Pi keeps a virtual diabetes patient running
- the patient advances every 5 minutes of simulated time
- the state stays on disk in SQLite
- a small local dashboard shows the live glucose curve
- Raspberry Pi Connect is used to present that dashboard from a laptop

This gives you a live "digital patient" instead of a one-shot script.

The current implementation now includes the four pieces that matter most for a fair setup:

- persistent runtime with SQLite state
- reproducible scenario profiles with fixed default seeds
- an `expo-reset` command that warm-starts into an interesting situation
- systemd export so the Pi can auto-restart after a reboot

## Recommended Hardware

The safest starting setup is:

- `Raspberry Pi 5` with `8 GB` RAM
- active cooling
- Raspberry Pi OS `Bookworm` Desktop
- Raspberry Pi Connect enabled
- Python `3.10+`

Why this setup:

- Desktop edition makes screen sharing easier for an expo table
- Raspberry Pi Connect gives browser-based remote access
- Pi 5 has enough headroom for the SDK runtime and the dashboard

## What The Runtime Gives You

The new namespace is:

```bash
iints patient ...
```

Core commands:

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

## Fastest Pi Setup

Start from a clean project folder on the Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[edge,mdmp]"

iints quickstart --project-name iints_pi_demo
cd iints_pi_demo
```

Then start the persistent patient:

```bash
iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile normal_day \
  --mode demo-time \
  --speed 60x
```

That starts:

- the persistent loop
- the SQLite runtime store
- the local dashboard API
- a run-like bundle under `patient_runtime/live_bundle/`
- a kiosk-capable fullscreen dashboard at `/kiosk`

## Daily Expo Flow

Check status:

```bash
iints patient status --workspace patient_runtime
iints edge status --workspace patient_runtime
```

List the available day profiles before a jury chooses one:

```bash
iints patient scenarios
```

Inject a manual meal during a live explanation:

```bash
iints patient inject-meal --carbs 60 --workspace patient_runtime
```

Pause and resume:

```bash
iints patient pause --workspace patient_runtime
iints patient resume --workspace patient_runtime
```

Reset to a clean expo-ready morning:

```bash
iints patient expo-reset --workspace patient_runtime
```

By default, `expo-reset` switches to the special `expo_hot_start` profile.
That profile is warm-started into the middle of an under-counted lunch challenge so visitors do not arrive to a flat curve.

If you want to jump to a specific profile instead:

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

## Dashboard

By default the dashboard lives at:

```text
http://127.0.0.1:8765/dashboard
```

The fullscreen expo view lives at:

```text
http://127.0.0.1:8765/kiosk
```

Open that page on the Pi itself and present it with Raspberry Pi Connect screen sharing.

The dashboard is intentionally lightweight:

- current glucose
- simulated clock
- last event
- certification badge
- realism review status
- live glucose curve
- pause / resume / meal / expo reset buttons
- one-click scenario shortcuts

This makes it robust for offline or low-friction fair demos.

## Scenario Profiles

The digital patient now ships with these built-in profiles:

- `normal_day`
- `sport_day`
- `bad_carb_count`
- `night_hypo_risk`
- `expo_hot_start`

The first four are the main scientific day profiles.
`expo_hot_start` is a presentation-oriented profile designed to start mid-challenge.

Each profile has a fixed default seed.
That means:

- same profile
- same seed
- same algorithm

should produce the same live start conditions again, which is much easier to defend in front of a jury.

If you want to override the default seed:

```bash
iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile bad_carb_count \
  --seed 7777 \
  --mode demo-time \
  --speed 60x
```

## Raspberry Pi Connect Workflow

The cleanest fair setup is:

1. Start the patient on the Pi.
2. Open the dashboard on the Pi in a browser.
3. Use Raspberry Pi Connect from your laptop.
4. Screen-share the Pi browser window.
5. Use the shell or dashboard buttons live during the conversation.

This keeps the real application on the Pi while the laptop only acts as the remote presenter.

## systemd Auto-Start

For an expo table, auto-restart matters.
If the Pi reboots, the patient should come back without manual repair work.

After starting the runtime once, export a ready-to-install systemd unit:

```bash
iints patient export-service --workspace patient_runtime
```

That writes:

- `patient_runtime/iints-digital-patient.service`
- `patient_runtime/iints-digital-patient.INSTALL.txt`

Then install it on the Pi:

```bash
sudo cp patient_runtime/iints-digital-patient.service /etc/systemd/system/iints-digital-patient.service
sudo systemctl daemon-reload
sudo systemctl enable iints-digital-patient.service
sudo systemctl start iints-digital-patient.service
systemctl status iints-digital-patient.service
```

This is the recommended way to make the demo resilient for the fair.

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

That means the "digital patient" is not just a display.
It keeps a traceable, reviewable history that can later feed the wider IINTS analysis pipeline.

## Edge Setup And Update Helpers

If you want the Pi project scaffold generated for you, use:

```bash
iints edge setup --output-dir iints_edge_demo --board raspberry_pi
```

That writes:

- `run_edge_patient.sh`
- `launch_kiosk.sh`
- `update_edge_runtime.sh`
- `patient_runtime/iints-digital-patient.service`
- `EDGE_SETUP.md`

To refresh the SDK on the device later:

```bash
iints edge update --output-script update_edge_runtime.sh
```

## AI Review On Demand

If Ollama and a local Ministral model are available, you can ask for a realism review of the live runtime:

```bash
iints patient review \
  --workspace patient_runtime \
  --model ministral-3:3b
```

Output goes to:

- `patient_runtime/live_bundle/ai/realism_review.md`

That is useful when you want the Pi to do more than just simulate:

- simulate
- log
- certify
- explain
- critique realism

## Expo Script In One Minute

The most reliable fair flow is:

1. `iints patient scenarios`
2. `iints patient start --algo algorithms/example_algorithm.py --workspace patient_runtime --scenario-profile normal_day --mode demo-time --speed 60x`
3. open `http://127.0.0.1:8765/dashboard`
4. explain the live glucose curve
5. trigger `iints patient inject-meal --carbs 60 --workspace patient_runtime`
6. show `iints patient expo-reset --workspace patient_runtime`
7. optionally run `iints patient review --workspace patient_runtime --model ministral-3:3b`

## Current Scope

- built for Raspberry Pi demos, teaching, and long-running virtual patient studies
- research use only
- not a medical device
- not a clinical treatment controller
