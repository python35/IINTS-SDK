# Edge Remote Deploy

Use this guide when you want to prepare a Raspberry Pi from your laptop in one command, while keeping the live dashboard local to the Pi.

**Before this page:** [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) for the runtime model.

**After this page:** [Maker Faire Pi Mode](MAKERFAIRE_PI.md) for booth startup, or [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md) before an event.

## Recommended Remote Pattern

The safest default is:

- deploy and maintain the Pi over `SSH`
- present the Pi through `Raspberry Pi Connect`
- keep the IINTS dashboard bound to `127.0.0.1`
- only expose the dashboard API to the network when another machine truly needs direct access

That gives you remote shell access, browser-based screen sharing, and a safer demo posture than opening the dashboard to the full LAN.

## One-Command Remote Deploy

For a Raspberry Pi-only booth setup:

```bash
iints edge deploy \
  --host raspberrypi.local \
  --user pi \
  --local-output-dir iints_pi_demo \
  --remote-dir ~/iints_pi_demo \
  --board raspberry_pi \
  --scenario-profile expo_hot_start
```

For a Pi plus Arduino UNO Q booth:

```bash
iints edge deploy \
  --host raspberrypi.local \
  --user pi \
  --local-output-dir iints_pi_demo \
  --remote-dir ~/iints_pi_demo \
  --board uno_q \
  --scenario-profile expo_hot_start \
  --uno-bridge-port /dev/ttyACM0
```

That command now does all of this for you:

- scaffolds the edge project locally
- installs or updates the edge SDK on the Pi
- installs the systemd service, kiosk autostart, and watchdog
- starts the Maker Faire runtime
- optionally generates and installs an UNO Q bridge service

You can also preview the plan first:

```bash
iints edge deploy \
  --host raspberrypi.local \
  --user pi \
  --local-output-dir iints_pi_demo \
  --remote-dir ~/iints_pi_demo \
  --board raspberry_pi \
  --scenario-profile expo_hot_start \
  --dry-run
```

## What Gets Installed On The Pi

After deploy, the remote project contains:

- `patient_runtime/*.service`
- `install_makerfaire_autostart.sh`
- `run_makerfaire_watchdog.sh`
- `EDGE_REMOTE_ACCESS.md`
- `MAKERFAIRE_START.md`
- `MAKERFAIRE_CHECKLIST.md`

If you pass `--uno-bridge-port`, it also contains:

- `iints-uno-q-bridge.service`
- UNO Q bridge setup files under `uno_q_bridge/`

## Recommended Remote Access After Deploy

### Raspberry Pi Connect

Use Raspberry Pi Connect when you want:

- browser-based screen sharing to the kiosk
- browser-based remote shell access
- remote access without opening your dashboard port to the network

Helpful commands on the Pi:

```bash
rpi-connect status
rpi-connect shell on
rpi-connect vnc on
loginctl enable-linger
```

If screen sharing is unavailable after a reboot, turn on **Desktop Autologin** in Raspberry Pi OS.

### SSH Maintenance

The deploy command prints ready-to-copy remote maintenance commands for:

- status
- reset
- stop

There are also direct wrappers:

```bash
iints edge remote-status --host raspberrypi.local --user pi --remote-dir ~/iints_pi_demo
iints edge remote-reset --host raspberrypi.local --user pi --remote-dir ~/iints_pi_demo
iints edge remote-stop --host raspberrypi.local --user pi --remote-dir ~/iints_pi_demo
```

You can also SSH in directly and run:

```bash
cd ~/iints_pi_demo
iints edge status --project-dir .
iints edge reset --project-dir .
iints edge stop --project-dir .
```

## UNO Q Notes

The UNO Q does not need its own network access.

The Raspberry Pi stays the main remote entry point:

- the Pi runs the virtual patient and dashboard
- the Pi drives the UNO Q bridge over USB
- remote control happens on the Pi side

If you pass `--uno-bridge-port`, the generated bridge service can keep the board synced automatically after boot.

## Safe Default

For most demos and Maker Faire setups:

- use `Raspberry Pi Connect` for remote viewing and shell access
- keep `api_host=127.0.0.1`
- avoid `--allow-remote-api` unless another machine really must talk to the dashboard directly

## Where To Go Next

| If you want to... | Continue with |
|---|---|
| start the booth runtime | [Maker Faire Pi Mode](MAKERFAIRE_PI.md) |
| check event readiness | [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md) |
| understand long studies on the Pi | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) |
| add UNO Q output | [Arduino UNO Q Setup](ARDUINO_UNO_Q.md) |
| fix SSH or install problems | [Troubleshooting](TROUBLESHOOTING.md) |
