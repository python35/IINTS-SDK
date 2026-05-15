# Maker Faire Pi Mode

Use this page when the Raspberry Pi is your show-ready virtual patient for Maker Faire.

**Read before:** [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) or [Remote Deploy & Pi Connect](EDGE_REMOTE_DEPLOY.md), depending on whether you set up locally or remotely.

**Read next:** [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md) before using the setup in public.

The goal is simple:

- one Pi runs the digital patient
- the kiosk opens full-screen on the booth screen
- the patient can be reset quickly between visitors
- the flow stays simple enough that you can recover fast during the event

## Fast Path

Generate the Raspberry Pi project once:

```bash
iints edge setup --output-dir iints_pi_demo --board raspberry_pi --scenario-profile expo_hot_start
cd iints_pi_demo
```

If you are preparing the Pi from another machine, you can now do the scaffold + sync + install in one command:

```bash
iints edge deploy \
  --host raspberrypi.local \
  --user pi \
  --local-output-dir iints_pi_demo \
  --remote-dir ~/iints_pi_demo \
  --board raspberry_pi \
  --scenario-profile expo_hot_start
```

Then use the Maker Faire startup command:

```bash
iints makerfaire up --project-dir .
```

That command:

- loads the generated edge runtime config
- starts the persistent patient if it is not already running
- uses the booth-friendly `expo_hot_start` profile by default
- prints the kiosk URL and next commands

## Generated Helper Files

`iints edge setup` now writes two Maker Faire artifacts into the project root:

- `start_makerfaire_patient.sh`
- `MAKERFAIRE_START.md`
- `open_makerfaire_kiosk.sh`
- `iints-makerfaire-kiosk.desktop`
- `install_makerfaire_autostart.sh`
- `MAKERFAIRE_AUTOSTART.md`
- `run_makerfaire_watchdog.sh`
- `iints-digital-patient-watchdog.timer`
- `MAKERFAIRE_CHECKLIST.md`

So you can also use:

```bash
./start_makerfaire_patient.sh
```

The generated kiosk launcher now tries a hardened full-screen browser path first:

- Chromium / Chrome in kiosk mode if available
- Firefox kiosk mode as fallback
- plain `xdg-open` only as the last fallback

It also tries to disable screen blanking and DPMS on Raspberry Pi OS when a desktop session is present.
For Chromium-based browsers it also disables first-run prompts, restore bubbles, background networking, sync, and common booth-unfriendly popups.

## What To Expect

After `iints makerfaire up --project-dir .` you should see:

- the workspace path
- the active scenario profile
- the kiosk URL
- the dashboard URL
- the quick reset / stop commands

If you keep the Pi connected to a booth display, the main URL you care about is:

```text
http://127.0.0.1:8765/kiosk
```

## Booth Routine

### 1. Start the patient

```bash
iints makerfaire up --project-dir .
```

### 2. Show the kiosk

If the command already printed the kiosk panel, open that URL on the Pi screen.

You can also reprint it at any time:

```bash
iints edge kiosk --project-dir .
```

### 3. Reset between visitor sessions

```bash
iints edge reset --project-dir .
```

### 4. Check whether the runtime is still healthy

```bash
iints edge status --project-dir .
```

### 4b. Let the booth watchdog repair a dead runtime

```bash
iints makerfaire watchdog --project-dir .
```

That command is safe to run manually. It checks whether the booth patient is still alive, and if not, it restarts it with the booth-safe profile.

### 5. Stop cleanly after the event

```bash
iints edge stop --project-dir .
```

## If You Also Use An Arduino UNO Q

Keep the Pi as the main brain.

- the Pi runs the virtual patient and kiosk
- the UNO Q shows physical states like `OK`, `OVERRIDE`, and `CRITICAL`

Start the Linux-side runtime first, then in a second terminal run:

```bash
iints edge bridge-run --project-dir . --port /dev/ttyACM0
```

That keeps the physical UNO Q layer synced with the Pi runtime.

## Autostart On Boot

If you want the Pi to come up straight into the virtual patient after power-on:

1. first confirm the normal command-line path works
2. inspect the generated autostart files:

```bash
iints makerfaire autostart --project-dir .
```

3. install the generated booth autostart bundle:

```bash
./install_makerfaire_autostart.sh
```

4. enable **Desktop Autologin** in Raspberry Pi Configuration so the kiosk browser can open after login
5. reboot and verify that:

- the digital patient starts automatically
- the watchdog timer is active
- the kiosk opens automatically in full-screen mode
- `iints edge reset --project-dir .` still works between visitors

You can still regenerate the plain systemd service with:

```bash
iints edge service --project-dir .
```

The generated autostart stack is split on purpose:

- `patient_runtime/*.service` starts the digital patient daemon on boot
- `iints-makerfaire-kiosk.desktop` opens the kiosk after the Pi desktop session logs in
- `iints-digital-patient-watchdog.timer` re-checks the booth runtime in the background and restarts it if the patient went down

That split is more reliable on Raspberry Pi OS than trying to force the browser open from a root-level boot service.

## Real Event Checklist

Use the full live checklist here:

- [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md)

## Practical Advice

- Use `expo_hot_start` as the default booth profile.
- Install Chromium on the Pi if you want the strongest full-screen kiosk behavior.
- Keep one terminal open only for runtime control.
- Do not rely on editing configs during the event.
- Test the full flow on the exact Pi you will bring.
- If the UNO Q is attached, test the bridge separately before the event opens.

## Where To Go Next

| If you want to... | Continue with |
|---|---|
| run the event checklist | [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md) |
| add the UNO Q physical layer | [Arduino UNO Q Setup](ARDUINO_UNO_Q.md) |
| deploy the Pi remotely | [Remote Deploy & Pi Connect](EDGE_REMOTE_DEPLOY.md) |
| analyze results afterward | [Study Analysis](STUDY_ANALYSIS.md) |
| recover from setup problems | [Troubleshooting](TROUBLESHOOTING.md) |
