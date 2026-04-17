# Maker Faire Pi Mode

Use this page when the Raspberry Pi is your show-ready virtual patient for Maker Faire.

The goal is simple:

- one Pi runs the digital patient
- the kiosk URL is easy to open on the booth screen
- the patient can be reset quickly between visitors
- the flow stays simple enough that you can recover fast during the event

## Fast Path

Generate the Raspberry Pi project once:

```bash
iints edge setup --output-dir iints_pi_demo --board raspberry_pi --scenario-profile expo_hot_start
cd iints_pi_demo
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

So you can also use:

```bash
./start_makerfaire_patient.sh
```

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
2. then install the generated service file from `patient_runtime/`

You can regenerate service instructions with:

```bash
iints edge service --project-dir .
```

## Practical Advice

- Use `expo_hot_start` as the default booth profile.
- Keep one terminal open only for runtime control.
- Do not rely on editing configs during the event.
- Test the full flow on the exact Pi you will bring.
- If the UNO Q is attached, test the bridge separately before the event opens.
