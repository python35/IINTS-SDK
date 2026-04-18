# Maker Faire Pi Checklist

Use this page on the exact Raspberry Pi you will bring to the event.

The goal is not to be clever. The goal is to arrive with a booth setup that starts fast, resets cleanly, and recovers without stress.

## The Day Before

### 1. Update only if you still have time to test

```bash
cd iints_pi_demo
./update_edge_runtime.sh
```

If you update the SDK, always rerun the full booth flow once before packing the Pi.

### 2. Confirm the normal booth path works

```bash
./start_makerfaire_patient.sh
```

Check:

- the runtime starts
- the kiosk URL opens
- the screen shows the digital patient

### 3. Confirm resets still work

```bash
iints edge reset --project-dir .
```

Check:

- the patient goes back to the booth-safe state
- the kiosk keeps responding

### 4. Confirm the watchdog can recover the booth

```bash
iints makerfaire watchdog --project-dir .
```

You want to know this command before the event, not during the event.

### 5. If you want auto-boot, test it once

```bash
iints makerfaire autostart --project-dir .
./install_makerfaire_autostart.sh
```

Then reboot and confirm:

- the patient service starts
- the watchdog timer is active
- the kiosk browser opens after desktop login

### 6. Pack the boring but important things

- Raspberry Pi power supply
- HDMI cable
- keyboard
- mouse
- one spare USB cable if you also bring the UNO Q

## When You Arrive

### 1. Power on the Pi

Wait until Raspberry Pi OS is fully up.

### 2. If autostart is enabled, wait first

If the kiosk opens by itself, let it finish before touching anything.

### 3. If autostart is not enabled, start manually

```bash
cd iints_pi_demo
./start_makerfaire_patient.sh
```

### 4. Check the booth screen

You should see:

- the kiosk page
- the digital patient alive
- the booth-safe profile loaded

### 5. Keep one terminal for recovery only

Best commands to keep ready:

```bash
iints edge status --project-dir .
iints edge reset --project-dir .
iints makerfaire watchdog --project-dir .
iints edge kiosk --project-dir .
```

## Between Visitors

### 1. Reset the patient

```bash
iints edge reset --project-dir .
```

### 2. Confirm the kiosk looks clean again

Make sure you are back in the default booth state before the next person starts pressing buttons.

## If Something Breaks

### The kiosk browser closed

```bash
iints edge kiosk --project-dir .
```

### The patient looks frozen or dead

```bash
iints makerfaire watchdog --project-dir .
```

### You are not sure what is happening

```bash
iints edge status --project-dir .
```

### The normal runtime flow is still weird

```bash
./start_makerfaire_patient.sh
```

### The Pi is really stuck

Reboot it and let the autostart flow bring the booth back.

## End Of The Day

### 1. Stop cleanly

```bash
iints edge stop --project-dir .
```

### 2. Optional: save the booth bundle

```bash
iints edge bundle --project-dir . --output edge_bundle.zip
```

### 3. Shut the Pi down normally

Do not just pull the power unless you really have to.

## Minimal Command Card

If you only remember four commands, remember these:

```bash
./start_makerfaire_patient.sh
iints edge reset --project-dir .
iints makerfaire watchdog --project-dir .
iints edge kiosk --project-dir .
```
