# Arduino UNO Q Setup

Use this guide when you want the Linux side of an Arduino UNO Q to run the IINTS digital patient runtime and the STM32 side to act as a simple LED / buzzer bridge.

This page is intentionally step by step. Follow it top to bottom once, get the baseline working, and only then start customizing it.

## Quick Path

If you want the fastest possible route to a working baseline, follow these four blocks first.

!!! success "Step 1 - Install IINTS on the Linux side"
    Run:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -U "iints-sdk-python35[edge,mdmp]"
    ```

    Success looks like:

    - `iints doctor --smoke-run` works
    - `python -c "import iints; print(iints.__version__)"` prints a version

!!! info "Step 2 - Generate the UNO Q project"
    Run:

    ```bash
    iints edge setup --output-dir iints_uno_q_demo --board uno_q
    cd iints_uno_q_demo
    ```

    Success looks like:

    - `run_edge_patient.sh` exists
    - `uno_q_bridge/iints_supervisor_bridge.ino` exists

!!! tip "Step 3 - Start the Linux runtime"
    Run:

    ```bash
    ./run_edge_patient.sh
    iints edge status --workspace patient_runtime
    ```

    Success looks like:

    - `daemon_status` is `running`
    - the kiosk opens at `http://127.0.0.1:8765/kiosk`

!!! warning "Step 4 - Flash and test the STM32 bridge"
    Open `uno_q_bridge/iints_supervisor_bridge.ino` in Arduino IDE, upload it, open Serial Monitor at `115200` baud, and send:

    ```text
    OK
    OVERRIDE
    CRITICAL
    ```

    Success looks like:

    - `OK` turns the green LED on
    - `OVERRIDE` turns the red LED on
    - `CRITICAL` turns the red LED on and chirps the buzzer

Once those four blocks work, you have a real working UNO Q baseline.

## What Works Today

The current UNO Q path gives you:

- a working Linux-side IINTS edge runtime
- a generated STM32 sketch scaffold for the bridge side
- a simple serial protocol for bridge testing
- a kiosk dashboard and persistent runtime workspace

Important:

- the Linux runtime works out of the box
- the STM32 sketch works as a serial target out of the box
- automatic Linux-to-STM32 serial forwarding is not bundled yet

So the first success target is:

1. get the Linux runtime running
2. flash the STM32 bridge sketch
3. manually verify the bridge with serial messages

That gives you a solid working baseline on UNO Q.

## Before You Start

You need:

- an Arduino UNO Q with access to both the Linux side and the STM32 side
- Python `3.10+` on the Linux side
- Arduino IDE for flashing the STM32 sketch
- a USB cable and a serial port that reaches the STM32 side

Optional hardware for the bridge demo:

- green LED: built-in `LED_BUILTIN`
- red LED on pin `6`
- buzzer on pin `9`

If you do not wire an external red LED or buzzer yet, you can still complete the setup and verify the sketch through the built-in LED plus serial output.

## Step 1: Install The Edge Runtime On The Linux Side

Open a terminal on the Linux side of the UNO Q and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[edge,mdmp]"
```

Verify the install:

```bash
iints doctor --smoke-run
python -c "import iints; print(iints.__version__)"
```

If that succeeds, the SDK is installed correctly on the Linux side.

## Step 2: Generate A UNO Q Edge Project

Create the edge scaffold:

```bash
iints edge setup --output-dir iints_uno_q_demo --board uno_q
cd iints_uno_q_demo
```

This creates:

- `algorithms/example_algorithm.py`
- `patient_runtime/patient_runtime_config.json`
- `run_edge_patient.sh`
- `launch_kiosk.sh`
- `update_edge_runtime.sh`
- `EDGE_SETUP.md`
- `uno_q_bridge/iints_supervisor_bridge.ino`
- `uno_q_bridge/README.md`
- `uno_q_bridge/bridge_protocol.txt`

If you only want the MCU bridge scaffold without the full edge project, you can also run:

```bash
iints edge hardware-bridge --board uno_q --output-dir uno_q_bridge
```

## Step 3: Start The Linux-Side Digital Patient

From inside `iints_uno_q_demo`, start the runtime:

```bash
./run_edge_patient.sh
```

In a second terminal, check status:

```bash
iints edge status --workspace patient_runtime
```

What you want to see:

- `daemon_status` shows `running`
- a dashboard URL is printed
- the workspace contains `patient_state.db`

If that works, the Linux side is ready.

## Step 4: Open The Kiosk Dashboard

Launch the local kiosk view:

```bash
./launch_kiosk.sh
```

Or open the dashboard manually:

```text
http://127.0.0.1:8765/kiosk
```

At this point you should have a working UNO Q Linux-side demo, even before touching the STM32 bridge.

## Step 5: Flash The STM32 Bridge Sketch

Open this file in Arduino IDE:

```text
iints_uno_q_demo/uno_q_bridge/iints_supervisor_bridge.ino
```

Then:

1. Select the correct Arduino UNO Q board target in Arduino IDE.
2. Select the correct serial port.
3. Upload the sketch to the STM32 side.
4. Open Serial Monitor.
5. Set the baud rate to `115200`.
6. Set line ending to `Newline`.

If the upload worked, you should see:

```text
IINTS UNO Q supervisor bridge ready
```

## Step 6: Verify The Bridge Manually

Before trying to automate anything, test the bridge manually from Serial Monitor.

Send these commands one by one:

```text
OK
OVERRIDE
CRITICAL
```

Expected behavior:

- `OK`
  - green LED on
- `OVERRIDE`
  - red LED on
- `CRITICAL`
  - red LED on
  - buzzer chirps

The sketch also prints confirmations such as:

```text
STATE=OK
STATE=OVERRIDE
STATE=CRITICAL
```

If this works, the STM32 side is ready.

## Step 7: Day-To-Day Commands On The Linux Side

Useful commands once the runtime is running:

```bash
iints edge status --workspace patient_runtime
iints patient kiosk --workspace patient_runtime
iints patient expo-reset --workspace patient_runtime
iints edge bundle --workspace patient_runtime --output edge_bundle.zip
```

Use these to:

- check whether the runtime is alive
- reopen the kiosk view
- reset to a known demo state
- export the runtime bundle back to a workstation

## Step 8: Optional Auto-Start

If the Linux side should come back after reboot:

```bash
iints patient export-service --workspace patient_runtime
```

Then install the generated service:

```bash
sudo cp patient_runtime/iints-digital-patient.service /etc/systemd/system/iints-digital-patient.service
sudo systemctl daemon-reload
sudo systemctl enable iints-digital-patient.service
sudo systemctl start iints-digital-patient.service
systemctl status iints-digital-patient.service
```

## What Is Still Manual

Be aware of the current boundary:

- the SDK generates the Linux runtime scaffold
- the SDK generates the STM32 bridge sketch and protocol
- the SDK does not yet ship an automatic serial forwarder from Linux runtime state to the STM32 side

So if you want the LEDs and buzzer to react automatically during the live runtime, you need one extra custom layer that sends `OK`, `OVERRIDE`, or `CRITICAL` over serial.

That is why the recommended first milestone is:

1. Linux runtime works
2. bridge sketch works
3. manual serial test works

Once those three work, you have a reliable baseline to extend.

## Troubleshooting

### `iints` command not found

Activate the virtual environment again:

```bash
source .venv/bin/activate
```

### The runtime does not start

Check the workspace:

```bash
iints edge status --workspace patient_runtime
ls patient_runtime
```

You should see `patient_state.db` after a successful start.

### The dashboard does not open

Open the URL manually:

```text
http://127.0.0.1:8765/dashboard
```

If that works in the browser but not through `./launch_kiosk.sh`, the runtime is fine and the issue is only the browser launcher.

### The STM32 sketch uploads, but the LED test does nothing

Check these items:

1. Serial Monitor baud rate is `115200`
2. line ending is `Newline`
3. you are sending exactly `OK`, `OVERRIDE`, or `CRITICAL`
4. your red LED and buzzer wiring matches pin `6` and pin `9`

### I only want a stable UNO Q demo quickly

Use this order:

1. get the Linux kiosk working
2. flash the STM32 sketch
3. test the sketch manually
4. keep the bridge automation for later

That is the fastest path to a dependable Maker Faire or classroom demo.
