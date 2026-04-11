# UNO Q Supervisor Bridge

This folder contains the STM32-side bridge scaffold for Arduino UNO Q.

## What This Bridge Is For

Use it when:

- the Linux side runs the IINTS digital patient runtime
- the STM32 side gives simple physical feedback
- you want LEDs or a buzzer to react to supervisor state

Current scope:

- the sketch is ready to flash
- the serial protocol is ready to test
- the Linux-side runtime can forward `OK`, `OVERRIDE`, and `CRITICAL` over serial through `iints edge bridge-run`

Recommended CLI flow:

1. `iints edge setup --output-dir iints_uno_q_demo --board uno_q`
2. `iints edge up --project-dir iints_uno_q_demo`
3. flash the sketch below
4. `iints edge bridge-test --port /dev/ttyACM0`
5. `iints edge bridge-run --project-dir iints_uno_q_demo --port /dev/ttyACM0`

## Serial Protocol

Send one newline-terminated message at a time:

- `OK`
- `OVERRIDE`
- `CRITICAL`

The sketch listens at:

- `115200` baud

## Pin Mapping

- status LED: `LED_BUILTIN`
- red LED: pin `6`
- buzzer: pin `9`

Important:

- on some UNO Q boards the built-in status LED looks blue instead of green
- the bridge latches the last received state until you send a new one

If you only want a first smoke test, the built-in LED is enough to verify the `OK` state.

## Step-By-Step Test

1. Flash the sketch with either:

```bash
iints edge bridge-flash --project-dir iints_uno_q_demo --port /dev/ttyACM0 --fqbn <your-board-fqbn>
```

or by opening `iints_supervisor_bridge.ino` in Arduino IDE.
2. Select the UNO Q target and the correct serial port if you use Arduino IDE.
3. Upload the sketch to the STM32 side.
4. Open Serial Monitor.
5. Set baud rate to `115200`.
6. Set line ending to `Newline`.
7. Confirm you see:

```text
IINTS UNO Q supervisor bridge ready
```

8. Send:

```text
OK
OVERRIDE
CRITICAL
```

Expected behavior:

- `OK`: status LED on
- `OVERRIDE`: red LED on
- `CRITICAL`: red LED on and buzzer chirps

You should also see serial echoes such as:

```text
STATE=OK
```

Or use the CLI shortcut:

```bash
iints edge bridge-test --port /dev/ttyACM0
```

The CLI now waits for the board to finish its USB reset cycle and accepts acknowledgements such as:

- `STATE=OK`
- `STATE=OVERRIDE`
- `STATE=CRITICAL`

If the test finishes on `CRITICAL`, the LEDs may stay in that state until you send `OK` or start `iints edge bridge-run`.

## Where To Read More

For the full Linux-side plus STM32-side setup flow, see the published SDK docs page:

- `Arduino UNO Q Setup`
