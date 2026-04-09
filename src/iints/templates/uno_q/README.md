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
- automatic Linux-to-STM32 forwarding is not bundled yet

That means the first goal is to get the sketch working as a serial target.

## Serial Protocol

Send one newline-terminated message at a time:

- `OK`
- `OVERRIDE`
- `CRITICAL`

The sketch listens at:

- `115200` baud

## Pin Mapping

- green LED: `LED_BUILTIN`
- red LED: pin `6`
- buzzer: pin `9`

If you only want a first smoke test, the built-in LED is enough to verify the `OK` state.

## Step-By-Step Test

1. Open `iints_supervisor_bridge.ino` in Arduino IDE.
2. Select the UNO Q target and the correct serial port.
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

- `OK`: green LED on
- `OVERRIDE`: red LED on
- `CRITICAL`: red LED on and buzzer chirps

You should also see serial echoes such as:

```text
STATE=OK
```

## Where To Read More

For the full Linux-side plus STM32-side setup flow, see the published SDK docs page:

- `Arduino UNO Q Setup`
