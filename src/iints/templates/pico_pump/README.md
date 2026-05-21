# IINTS Pico Pump Bench Firmware

This firmware is a **bench-only** Raspberry Pi Pico/CircuitPython-style serial target.
It is designed for SDK workflow testing, not for insulin delivery.

## Safety Scope

- no motor pins are defined
- no pump movement code is included
- `DOSE`, `BOLUS`, `BASAL`, and `PRIME` commands are rejected
- the board reports `hardware_actuation_enabled=false`
- use LEDs, serial logs, dummy loads, or disconnected mechanics only

Do not connect this firmware to a reservoir, infusion set, person, or animal.

## Commands

Serial baud rate: `115200`

Send newline-terminated commands:

```text
PING
STATUS
LOCKOUT
```

Expected responses are JSON lines. Example:

```json
{"reply":"PONG","ok":true,"device":"iints-pico-pump-bench","hardware_actuation_enabled":false,"locked":true}
```

## Upload

Use the SDK command:

```bash
iints edge pump upload \
  --bundle-dir bundles/pico_bench_bundle \
  --mount-dir /Volumes/CIRCUITPY \
  --bench-only-confirm "I understand this is bench-only and not for human use" \
  --write
```

If the board appears as `RPI-RP2`, it is in BOOTSEL/UF2 mode. Install a Python-capable
runtime first, then mount a writable drive such as `CIRCUITPY`.
