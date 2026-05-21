"""IINTS Pico Pump Bench Firmware.

Bench-only serial target for Raspberry Pi Pico / CircuitPython-style boards.
This firmware intentionally contains no motor actuation code and no insulin
commands. It is a USB serial target for SDK upload, lockout, and status tests.
"""

import json
import sys
import time

DEVICE_ID = "iints-pico-pump-bench"
READY_BANNER = "IINTS Pico Pump Bench firmware ready"
HARDWARE_ACTUATION_ENABLED = False
LOCKED = True
BOOT_TIME = time.monotonic()


def _response(ok, **payload):
    payload["ok"] = bool(ok)
    payload["device"] = DEVICE_ID
    payload["hardware_actuation_enabled"] = HARDWARE_ACTUATION_ENABLED
    payload["locked"] = LOCKED
    return json.dumps(payload, separators=(",", ":"))


def _handle_command(line):
    command = line.strip().upper()
    if command == "PING":
        return _response(True, reply="PONG")
    if command == "STATUS":
        return _response(True, uptime_seconds=round(time.monotonic() - BOOT_TIME, 2), mode="bench_locked")
    if command == "LOCKOUT":
        return _response(True, reply="LOCKOUT_ACTIVE")
    if command in {"DOSE", "BOLUS", "BASAL", "PRIME"} or command.startswith(("DOSE ", "BOLUS ", "BASAL ", "PRIME ")):
        return _response(False, error="BLOCKED_BY_BENCH_FIRMWARE")
    return _response(False, error="UNKNOWN_COMMAND", command=command)


print(READY_BANNER)
sys.stdout.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        time.sleep(0.05)
        continue
    print(_handle_command(line))
    sys.stdout.flush()
