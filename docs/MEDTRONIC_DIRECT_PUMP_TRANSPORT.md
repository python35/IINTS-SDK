# Medtronic Direct Pump Transport

This SDK layer is for authorized, read-only, direct pump experiments. It gives
IINTS a stable transport boundary for live pump snapshots without embedding
private mobile-app or sensor impersonation logic in the SDK.

## What This Supports

- A direct pump snapshot contract for glucose, carbs, insulin, IOB, basal rate,
  reservoir, battery, alerts, and pump state
- A bench-only simulated Medtronic transport for integration testing
- A plugin point for an approved internal Medtronic transport module
- CSV/JSON outputs compatible with the existing IINTS data pipeline

## What This Does Not Implement

- Mobile app spoofing
- Sensor spoofing
- Pairing-key extraction
- Auth challenge replay
- BLE characteristic write flows
- Pump command/control or therapy delivery

Those belong in a formally approved Medtronic protocol stack or test firmware,
not in this research SDK.

## Simulated Direct Pump Test

```bash
iints medtronic-pump-direct \
  --transport simulated \
  --samples 12 \
  --poll-seconds 0 \
  --output-dir results/medtronic_pump_direct
```

Outputs:

- `pump_timeline.csv`: absolute pump snapshot timeline
- `cgm_standard.csv`: IINTS universal schema with relative minute timestamps
- `pump_latest.json`: latest snapshot for dashboards and smoke tests

## Approved Internal Transport

Implement a Python factory inside your internal Medtronic package:

```python
class OfficialReadOnlyPumpTransport:
    def connect(self):
        ...

    def read_snapshot(self):
        return {
            "timestamp": "2026-05-25T10:00:00Z",
            "glucoseMgDl": 122,
            "iobU": 1.4,
            "basalRateUPerHr": 0.85,
            "reservoirU": 142,
            "batteryPercent": 96,
            "pumpState": "normal",
        }

    def disconnect(self):
        ...


def create_transport():
    return OfficialReadOnlyPumpTransport()
```

Then load it through the SDK:

```bash
iints medtronic-pump-direct \
  --transport official-module \
  --official-factory medtronic_internal.pump_readonly:create_transport \
  --read-only-confirm "I confirm this is an authorized read-only Medtronic transport" \
  --samples 0 \
  --poll-seconds 30 \
  --output-dir results/medtronic_pump_direct
```

The factory object must expose `connect()`, `read_snapshot()`, and optionally
`disconnect()`. `read_snapshot()` can return either a `PumpSnapshot` or a mapping
with common field aliases.

The SDK rejects command/auth-like fields in snapshot payloads so that this
boundary stays read-only.
