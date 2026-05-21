---
tags:
  - iints/workflow
  - iints/pump
  - iints/safety
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Workflow - Pump Lab Bench Gate

> [!danger] Bench-only
> This workflow is for simulated commands, fake dosing packets, and bench electronics. No insulin, no patient, no animal, no therapy.

## Gate Checklist

| Gate | Required evidence |
| --- | --- |
| Algorithm simulated | completed IINTS run folder |
| Safety wrapper present | explicit dose clamps / reject rules in bench firmware |
| Dry-run upload | `iints edge pump upload --dry-run` output saved |
| Serial protocol understood | [[Pico Serial Protocol]] reviewed |
| Hardware isolated | no real pump, no real insulin, no body connection |
| Failure behavior | watchdog / reject / safe idle described |

## Commands

```bash
iints edge pump init --output-dir pico_pump_lab
iints edge pump package --project-dir pico_pump_lab
iints edge pump upload --package pico_pump_lab/dist/pico_pump_lab_package --dry-run
```

## Open Next

- [[Pump Workbench]]
- [[Bench Safety Contract]]
- [[Pump Risk Register]]
- [[Future Real Actuator Gate]]
