---
tags:
  - iints/user
  - iints/journey
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
role: edge user
---
# User Journey - Edge Hardware Demo

> [!tip] Who this is for
> A edge user who wants a clear path through the SDK without reading the whole repository first.

## Goal

Get from intention to a reproducible IINTS output bundle with the fewest confusing detours.

## Steps

- [ ] Run `iints doctor` and edge/jetson doctor commands.
- [ ] Use quickstart outputs before hardware demos.
- [ ] Keep Pico pump workflows bench-only and use dry-run upload first.
- [ ] For long Jetson runs, monitor status and artifacts, not just terminal output.
- [ ] Record every command and result for reproducibility.

## Commands

```bash
iints edge doctor
iints jetson doctor
iints edge pump init --output-dir pico_pump_lab
iints edge pump upload --package pico_pump_lab/dist/pico_pump_lab_package --dry-run
```

## Open Next

- [[Pump Workbench]]
- [[Pico Pump Lab Workflow]]
- [[Local AI and Hardware Sources]]
