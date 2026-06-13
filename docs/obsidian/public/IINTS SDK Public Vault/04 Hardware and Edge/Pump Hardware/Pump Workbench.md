---
tags:
  - area/pump
  - dashboard
  - safety
cssclasses:
  - iints-dashboard
status: active
---
# Pump Workbench

> [!warning] Bench-only
> This is for Pico/PCB bench research. No reservoir, no infusion set, no human/animal use.

## Main Flow

| Step | Note |
| --- | --- |
| Algorithm to board | [[Algorithm to Pump Pipeline]] |
| Pico workflow | [[Pico Pump Lab Workflow]] |
| Safety contract | [[Bench Safety Contract]] |
| Serial protocol | [[Pico Serial Protocol]] |
| PCB bring-up | [[PCB Bring-Up Plan]] |
| Risks | [[Pump Risk Register]] |
| Hard future stop | [[Future Real Actuator Gate]] |

## Current Pump Tasks

- [ ] Test `iints edge pump init` from a clean folder
- [ ] Run package dry-run for one real SDK algorithm
- [ ] Verify serial test output on actual Pico
- [ ] Add physical BENCH ONLY label plan to PCB note
- [ ] Document first PCB power-only bring-up

## Pump Search

```query
path:"40 Pump Hardware" OR path:"20 Official Documentation/docs/PICO_PUMP_LAB.md" OR "pico" OR "pump" OR "pcb"
```
