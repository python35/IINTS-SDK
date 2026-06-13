---
tags:
  - iints/workflow
  - iints/edge
  - iints/local-ai
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Workflow - Jetson Long Run

## Goal

Run a long edge experiment in a way that is monitorable, resumable, and explainable.

## Before Starting

- [ ] `iints jetson doctor` passes or explains missing capabilities.
- [ ] Algorithm path exists.
- [ ] Output folder is empty or intentionally resumed.
- [ ] Execution mode is understood: simulated-time vs wall-clock.
- [ ] If local AI is used, model/version/hardware are recorded.

## Commands

```bash
iints jetson doctor
iints jetson endurance start --algo iints_quickstart/algorithms/example_algorithm.py --duration 1d --profile normal --output-dir results/jetson_research_day
iints jetson endurance status --output-dir results/jetson_research_day
iints jetson endurance monitor --output-dir results/jetson_research_day --watch
```

## Evidence To Keep

- `status.json`
- monitor output screenshots/logs
- run config
- model name if local AI used
- final summary
- notes about thermal throttling, power, and interruptions

## Interpretation Boundary

A 24h Jetson run proves software endurance on that device under that setup. It does not prove medical safety.
