---
tags:
  - iints/user
  - iints/journey
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
role: algorithm developer
---
# User Journey - Algorithm Developer

> [!tip] Who this is for
> A algorithm developer who wants a clear path through the SDK without reading the whole repository first.

## Goal

Get from intention to a reproducible IINTS output bundle with the fewest confusing detours.

## Steps

- [ ] Start from a conservative algorithm template.
- [ ] Run a preview before long simulations.
- [ ] Compare against stable-demo and stress-test profiles.
- [ ] Generate a report and validation summary.
- [ ] Only move toward hardware after bench-only packaging checks pass.

## Commands

```bash
iints new-algo --name my_algorithm --template conservative
iints run preview --algo algorithms/my_algorithm.py --patient-config-path patients/stable_patient.yaml --scenario-path scenarios/clinic_safe_baseline.json
iints run --algo algorithms/my_algorithm.py --patient-config-path patients/stable_patient.yaml --scenario-path scenarios/clinic_safe_baseline.json --duration 1440 --output-dir results/my_algorithm_1d
```

## Open Next

- [[Algorithm to Pump Pipeline]]
- [[Physiology and Safety Sources]]
- [[Sources by SDK Feature]]
