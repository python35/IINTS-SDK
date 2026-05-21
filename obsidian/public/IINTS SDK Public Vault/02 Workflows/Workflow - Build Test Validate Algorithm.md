---
tags:
  - iints/workflow
  - iints/algorithm
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Workflow - Build Test Validate Algorithm

## Goal

Create an algorithm, test it in simulation, then generate evidence that is reproducible and not overclaimed.

## Gates

| Gate | Question | Pass condition |
| --- | --- | --- |
| Template | Does the algorithm run at all? | no import/runtime errors |
| Preview | Does the config look sane before long run? | no immediate hypo/crash warnings |
| Short run | Can it complete a short scenario? | completed, no critical termination |
| Day run | Can it complete a 24h baseline? | results + safety summary generated |
| Validation | Do metrics pass the chosen profile? | validation JSON/markdown exists |
| Explanation | Can you explain failures honestly? | [[Claim Library]] updated |

## Commands

```bash
iints new-algo --name my_algorithm --template conservative
iints run preview --algo algorithms/my_algorithm.py --patient-config-path patients/stable_patient.yaml --scenario-path scenarios/clinic_safe_baseline.json
iints run --algo algorithms/my_algorithm.py --patient-config-path patients/stable_patient.yaml --scenario-path scenarios/clinic_safe_baseline.json --duration 1440 --output-dir results/my_algorithm_1d
iints validate-run --run-dir results/my_algorithm_1d
```

## Source Links

- [[Physiology and Safety Sources]]
- [[Sources by SDK Feature]]
- [[Realism Reference Envelopes]]
