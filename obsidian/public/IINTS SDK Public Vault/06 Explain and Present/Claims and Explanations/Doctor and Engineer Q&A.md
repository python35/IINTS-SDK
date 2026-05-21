---
tags:
  - iints/qa
  - iints/claims
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Doctor and Engineer Q&A

## Questions From Doctors

| Question | Answer |
| --- | --- |
| Is this a medical device? | No. IINTS is research/education software for pre-clinical simulation and evidence bundles. |
| Does it recommend insulin? | No. It can simulate algorithm proposals in virtual scenarios, but it must not guide real treatment. |
| Why are TIR/TBR/TAR used? | They are standard CGM-style metrics; see [[ada_2026_glycemic_goals]] and [[attd_2019_time_in_range]]. |
| How do you avoid overclaiming? | Each report should include the safety boundary and sources from [[Source Library Index]]. |

## Questions From Engineers

| Question | Answer |
| --- | --- |
| What does the CLI do? | It creates projects, runs simulations, imports data, validates results, and generates reports. |
| Can I add my own algorithm? | Yes, start with [[Workflow - Build Test Validate Algorithm]]. |
| Can it run on edge devices? | It has edge/Jetson/patient tooling; see [[Workflow - Jetson Long Run]]. |
| Can it upload to a pump? | Only bench-only Pico packaging/dry-run workflows; see [[Workflow - Pump Lab Bench Gate]]. |
