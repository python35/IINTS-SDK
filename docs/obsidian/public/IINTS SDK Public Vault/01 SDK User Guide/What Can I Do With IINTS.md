---
tags:
  - iints/user
  - iints/sdk
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# What Can I Do With IINTS?

IINTS is a research SDK for building, testing, explaining, and auditing insulin-algorithm ideas **before** anything touches real hardware or real therapy.

## User Capabilities

| Capability | What you get | Main command |
| --- | --- | --- |
| First demo | a complete run with outputs | `iints demo` |
| Quickstart project | local algorithm, patient, scenario, README | `iints quickstart` |
| Algorithm testing | simulate a Python algorithm against virtual patients | `iints run` |
| Preflight check | catch unsafe configs before running | `iints run preview` / `iints validate` |
| Evidence bundle | results, metadata, report, source manifest | `iints certify-run` / `iints study-ready` |
| Data import | convert CGM/pump exports into IINTS format | `iints import-data`, `iints import-carelink`, `iints import-tidepool` |
| Data realism | compare synthetic/real traces with plausibility bands | `iints data realism-check` |
| Study matrix | run reproducible benchmark studies | `iints run-study`, `iints run-eucys-study` |
| Poster/demo | produce showable outputs for booths and calls | `iints demo-live`, `iints demo-booth`, `iints poster-study` |
| Edge demo | run local edge/Jetson/patient demos | `iints edge`, `iints jetson`, `iints patient` |
| Pump lab | package bench-only Pico artifacts | `iints edge pump` |

## What IINTS Is Not

- Not a medical device.
- Not treatment advice.
- Not an autonomous dosing system for humans.
- Not proof that an algorithm is clinically safe.

## The Simple Story

You write an algorithm, run it through simulated scenarios, validate the outputs, generate reports, and keep all sources/configs traceable.
