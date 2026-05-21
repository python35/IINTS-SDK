---
tags:
  - iints/user
  - iints/commands
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Command Cookbook - User Edition

## Health Check

```bash
iints doctor
iints check-deps
iints sources
```

## First Run

```bash
iints demo
iints quickstart --output-dir iints_quickstart
cd iints_quickstart
iints run preview --algo algorithms/example_algorithm.py --patient-config-path patients/stable_patient.yaml --scenario-path scenarios/clinic_safe_baseline.json
```

## Real Run Bundle

```bash
iints run --algo algorithms/example_algorithm.py --patient-config-path patients/stable_patient.yaml --scenario-path scenarios/clinic_safe_baseline.json --duration 1440 --output-dir results/one_day
iints validate-run --run-dir results/one_day
iints report results/one_day/results.csv --output results/one_day/report.pdf
```

## Study / EUCYS

```bash
iints run-eucys-study --algo algorithms/example_algorithm.py --output-dir results/eucys_study
iints eucys-results --study-dir results/eucys_study --output-dir results/eucys_bundle
iints poster-study --study-dir results/eucys_study --output results/eucys_poster.png
```

## Data

```bash
iints data list
iints data info azt1d
iints data cite azt1d
iints import-carelink export.csv --output-dir data/carelink_import
iints import-tidepool --help
```

## Edge / Hardware

```bash
iints edge doctor
iints jetson doctor
iints edge pump init --output-dir pico_pump_lab
iints edge pump package --project-dir pico_pump_lab
iints edge pump upload --package pico_pump_lab/dist/pico_pump_lab_package --dry-run
```

> [!danger] Pump warning
> The pump commands are for bench-only packaging and dry-run workflows. Do not connect them to real insulin delivery.
