# IINTS-AF SDK

[![EUCYS 2026](https://img.shields.io/badge/EUCYS-2026%20Selected-gold?style=flat)](https://www.uni-kiel.de/en/eucys2026)
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-IINTS--AF-0a66c2?style=flat)](https://python35.github.io/IINTS-SDK/)

IINTS-AF is an open-source research SDK for diabetes technology experiments.

It lets you create a virtual diabetes scenario, run an insulin-algorithm simulation, inspect the glucose result, and generate evidence files such as CSV outputs, reports, AGP-style figures, and audit records.

The goal is simple: make it easier to test and discuss diabetes algorithm behavior safely before anything is connected to a real person.

IINTS-AF is research and education software. It is not a medical device and must not be used for diagnosis, insulin dosing, treatment decisions, or real-time patient care.

## What The SDK Does

- Simulates virtual diabetes scenarios, meals, insulin delivery, sensors, and safety events.
- Runs example insulin algorithms against those scenarios.
- Generates results as CSV files, PDF reports, AGP-style reports, posters, and audit artifacts.
- Helps compare software logic, safety checks, physiology assumptions, and AI/glucose-prediction experiments.
- Provides command-line tools, a native desktop app, and hardware/demo helpers for Raspberry Pi, Jetson, FPGA, and Arduino-style experiments.

## Who It Is For

- Students and researchers learning about diabetes technology.
- Developers testing insulin-algorithm ideas in simulation.
- Teachers or demo builders who need understandable digital-patient examples.
- Reviewers who want reproducible outputs instead of screenshots without data.

## What It Is Not

- It is not a clinical simulator validated for patient care.
- It is not an insulin pump controller.
- It is not a replacement for clinical trials, medical review, or regulatory approval.
- It does not give medical advice.

## Install

```bash
pip install "iints-sdk-python35[full,mdmp]"
iints doctor --smoke-run
```

For the native desktop app:

```bash
pip install "iints-sdk-python35[full,desktop-qt,mdmp]"
iints-desktop
```

## Quick Start

Create a small project and run a baseline simulation:

```bash
iints quickstart --project-name my_study
cd my_study
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

Run a clear demo workflow:

```bash
iints demo eucys --output-dir results/live_demo
```

Other useful demo modes:

```bash
iints demo doctor --output-dir results/doctor_demo
iints demo booth --output-dir results/booth_demo
```

## Common Outputs

A normal run can create files such as:

```text
results/
├── results.csv
├── clinical_report.pdf
├── agp_report.png
├── audit.json
└── summary.md
```

These outputs are meant to make a run reviewable and reproducible.

## Main Commands

```bash
iints doctor
```

Check the installation.

```bash
iints presets list
```

Show available simulation presets.

```bash
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

Run a preset with an algorithm.

```bash
iints report --results results/results.csv
```

Generate a report from results.

```bash
iints-desktop
```

Open the native desktop workbench.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/iints/` | SDK source code |
| `src/iints_desktop/` | native desktop app wrapper |
| `docs/` | public documentation |
| `tests/` | automated tests |
| `examples/` | runnable examples |
| `algorithms/` | example algorithm entry points |
| `tools/` | build, release, and maintenance helpers |
| `data_packs/` | small bundled/demo data packs |

## Documentation

Full documentation is available at:

[python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/)

Useful pages:

- [Quickstart](https://python35.github.io/IINTS-SDK/QUICKSTART/)
- [Desktop App](https://python35.github.io/IINTS-SDK/DESKTOP_APP/)
- [Command Cheatsheet](https://python35.github.io/IINTS-SDK/CLI_CHEATSHEET/)
- [Updating The SDK](https://python35.github.io/IINTS-SDK/UPDATING/)

## Desktop App Branch

The main branch contains the full SDK. A separate `desktop-app` branch is maintained as an app-focused branch for users who mainly want the native desktop experience and app installation instructions.

```bash
git clone --branch desktop-app https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python -m pip install -U -e ".[full,desktop-qt,mdmp]"
iints-desktop
```

## License

Apache-2.0 licensed, with legacy MIT notices where applicable.

Built as research software by Rune Bobbaers.
