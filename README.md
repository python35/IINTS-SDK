# IINTS-AF SDK
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![Python Package CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-IINTS--AF-0a66c2?style=flat&logo=firefox-browser&logoColor=white)](https://python35.github.io/IINTS-SDK/)

> "Code shouldn't be a secret when it's managing a life."

Insulin pumps make hundreds of autonomous decisions about drug delivery every day.  
The algorithms behind those decisions are proprietary, unauditable, and difficult to inspect or improve -- even by the patients whose lives depend on them.

IINTS-AF is an open-source research platform that changes that.

## Core Research Question

**Can open-source simulation and deterministic safety supervision make insulin delivery algorithm development safer and more transparent for researchers and patients?**

---

## What It Does

**Simulate** -- Run virtual patients through thousands of scenarios before any algorithm reaches a real device. A deterministic safety supervisor audits every AI decision. The AI may suggest. The supervisor decides.

**Certify** -- Every dataset is fingerprinted and graded before it touches a study workflow. The goal is to keep benchmark inputs traceable, reviewable, and reproducible.

**Understand** -- Generate audit-ready reports, visual posters, and local AI summaries from the same study bundle. IINTS-AF can use local models such as Ministral for explanation workflows on your own hardware.

---

## Research Results

Final locked benchmark: `3600` simulation runs, `6` profiles, `4` scenario families, `5` algorithms, `10` fixed seeds.

| Metric | ExampleAlgorithm | PID Baseline | Delta |
|---|---:|---:|---:|
| Time in Range | 87.16% | 83.72% | +3.44% |
| Time < 70 mg/dL | 1.28% | 5.25% | -3.97% |
| Supervisor interventions | 99 | 177 | -78 |

Additional benchmark result:
- clean certified conditions showed `+17.64` Time-in-Range points versus corrupted uncertified conditions

For the full scientific write-up, see:
- `research/EUCYS_REPORT.md`
- `research/EUCYS_REPORT.pdf`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_SUMMARY.md`

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U "iints-sdk-python35[full,mdmp]"
iints doctor --smoke-run
```

**Edge devices (Raspberry Pi 5, Arduino UNO Q):**

```bash
pip install -U "iints-sdk-python35[edge,mdmp]"
```

---

## Quick Start

```bash
iints quickstart --project-name my_study
cd my_study
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
iints data certify contracts/clinical_mdmp_contract.yaml data/demo/diabetes_cgm.csv --output-json audit/certification.json
iints ai report results/<run_id>
```

---

## Final Benchmark Workflow

```bash
tools/research/run_eucys_final.sh \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10
```

Then render the report:

```bash
tools/research/render_eucys_report_pdf.sh
```

Main final artifacts:
- `results/eucys_2026/`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.png`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_RESULTS_TABLE.csv`
- `research/EUCYS_REPORT.md`
- `research/EUCYS_REPORT.pdf`

---

## Live Digital Patient (Raspberry Pi)

```bash
iints patient start \
  --algo algorithms/example_algorithm.py \
  --scenario-profile expo_hot_start \
  --mode demo-time --speed 60x
```

Open `http://127.0.0.1:8765/dashboard` -- a virtual patient running continuously, reacting to meals, exercise, and sleep in real time.

---

## Documentation

| | |
|---|---|
| Getting started | [python35.github.io/IINTS-SDK/GETTING_STARTED/](https://python35.github.io/IINTS-SDK/GETTING_STARTED/) |
| Edge hardware | [python35.github.io/IINTS-SDK/EDGE_HARDWARE/](https://python35.github.io/IINTS-SDK/EDGE_HARDWARE/) |
| Raspberry Pi setup | [python35.github.io/IINTS-SDK/DIGITAL_PATIENT_PI/](https://python35.github.io/IINTS-SDK/DIGITAL_PATIENT_PI/) |
| Data certification | [python35.github.io/IINTS-SDK/MDMP_QUICKSTART/](https://python35.github.io/IINTS-SDK/MDMP_QUICKSTART/) |
| Full manual | [python35.github.io/IINTS-SDK/manuals/IINTS-AF_SDK_Manual.pdf](https://python35.github.io/IINTS-SDK/manuals/IINTS-AF_SDK_Manual.pdf) |
| Research report | [research/EUCYS_REPORT.md](research/EUCYS_REPORT.md) |

---

> IINTS-AF is research software. Not a medical device.  
> No clinical dosing advice.  
>  
> MIT Licensed -- built by a 17-year-old with type 1 diabetes who wanted to understand the device managing his life.
