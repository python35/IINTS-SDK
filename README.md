# IINTS-AF SDK

[![EUCYS 2026](https://img.shields.io/badge/EUCYS-2026%20Selected-gold?style=flat)](https://www.uni-kiel.de/en/eucys2026)
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-IINTS--AF-0a66c2?style=flat)](https://python35.github.io/IINTS-SDK/)

**IINTS-AF** is an open-source research SDK for closed-loop diabetes-technology simulation, insulin-dosing algorithm benchmarking, and safety-supervisor evaluation. It combines mechanistic glucose–insulin physiology models (Bergman minimal model and 19-state Hovorka-style model), realistic CGM/pump fault injection, deterministic safety verification, and reproducible reporting.

> **Research-only boundary:** IINTS-AF is **not a medical device**. It must not be used for diagnosis, insulin dosing, treatment decisions, or real-time patient care.

---

## Quick Start

```bash
python -m pip install -U "iints-sdk-python35[full,mdmp,research,edge]"
iints demo
iints menu
iints version --refresh
```

For local development and running the test suite:

```bash
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
make dev
make test
```

---

## Repository Structure

| Directory / File | Purpose |
| :--- | :--- |
| [`src/iints/`](src/iints/) | Core Python SDK: physiology models (`core/patient/`), controllers (`core/algorithms/`), commercial AID emulations (`emulation/`), safety supervisor (`core/supervisor.py`), metrics, and CLI. |
| [`sources/`](sources/) | **Curated academic literature library** on closed-loop insulin pump (AID) algorithms, open-source AID, simulators, fault tolerance, regulation, and datasets — with verified DOIs and [`references.bib`](sources/references.bib). |
| [`research/`](research/) | Reproducible research pipelines: dataset preparation, glucose forecasting, simulator realism calibration, evidence exports, and benchmark artifacts. |
| [`algorithms/`](algorithms/) | Starter template for implementing and testing custom closed-loop insulin algorithms. |
| [`examples/`](examples/) | Guided demos ([`examples/demos/`](examples/demos/)), Jupyter notebooks ([`examples/notebooks/`](examples/notebooks/)), and analysis scripts (see [`examples/README.md`](examples/README.md)). |
| [`docs/`](docs/) | Full MkDocs documentation site, formula registry, governance dossier, experimental results ([`EXPERIMENTAL_RESULTS.md`](docs/EXPERIMENTAL_RESULTS.md)), and complete SDK bibliography. |
| [`tests/`](tests/) | Automated unit, regression, safety, governance, and performance test suites. |
| [`apps/iints-tauri/`](apps/iints-tauri/) | Optional native Rust/Tauri research workbench shell around the Python SDK. |
| [`tools/`](tools/) & [`scripts/`](scripts/) | CI governance gates, documentation generators, and helper scripts. |
| [`paper.md`](paper.md) & [`paper.bib`](paper.bib) | Short software paper and BibTeX references (see also [`CITATION.cff`](CITATION.cff)). |

---

## Closed-Loop AID Research & Literature (`sources/`)

If you are using this repository for university research on open-source closed-loop insulin pump algorithms, start in [`sources/README.md`](sources/README.md):

1. [`01_commercial_aid_algorithms.md`](sources/01_commercial_aid_algorithms.md) — Published mathematical foundations of commercial systems (PID-IFB, Zone-MPC, Hovorka NMPC, Omnipod 5, iLet).
2. [`02_open_source_aid_and_clinical_trials.md`](sources/02_open_source_aid_and_clinical_trials.md) — Open-source AID (`oref0`/`oref1`, OpenAPS, AndroidAPS, Loop, Tidepool Loop), CREATE RCT, and international consensus.
3. [`03_physiology_simulators_and_digital_twins.md`](sources/03_physiology_simulators_and_digital_twins.md) — Bergman, Hovorka, UVA/Padova, `simglucose`, `ReplayBG`, and glycemic consensus metrics.
4. [`04_control_theory_and_ai_rl.md`](sources/04_control_theory_and_ai_rl.md) — Control-engineering reviews, Safe/Offline Reinforcement Learning, Dual-Hormone AP, and CGM foundation models.
5. [`05_safety_fault_tolerance_and_regulation.md`](sources/05_safety_fault_tolerance_and_regulation.md) — Compression lows (PISA), pump occlusions, safety supervisors, FDA iAGC, and EU MDR pathways.
6. [`06_datasets_and_benchmarks.md`](sources/06_datasets_and_benchmarks.md) — Real-world T1D datasets (OhioT1DM, T1DEXI, OpenAPS Data Commons, DiaTrend, AZT1D, HUPA-UCM, MetaboNet).
7. [`references.bib`](sources/references.bib) — Verified BibTeX bibliography ready for LaTeX, Overleaf, or Zotero.

---

## Desktop App (Optional)

The optional Rust/Tauri research workbench offers guided simulation runs, result inspection, local AI review, and MDMP data-assurance tools.

| Platform | Latest beta |
| :--- | :--- |
| Windows | [`.exe` installer](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-windows-x64-setup.exe) |
| macOS | [`.dmg`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-macos.dmg) |
| Linux | [`.AppImage`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-linux-x64.AppImage) |

---

## Documentation & Links

- **Website:** [iints.org](https://iints.org)
- **Documentation:** [python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/) (source in [`docs/`](docs/))
- **Desktop guide:** [Tauri Research Workbench](https://python35.github.io/IINTS-SDK/TAURI_DESKTOP/)
- **Source code:** [github.com/python35/IINTS-SDK](https://github.com/python35/IINTS-SDK)
- **Security policy:** [`SECURITY.md`](SECURITY.md) · **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Citation & License

If you use IINTS-AF in academic work, please cite [`CITATION.cff`](CITATION.cff).

Licensed under **Apache-2.0** ([`LICENSE`](LICENSE)), with legacy MIT notices where applicable ([`LICENSE-MIT-IINTS-LEGACY`](LICENSE-MIT-IINTS-LEGACY)).
