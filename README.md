# IINTS-AF SDK

[![EUCYS 2026](https://img.shields.io/badge/EUCYS-2026%20Selected-gold?style=flat)](https://www.uni-kiel.de/en/eucys2026)
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-IINTS--AF-0a66c2?style=flat)](https://python35.github.io/IINTS-SDK/)

> "Code shouldn't be a secret when it's managing a life."

Open-source research platform for insulin delivery algorithm simulation and validation.

---

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/iints/` | installable SDK source code |
| `docs/` | public documentation, governance, release notes, and Obsidian vault mirror |
| `examples/` | runnable examples and notebooks |
| `tests/` | unit, CLI, system, and realism tests |
| `research/` | research artifacts and benchmark snapshots |
| `tools/` | maintainer scripts for docs, CI, and release workflows |
| `data_packs/` | small bundled/demo data packs |
| `algorithms/` | example insulin algorithm entry points |

Important starting points:

- `docs/THEORY_STRESS_LAB.md` — Jetson scientific stress-test mode
- `docs/governance/API_STABILITY.md` — semver and API policy
- `docs/governance/PRIVACY_POLICY.md` and `docs/governance/TERMS_OF_USE.md` — public legal docs

---

## What It Does

- **Simulate** virtual patients across thousands of scenarios before any algorithm reaches a real device
- **Certify** datasets cryptographically — traceable, consented, reproducible
- **Understand** results with audit-ready reports and local AI explanation via Ministral

---

## Install

```bash
pip install "iints-sdk-python35[full,mdmp]"
iints doctor --smoke-run
```

**Raspberry Pi / Arduino UNO Q:**
```bash
pip install "iints-sdk-python35[edge,mdmp]"
iints edge quickstart --board raspberry_pi
iints edge quickstart --board uno_q
```

---

## Quick Start

```bash
iints quickstart --project-name my_study
cd my_study
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
iints ai report results/<run_id>
```

Full documentation: [python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/)

---

## Live Demo

For a Zoom call, jury walkthrough, or sponsor demo where you want the story first and code as proof:

```bash
iints demo eucys --output-dir results/live_demo
```

Use `iints demo doctor` for clinical feedback conversations and `iints demo booth` for public digital-patient demos.

---

> Research software. Not a medical device. MIT Licensed.  
> *Built by a 17-year-old with type 1 diabetes.*
