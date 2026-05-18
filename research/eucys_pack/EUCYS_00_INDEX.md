# IINTS-AF EUCYS Evidence Pack

**Project:** IINTS-AF SDK

**Author:** Runebob Baers

**Competition:** EUCYS 2026

**Pack version:** 2026-05-10

**Repository commit:** 935406c

**Status:** Jury-facing explanation pack built from the maintained SDK repository

## What This Pack Is

This pack explains IINTS-AF as a scientific software project, not just as a demo. It is designed for judges who need to understand three things quickly:

- what problem the SDK solves
- how the SDK works technically
- why glucose physiology and data quality make this problem hard

The full benchmark report is still the primary results document. This evidence pack is the companion set of short PDFs that explain the system around the numbers.

## Recommended Reading Order

| Order | Document | Best for |
|---:|---|---|
| 1 | `EUCYS_00_INDEX.pdf` | Fast orientation |
| 2 | `EUCYS_02_PHYSIOLOGY_AND_DATA_BRIEF.pdf` | Why diabetes simulation is difficult |
| 3 | `EUCYS_01_TECHNICAL_BRIEF.pdf` | How the SDK is built |
| 4 | `EUCYS_03_IMPACT_ETHICS_AND_MAINTENANCE.pdf` | Why the project matters and how it is governed |
| 5 | `EUCYS_04_JURY_QA.pdf` | Fast answers during judging |
| 6 | `EUCYS_05_PHYSIOLOGY_REFERENCE_BROCHURE.pdf` | Standalone physiological reference for doctors and engineers |
| 7 | `EUCYS_06_JURY_PHYSIOLOGY_BRIEF.pdf` | Fast handout during a live conversation |
| 8 | `research/EUCYS_REPORT.pdf` | Full benchmark results |

## One-Minute Project Story

Type 1 diabetes requires constant decision-making around glucose, food, insulin, exercise, sensor data, and uncertainty. A wrong insulin decision can be dangerous, so research software in this domain should be reproducible, auditable, and safety-aware.

IINTS-AF is an open-source SDK for testing insulin decision systems on virtual patients. It provides simulation, baseline comparison, data-quality validation, safety supervision, result packaging, and edge-device workflows. The goal is not to replace clinicians or medical devices. The goal is to make pre-clinical algorithm research more transparent and easier to review.

The core idea is simple:

**Before an insulin algorithm is trusted, it should be tested against realistic virtual-patient scenarios, corrupted data, safety ablations, and reproducible baselines.**

## What Is New In This Project

IINTS-AF combines components that are often separated across research prototypes:

| Layer | What IINTS-AF Adds |
|---|---|
| Simulation | Virtual patients, meals, exercise, scenarios, run bundles |
| Algorithm comparison | Candidate controller plus explicit baselines |
| Data quality | MDMP certification and realism checks before training/evaluation |
| Safety | Deterministic supervisor guarding risky decisions |
| Reproducibility | Fixed seeds, protocol bundles, manifests, exact CLI paths |
| Review | Markdown/PDF reports, CSV tables, figures, judge-facing packs |
| Edge deployment | Raspberry Pi, UNO Q, Jetson-oriented workflows for demonstrators |

## Final Benchmark Snapshot

The final EUCYS benchmark bundle executed `3600` locked runs across profiles, scenarios, algorithms, study arms, and seeds.

| Metric | Result |
|---|---:|
| Total runs | 3600 |
| Overall mean Time in Range | 81.42% |
| 95% CI for overall TIR | 81.09 to 81.75 |
| Clean-arm candidate TIR | 93.04% |
| Clean vs corrupted TIR delta | +17.11 points |
| Python test suite after pack work | 369 passed, 4 skipped |
| GitHub Actions after latest push | Python CI, Health Badges, Notebook Bake all succeeded |

![EUCYS main benchmark figure](assets/EUCYS_MAIN_FIGURE.png)

## How To Verify The Main Evidence

The final benchmark can be reproduced with:

```bash
tools/research/run_eucys_final.sh \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --no-prepare-ai
```

The report PDF can be regenerated with:

```bash
tools/research/render_eucys_report_pdf.sh
```

This pack can be regenerated with:

```bash
tools/research/build_eucys_pack.sh
```

## What This Project Does Not Claim

IINTS-AF is not a medical device. It does not prescribe insulin to real patients. It is not a replacement for clinical trials or certified automated insulin delivery systems.

The correct claim is narrower and stronger:

**IINTS-AF is a reproducible open research platform for testing and explaining insulin decision algorithms before they ever reach real-world deployment.**

## Source Backbone

The pack uses the source legend already maintained in `docs/EVIDENCE_BASE.md`, including ADA Standards of Care 2026, the international Time in Range consensus, OhioT1DM, physiological simulation literature, UVA/Padova simulator literature, and recent work on generative type 1 diabetes simulators.
