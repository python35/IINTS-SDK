# IINTS-AF Physiology Reference Brochure

**Project:** IINTS-AF SDK

**Purpose:** Give doctors, engineers, and judges one compact document that explains which physiological quantities the SDK models, which numbers matter, and how to read them correctly.

**Audience:** Clinical reviewers, biomedical engineers, EUCYS judges.

## Why This Document Exists

Glucose simulation is easy to make visually convincing and hard to make scientifically honest. A curve can look smooth and still be physiologically weak. IINTS-AF therefore separates four ideas that are often mixed together:

| Layer | Question it answers |
|---|---|
| physiology | what happens inside the virtual patient? |
| measurement | what does the CGM-like sensor report? |
| interpretation | how do we summarize the trace? |
| protection | when does the safety layer block or stop a run? |

The SDK is strongest when those layers remain explicit rather than hidden inside one opaque score.

## The Core Physiological Story

Type 1 diabetes simulation in IINTS-AF revolves around a small set of interacting quantities:

| Quantity | Unit | Physiological role |
|---|---:|---|
| glucose | mg/dL | primary state variable shown in plots and reports |
| insulin-on-board (IOB) | U | insulin still active after delivery |
| carbs-on-board (COB) | g | carbohydrate still being absorbed |
| basal insulin rate | U/h | background insulin exposure |
| insulin sensitivity (ISF) | mg/dL/U | expected glucose effect of 1 U insulin |
| insulin-to-carb ratio (ICR) | g/U | carbohydrate amount covered by 1 U insulin |
| glucose trend | mg/dL/min | direction and steepness of change |
| exercise intensity | 0-1 | extra glucose-lowering pressure |

These quantities let the SDK represent common research questions:

- what happens after a meal?
- what happens if insulin stacks?
- what changes during exercise?
- what if the announced carbohydrates are wrong?
- what if the sensor is delayed or noisy?

## Numbers With Clinical Meaning

These numbers are used for **interpretation**, not as proof that a simulation is clinically validated.

| Quantity | Value | Why it matters |
|---|---:|---|
| target glucose range | 70-180 mg/dL | standard CGM Time-in-Range band |
| below-range threshold | <70 mg/dL | low-glucose exposure |
| level-2 hypoglycemia | <54 mg/dL | more serious low-glucose state |
| common adult TIR target | >70% | interpretable reporting benchmark |
| common adult time-below-range target | <4% below 70 mg/dL | about 58 minutes/day or less |
| common adult severe-low target | <1% below 54 mg/dL | about 15 minutes/day or less |

The SDK reports these values because they make results interpretable across studies. It does not claim that meeting one table row proves real-world safety.

## Software Safety Rails

The safety supervisor uses intentionally conservative software limits. These are **not** personalized treatment prescriptions.

| Safety rail | SDK default | Meaning inside the software |
|---|---:|---|
| max single bolus | 5.0 U | blocks an oversized one-step insulin request |
| max insulin per hour | 3.0 U | limits recent cumulative delivery |
| max insulin on board | 4.0 U | limits active insulin burden |
| falling-trend stop | -2.0 mg/dL/min | avoids dosing into a rapid fall |
| critical stop | <40 mg/dL for 30 min | terminates an unsafe synthetic trajectory |
| plausible sensor range | 40-500 mg/dL | guards impossible CGM-like input |

This separation is useful for judges: a physiological threshold says what glucose means; a supervisor threshold says what the research software is allowed to do.

## What A Meaningful Day Looks Like

A realistic virtual day should not be a decorative sine wave. It should tell a plausible physiological story.

| Event | Example time | Example magnitude | Expected effect |
|---|---:|---:|---|
| breakfast | 07:30 | 48 g carbs | first post-meal rise |
| lunch | 12:15 | 62 g carbs | larger midday disturbance |
| exercise | 12:45 | intensity 0.35 for 30 min | lowers glucose after lunch |
| dinner | 18:00 | 74 g carbs | strongest evening challenge |
| snack | 21:30 | 18 g carbs | small late excursion |

That table matters because it links the plotted glucose curve to causes a doctor or engineer can reason about.

## Starter Profiles

The SDK now ships with explicit first-use profiles instead of leaving beginners with ambiguous defaults.

| Profile | Initial glucose | Basal | ISF | ICR | Drift | Best use |
|---|---:|---:|---:|---:|---:|---|
| `stable-demo` | 130 mg/dL | 0.2 U/h | 40 mg/dL/U | 15 g/U | 0.001 | demos and smoke tests |
| `stress-test` | 120 mg/dL | 0.5 U/h | 50 mg/dL/U | 10 g/U | 0.003 | harder disturbances |
| `endurance` | 140 mg/dL | 0.0 U/h | 20 mg/dL/U | 25 g/U | 0.0 | long unattended software runs |

The most important design lesson is that a harmless-looking coefficient such as a glucose-decay setting can completely change the trajectory. A simulator is only as honest as its defaults.

## Measurement Is Not Physiology

Algorithms usually see a CGM-like signal, not perfect latent glucose. IINTS-AF therefore models sensor behavior separately.

| Sensor profile | Noise | Lag | Typical use |
|---|---:|---:|---|
| `ideal` | 0 mg/dL | 0 min | debugging |
| `clinical_cgm` | 7 mg/dL | 10 min | clean realistic baseline |
| `free_living_cgm` | 8 mg/dL | 10 min | free-living realism |
| `compression_prone` | 8.5 mg/dL | 12 min | artifact-heavy stress testing |

This makes an important scientific distinction possible:

- true glucose can be physiologically stable
- the reported glucose can still be noisy, delayed, or temporarily wrong

## Model Families

| Model family | Strength | Best use |
|---|---|---|
| simplified custom model | transparent, fast, stress-test friendly | demos, regressions, safety sweeps |
| Bergman-style model | more mechanistic internal states | physiology-focused experiments |
| empirical residual layer | adds day-scale irregularity | realism studies and synthetic mirrors |

No single model is “the real patient.” The scientific goal is to keep assumptions visible and compareable.

## What The SDK Still Does Not Fully Model

| Limitation | Current status |
|---|---|
| glucagon and counter-regulatory hormones | not explicit |
| illness, infection, steroids, menstrual-cycle effects | scenario-level approximation only |
| mixed fat/protein meal kinetics | only coarse delayed-meal approximations |
| formulation-specific insulin pharmacokinetics | simplified configurable timing |
| individualized real-patient digital twins | not claimed |

Those omissions are not hidden defects; they define the boundary of what the SDK can honestly claim today.

## How To Explain The SDK In One Minute

1. IINTS-AF does not dose real patients; it is a research simulator.
2. It models glucose, insulin action, meals, exercise, and CGM imperfections separately.
3. It reports standard CGM metrics such as Time in Range and hypoglycemia exposure.
4. It wraps algorithms in a deterministic safety supervisor so risky outputs become visible and auditable.
5. It is useful because it turns insulin-algorithm development from “one pretty graph” into reproducible evidence.

## Best One-Sentence Claim

**IINTS-AF is a pre-clinical research SDK that keeps virtual-patient physiology, CGM measurement behavior, and safety supervision explicit so insulin algorithms can be tested more honestly before any real-world claim is made.**

## Source Backbone

This brochure follows the maintained source legend in `docs/EVIDENCE_BASE.md`, including the ADA Standards of Care 2026, the international Time-in-Range consensus, OhioT1DM, Dalla Man/Cobelli meal-model literature, UVA/Padova simulator literature, and the 2024 Communications Medicine paper on generative T1D simulation.
