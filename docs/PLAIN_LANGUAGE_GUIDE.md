# Plain Language Overview

This page explains the SDK in simple terms.

## What Is IINTS-AF?
IINTS-AF is a **safe testing environment** for insulin algorithms.

You can think of it as a flight simulator:
- You test ideas in simulation.
- You measure what happens.
- You keep an audit trail.
- You do **not** treat real patients with this SDK.

## What Happens During a Run?
When you run a simulation, the SDK does this:

1. Reads glucose and scenario events (meal, exercise, sensor issue).
2. Checks if glucose input is biologically plausible (`InputValidator`).
3. Gets an insulin suggestion from your algorithm.
4. Optionally reads AI forecast signals (advisory only).
5. Applies hard deterministic safety checks (`IndependentSupervisor`).
6. Simulates the patient response and stores all outputs.

## Why “Open Logic” Matters
IINTS-AF is not “AI decides everything.”

- Layer 1: `InputValidator` filters impossible sensor values.
- Layer 2: `Predictor` estimates future glucose (optional, advisory).
- Layer 3: `IndependentSupervisor` can block or reduce unsafe doses.

Final dosing is always safety-constrained.

## Who Should Use It?
- Researchers testing control algorithms.
- Developers building simulation pipelines.
- Clinical innovation teams preparing pre-clinical evidence.
- Students learning diabetes algorithm validation.

## What It Is Not
- Not a medical device.
- Not cleared for direct patient treatment.
- Not clinical decision support in production care.

## 5-Minute Start
```bash
pip install iints-sdk-python35
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

You will get output files like:
- `results.csv` (time-series outputs)
- `audit/` (decision trace)
- `report.pdf` (visual summary)

## “I Need X” Quick Map
- “I want a first run” -> `README.md`
- “I want full CLI commands” -> `docs/TECHNICAL_README.md`
- “I want full architecture details” -> `docs/COMPREHENSIVE_GUIDE.md`
- “I want the real research sources behind assumptions” -> `docs/EVIDENCE_BASE.md`
- “I want predictor training” -> `research/README.md`
- “I want step-by-step notebooks” -> `examples/notebooks/README.md`

## Short Glossary
- **TIR**: Time in Range (usually 70-180 mg/dL).
- **IOB**: Insulin On Board (active insulin still working).
- **COB**: Carbs On Board (carbohydrates still absorbing).
- **Fail-soft**: Keep simulation running using last safe value when input is invalid.
- **Dual-Guard**: Predictor + deterministic supervisor, with safety as final authority.
