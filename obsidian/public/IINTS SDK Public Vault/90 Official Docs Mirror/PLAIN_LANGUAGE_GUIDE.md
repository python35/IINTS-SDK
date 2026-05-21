# Plain-Language Overview

Use this page if you want to understand the SDK before reading commands, code, or research detail.

## What IINTS-AF Is

IINTS-AF is a research sandbox for testing insulin algorithms without treating real patients.

The simplest analogy is a flight simulator:

- it creates realistic test situations,
- lets an algorithm respond,
- records what happened,
- and keeps the final safety logic visible and inspectable.

It is built for research, education, validation, and technical review. It is not a medical device and does not provide clinical dosing advice.

## What Happens During A Run

1. The simulator reads glucose and scenario events such as meals, exercise, or sensor issues.
2. An input validator rejects values that are biologically impossible.
3. An algorithm proposes an insulin action.
4. An optional predictor may estimate future glucose, but only as advisory information.
5. A deterministic safety supervisor can block or reduce unsafe actions.
6. The simulator stores the results, reports, and audit trail.

## Why The Safety Design Matters

IINTS-AF is not built around “AI decides everything.”

| Layer | Role |
| --- | --- |
| `InputValidator` | rejects impossible sensor values |
| optional predictor | estimates future glucose trends |
| `IndependentSupervisor` | applies final deterministic safety limits |

The supervisor remains the final authority. That separation is the central design choice of the SDK.

## Who It Helps

- researchers comparing insulin-control strategies
- developers building simulation or validation workflows
- clinical-innovation teams preparing pre-clinical evidence
- students learning how safety-constrained algorithms are evaluated

## The Fastest First Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"
iints demo
```

That gives you:

- `results.csv` for the simulated time series
- `report.pdf` for a readable summary
- `audit/` for the safety trail

If you want the full first workflow rather than only a demo, continue to [Getting Started](GETTING_STARTED.md).

## Where To Go Next

| If you want to... | Continue with |
| --- | --- |
| get one successful run quickly | [Quickstart](QUICKSTART.md) |
| understand the full first workflow | [Getting Started](GETTING_STARTED.md) |
| inspect the science behind the claims | [Evidence Base](EVIDENCE_BASE.md) |
| train or evaluate predictors | `../research/README.md` |
| browse exact commands | [Command Reference](COMMAND_REFERENCE.md) |

## Short Glossary

- **TIR**: Time in Range, commonly 70-180 mg/dL.
- **IOB**: Insulin On Board, meaning active insulin still working.
- **COB**: Carbs On Board, meaning carbohydrates still absorbing.
- **Fail-soft**: keep the simulation running with the last safe value when input is invalid.
- **Dual-Guard**: optional predictor plus deterministic supervisor, with safety remaining final.
