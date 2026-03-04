# IINTS SDK Demo Scripts

These scripts are reference-quality examples for showing the SDK to developers, researchers, clinicians, and industry partners.

## Design Goals

- One concept per script.
- Explicit comments and readable structure.
- Safe-by-default behavior (deterministic supervisor always active).
- Reproducible outputs under `results/`.

## Demo Index

1. `01_basic_simulation.py`  
   Build a custom algorithm class and run a full simulation pipeline.
2. `02_dual_guard_predictor.py`  
   Add predictor-driven advisory forecasts with uncertainty + OOD gates.
3. `03_validation_and_contracts.py`  
   Validate run metrics against a profile and verify the formal safety contract.
4. `04_scenario_bank_scorecard.py`  
   Benchmark one algorithm across multiple preset scenarios and generate a scorecard.
5. `05_open_logic_architecture.py`  
   Presentation-focused proof of the three-layer “Open Logic” architecture.

## Prerequisites

- From repository root.
- Core deps installed.
- For Demo 02 (predictor): install research extras and provide a predictor checkpoint.

## Run Commands

```bash
PYTHONPATH=src python3 examples/demos/01_basic_simulation.py
PYTHONPATH=src python3 examples/demos/02_dual_guard_predictor.py --predictor models/hupa_finetuned_v2/predictor.pt
PYTHONPATH=src python3 examples/demos/03_validation_and_contracts.py
PYTHONPATH=src python3 examples/demos/04_scenario_bank_scorecard.py --max-presets 3
PYTHONPATH=src python3 examples/demos/05_open_logic_architecture.py
```

All scripts write artifacts in `results/` and print a compact machine-readable summary to stdout.
