# Manual Overview

Use this page to choose the right manual quickly.

## Environment Baseline

Before following any manual, activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

All command examples in the documentation assume `.venv` is active.

## Best Entry Points

- `GETTING_STARTED.md`: fastest route to a working run.
- `PLAIN_LANGUAGE_GUIDE.md`: non-technical explanation.
- `TECHNICAL_README.md`: command and integration reference.
- `AI_ASSISTANT.md`: how the local open-weight Ministral 3 assistant works, how it is gated, and how to debug it.

## Full Manuals

- `manuals/IINTS-AF_SDK_Manual.md`: full long-form technical manual.
- `manuals/IINTS-AF_SDK_Manual.pdf`: printable/export version.
- `COMPREHENSIVE_GUIDE.md`: architecture + workflows + SDK concepts.

## Research-Specific Manual

- `../research/README.md`: predictor data prep, training, and evaluation.

## Scope

- Research and simulation only.
- Not a medical device.
