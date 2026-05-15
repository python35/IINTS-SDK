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

- `USER_GUIDE_MAP.md`: connected route through the public docs for external users.
- `DEVELOPER_PORTAL.md`: technical entry point for contributors who modify the SDK itself.
- `ARCHITECTURE_OVERVIEW.md`: visual map of subsystem boundaries and run flow.
- `API_REFERENCE.md`: generated public-symbol map for every Python module.
- `CONTRIBUTING_SAFELY.md`: safe contribution workflow and validation matrix.
- `GETTING_STARTED.md`: fastest route to a working run.
- `PLAIN_LANGUAGE_GUIDE.md`: non-technical explanation.
- `TECHNICAL_README.md`: command and integration reference.
- `AI_ASSISTANT.md`: how the local open-weight Ministral 3 assistant works, how it is gated, and how to debug it.

## Full Manuals

- `manuals/IINTS-AF_SDK_Manual.md`: full long-form technical manual.
- `manuals/IINTS-AF_SDK_Manual.pdf`: printable/export version.
- `COMPREHENSIVE_GUIDE.md`: architecture + workflows + SDK concepts.

## Recommended Reading Order

For a new external user:

1. [Choose Your Path](USER_GUIDE_MAP.md)
2. [Quickstart](QUICKSTART.md)
3. [Getting Started](GETTING_STARTED.md)
4. [Command Reference](COMMAND_REFERENCE.md)

For a research user:

1. [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
2. [Study Analysis](STUDY_ANALYSIS.md)
3. [MDMP Quickstart](MDMP_QUICKSTART.md)

For an edge user:

1. [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md)
2. [Remote Deploy & Pi Connect](EDGE_REMOTE_DEPLOY.md)
3. [Maker Faire Pi Mode](MAKERFAIRE_PI.md)

## Research-Specific Manual

- `../research/README.md`: predictor data prep, training, and evaluation.

## Scope

- Research and simulation only.
- Not a medical device.
