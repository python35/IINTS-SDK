# IINTS-AF SDK Documentation

IINTS-AF is a research SDK for testing insulin algorithms on virtual patients, validating data quality, and reviewing outputs with reproducible reports.

It combines three core workflows:
- **Simulate** closed-loop behavior on virtual patients
- **Certify** datasets and outputs with built-in trust grading
- **Review** runs with reports, posters, and optional local AI tooling

!!! important "Use a Virtual Environment"
    Run SDK commands from an active Python virtual environment such as `.venv`.
    This avoids package conflicts and missing dependency issues.

## Decide What You Need

| I want to... | Start here | Why |
|---|---|---|
| Get a recommended command | `iints start` | Goal-based first-run plan |
| See something work in a few minutes | [Quickstart](QUICKSTART.md) | Fastest route to a successful first run |
| Let the CLI guide me | `iints guide` | Interactive beginner path |
| Run a zero-config example | `iints demo` | No custom files required |
| Build my own simulation command | `iints run --wizard` | Guided custom run builder |
| Install or repair the environment | [Installation](INSTALLATION.md) | Dependency and environment setup |
| Fix a failing command | [Troubleshooting](TROUBLESHOOTING.md) | Common errors and exact fixes |
| Browse the CLI surface | [Command Reference](COMMAND_REFERENCE.md) | Short command map without deep internals |
| Run reproducible studies | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) | Protocols, study bundles, comparisons |
| Bring a Pi to Maker Faire | [Maker Faire Pi Mode](MAKERFAIRE_PI.md) | Show-ready Raspberry Pi flow |
| Maintain or release the SDK | [Maintainer Guide](MAINTAINER_GUIDE.md) | Local checks, release audit, and manual upkeep |

## 5-Minute First Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp]"

iints doctor --suggest
iints demo
```

If you want to inspect the plan before running anything:

```bash
iints demo --dry-run
```

## Friendly CLI Entry Points

- `iints start`: prints the shortest path for demo, project, study, edge, or data work
- `iints guide`: asks what you want to do and points you to the right flow
- `iints demo`: zero-config first run
- `iints run --wizard`: guided custom run
- `iints doctor --full --suggest`: diagnostics plus concrete next commands

## Choose Your Path

### New Users
1. [Quickstart](QUICKSTART.md)
2. [Installation](INSTALLATION.md)
3. [Troubleshooting](TROUBLESHOOTING.md)
4. [Command Reference](COMMAND_REFERENCE.md)

### Researchers
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
- [Study Analysis](STUDY_ANALYSIS.md)
- [Evidence Base](EVIDENCE_BASE.md)

### Edge / Maker Faire Users
- [Edge Hardware & SBC Matrix](EDGE_HARDWARE.md)
- [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md)
- [Maker Faire Pi Mode](MAKERFAIRE_PI.md)
- [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md)
- [Arduino UNO Q Setup](ARDUINO_UNO_Q.md)

### Maintainers
- [Maintainer Guide](MAINTAINER_GUIDE.md)
- [Maintainer Release Checklist](PUBLIC_RELEASE_CHECKLIST.md)
- [Release Archive](releases/INDEX.md)

## Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
