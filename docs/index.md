# IINTS-AF SDK Documentation

IINTS-AF is a research and education SDK for insulin-algorithm simulation, glucose-data quality review, local AI experiments, and bench-only hardware workflows.

!!! warning "Scope"
    IINTS-AF is not a medical device, does not provide clinical dosing advice, and is not intended for real insulin delivery. Use it for simulation, teaching, benchmarking, documentation, and controlled bench research.

## Start Here

| Need | Page | Command |
| --- | --- | --- |
| Install and verify the SDK | [Quickstart](QUICKSTART.md) | `iints doctor --smoke-run` |
| Choose the right workflow | [Choose Your Path](USER_GUIDE_MAP.md) | `iints guide` |
| Look up practical commands | [Command Cheatsheet](CLI_CHEATSHEET.md) | `iints --help` |
| Prepare a live demonstration | [Booth Demo & Presentation](BOOTH_DEMO.md) | `iints demo eucys` |
| Keep the install current | [Updating The SDK](UPDATING.md) | `iints update` |
| Explain the safety boundary | [Project Boundaries](PROJECT_BOUNDARIES.md) | `iints safety-visualize` |
| Understand sources and assumptions | [Complete Source Library](SOURCE_LIBRARY.md) | `iints sources` |
| Work with hardware | [Hardware Hub](HARDWARE.md) | `iints edge doctor` |

## What The SDK Covers

| Area | What it does | Main pages |
| --- | --- | --- |
| Simulation | Runs virtual-patient scenarios with algorithms, safety supervision, and reproducible outputs | [Getting Started](GETTING_STARTED.md), [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| Data quality | Imports CGM/pump data, checks realism, and creates MDMP-style certification artifacts | [MDMP Quickstart](MDMP_QUICKSTART.md), [Real-Data Realism Gate](REAL_DATA_REALISM.md) |
| Research AI | Tracks local AI setup, Mistral model migration, and public/request-gated diabetes datasets for research | [AI Assistant](AI_ASSISTANT.md), [Mistral Model Migration](MISTRAL_MODEL_MIGRATION.md), [Local AI Research](LOCAL_AI_RESEARCH.md) |
| Reports | Generates run reports, evidence bundles, posters, and AGP-style research glucose summaries | [Research Evidence Bundle](EVIDENCE_BUNDLE.md), [Command Reference](COMMAND_REFERENCE.md) |
| Edge hardware | Supports Raspberry Pi, Jetson endurance runs, and bench-only Pico/UNO workflows | [Hardware Hub](HARDWARE.md), [Jetson Endurance Mode](JETSON_ENDURANCE.md), [Raspberry Pi Pico Pump Lab](PICO_PUMP_LAB.md) |
| Development | Documents architecture, API symbols, contribution checks, and release maintenance | [Developer Portal](DEVELOPER_PORTAL.md), [API Reference](API_REFERENCE.md), [Maintainer Guide](MAINTAINER_GUIDE.md) |

## Core Workflow

1. Configure a patient, scenario, algorithm, seed, and safety settings.
2. Run a simulation or long study and preserve the output bundle.
3. Validate results with realism, safety, and reproducibility checks.
4. Package evidence through reports, manifests, plots, and citations.

## First Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[full,mdmp,research,edge]"

iints doctor --smoke-run
iints update --dry-run
iints demo eucys --output-dir results/live_demo
```

For source-install testing from the latest GitHub version:

```bash
python -m pip install -U "iints-sdk-python35[full,mdmp,research,edge] @ git+https://github.com/python35/IINTS-SDK.git"
```

## What To Read Next

| If you are... | Read next |
| --- | --- |
| A first-time user | [Quickstart](QUICKSTART.md) then [Getting Started](GETTING_STARTED.md) |
| Preparing for a jury or booth demo | [Booth Demo & Presentation](BOOTH_DEMO.md) then [Command Cheatsheet](CLI_CHEATSHEET.md) |
| Training local AI models | [Diabetes Research Datasets](DIABETES_RESEARCH_DATASETS.md) then [Local AI Research](LOCAL_AI_RESEARCH.md) |
| Reviewing evidence | [Complete Source Library](SOURCE_LIBRARY.md) then [Research Evidence Bundle](EVIDENCE_BUNDLE.md) |
| Building hardware demos | [Hardware Hub](HARDWARE.md) then the board-specific guide |
| Maintaining the SDK | [Developer Portal](DEVELOPER_PORTAL.md) then [Contribute Safely](CONTRIBUTING_SAFELY.md) |

## Project Boundary

IINTS-AF is useful for asking research questions such as whether a simulation is reproducible, whether a glucose trace is plausible, whether a controller behaves safely in a virtual patient, and whether a demo can be explained transparently.

It is not proof that an insulin algorithm is clinically safe. Any real-world medical use would require clinical validation, regulatory review, cybersecurity review, hardware verification, and qualified medical oversight outside the scope of this SDK.
