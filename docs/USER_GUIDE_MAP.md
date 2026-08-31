# Choose By Goal

Use this page when you know what you want to do but not which guide to open.

## New To IINTS-AF

Follow the [Learning Path](LEARNING_PATH.md). The shortest useful sequence is:

1. [Installation](INSTALLATION.md)
2. [System Requirements](SYSTEM_REQUIREMENTS.md)
3. [First Run](QUICKSTART.md)
4. [Core Concepts](CORE_CONCEPTS.md)
5. [Understand A Run](RUN_OUTPUTS.md)
6. [Complete First Workflow](GETTING_STARTED.md)

## Choose A Task

| I want to... | Start here | Main tool |
| --- | --- | --- |
| explore SDK interactively | [Interactive Hub](CLI_CHEATSHEET.md) | `iints menu` |
| check whether a machine is suitable | [System Requirements](SYSTEM_REQUIREMENTS.md) | `iints doctor --full --suggest` |
| install on Omarchy Linux | [Omarchy Linux](OMARCHY_INSTALL.md) | `install_omarchy.sh` |
| view full command cheat-sheet | [Command Map](COMMAND_REFERENCE.md) | `iints map` |
| run a small simulation | [First Run](QUICKSTART.md) | `iints demo quick` |
| create a controlled project | [Complete First Workflow](GETTING_STARTED.md) | `iints quickstart` and `iints run` |
| use the graphical application | [Desktop App](DESKTOP_APP.md) | `iints-desktop` or a beta installer |
| compare algorithms | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) | `iints run-study` |
| inspect completed results | [Understand A Run](RUN_OUTPUTS.md) | CSV, metadata, manifest, and report |
| encrypt & certify a dataset | [Certification Quickstart](MDMP_QUICKSTART.md) | `iints mdmp encrypt-data`, `iints data certify` |
| explain a run with local AI | [AI Assistant](AI_ASSISTANT.md) | `iints ai report` |
| train a glucose predictor | [Glucose Forecast Model](GLUCOSE_MODEL.md) | `iints research glucose-model` |
| understand the physiology | [Physiology Reference](PHYSIOLOGY_REFERENCE.md) | formula and source references |
| study stem-cell-derived islet evidence | [Regenerative Islet Research](REGENERATIVE_ISLET_RESEARCH.md) | validated protein panels and the graft research API |
| use edge hardware | [Hardware Hub](HARDWARE.md) | board-specific commands |
| present a safe live demo | [Booth Demo](BOOTH_DEMO.md) | `iints demo doctor`, `eucys`, or `booth` |
| modify the SDK | [Developer Portal](DEVELOPER_PORTAL.md) | `tools/dev/sdk_check.sh quick` |
| find a command | [Command Cheatsheet](CLI_CHEATSHEET.md) | `iints map` or `iints --help` |

## Choose By Responsibility

**Research user:** [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) → [Study Analysis](STUDY_ANALYSIS.md) → [Evidence Bundle](EVIDENCE_BUNDLE.md)

**Data reviewer:** [Certification Quickstart](MDMP_QUICKSTART.md) → [Certification Guide](MDMP_FULL_GUIDE.md) → [MDMP Specification](MDMP.md)

**AI researcher:** [Glucose Forecast Model](GLUCOSE_MODEL.md) → [Interpreting Models](comparison_interpretation.md) → [AI Safety Gates](LOCAL_AI_SAFETY_GATES.md)

**Hardware builder:** [Hardware Hub](HARDWARE.md) → device guide → [Edge Hardware Matrix](EDGE_HARDWARE.md)

**Contributor:** [Developer Portal](DEVELOPER_PORTAL.md) → [Visual Architecture](ARCHITECTURE_OVERVIEW.md) → [Contribute Safely](CONTRIBUTING_SAFELY.md)

## If You Are Lost

```bash
iints menu
iints map
iints guide
iints doctor --full --suggest
```

Then return to [Learning Path](LEARNING_PATH.md) or [Troubleshooting](TROUBLESHOOTING.md).
