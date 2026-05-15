# Choose Your Path

Use this page when you know what you want to achieve, but not yet which guide deserves your time.

## If You Only Click Three Things

1. [Quickstart](QUICKSTART.md) - prove the SDK works on your machine.
2. [Getting Started](GETTING_STARTED.md) - make your first complete run bundle.
3. [Workflow Hub](WORKFLOWS.md) - continue into research, data, AI, or demo work.

## Choose By Goal

| I want to... | Start here | Continue with | Main command |
| --- | --- | --- | --- |
| see the SDK work quickly | [Quickstart](QUICKSTART.md) | [Getting Started](GETTING_STARTED.md) | `iints demo` |
| understand the project without jargon | [Plain-Language Overview](PLAIN_LANGUAGE_GUIDE.md) | [Getting Started](GETTING_STARTED.md) | `iints start` |
| run a scientific benchmark | [Workflow Hub](WORKFLOWS.md) | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) | `iints run-study` |
| certify data quality | [MDMP Quickstart](MDMP_QUICKSTART.md) | [Data Certification Full Guide](MDMP_FULL_GUIDE.md) | `iints data certify` |
| analyze a finished study | [Study Analysis](STUDY_ANALYSIS.md) | [Evidence Base](EVIDENCE_BASE.md) | `iints study analyze` |
| prepare a live demo | [Booth Demo Guide](BOOTH_DEMO.md) | [Workflow Hub](WORKFLOWS.md) | `iints demo-live` |
| deploy hardware | [Hardware Hub](HARDWARE.md) | board-specific guide | `iints edge quickstart` |
| change SDK code | [Developer Portal](DEVELOPER_PORTAL.md) | [Contribute Safely](CONTRIBUTING_SAFELY.md) | `tools/dev/sdk_check.sh quick` |
| look up a command fast | [Command Reference](COMMAND_REFERENCE.md) | [Reference Hub](REFERENCE_OVERVIEW.md) | `iints --help` |

## Choose By Role

### First-Time User

```text
Quickstart
  -> Getting Started
  -> Troubleshooting if needed
```

### Researcher

```text
Workflow Hub
  -> Scientific Workflow
  -> Study Analysis
  -> Evidence Base
```

### Data Reviewer

```text
MDMP Quickstart
  -> Data Certification Full Guide
  -> Evidence Base
```

### Hardware Builder

```text
Hardware Hub
  -> Raspberry Pi / UNO Q / Jetson guide
  -> Remote Deploy if the board is off your desk
```

### SDK Contributor

```text
Developer Portal
  -> Visual Architecture
  -> API Reference
  -> Contribute Safely
```

## If You Get Lost

Run:

```bash
iints start
iints doctor --full --suggest
```

Then return to:

- [Troubleshooting](TROUBLESHOOTING.md)
- [Reference Hub](REFERENCE_OVERVIEW.md)
- [Documentation By Role](DOCUMENTATION_INDEX.md)
