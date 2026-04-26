# User Guide Map

Use this page when you are outside the project and want the documentation to behave like one connected manual.

## First Three Clicks

If you are new, follow this order:

1. [Quickstart](QUICKSTART.md)
2. [Getting Started](GETTING_STARTED.md)
3. [Troubleshooting](TROUBLESHOOTING.md)

If you prefer the CLI to choose the route for you:

```bash
iints start
```

## Choose Your Path

| Goal | Read first | Then read | Main command |
|---|---|---|---|
| See the SDK work quickly | [Quickstart](QUICKSTART.md) | [Getting Started](GETTING_STARTED.md) | `iints demo` |
| Build your own starter project | [Quickstart](QUICKSTART.md) | [Command Reference](COMMAND_REFERENCE.md) | `iints start --goal project --run` |
| Run scientific benchmarks | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) | [Study Analysis](STUDY_ANALYSIS.md) | `iints run-study` |
| Certify data quality | [MDMP Quickstart](MDMP_QUICKSTART.md) | [MDMP Guide](MDMP_FULL_GUIDE.md) | `iints data certify` |
| Add your own algorithms or models | [Command Reference](COMMAND_REFERENCE.md) | [Technical Reference](TECHNICAL_README.md) | `iints plugin install` |
| Prepare a Raspberry Pi | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) | [Remote Deploy & Pi Connect](EDGE_REMOTE_DEPLOY.md) | `iints edge setup` |
| Prepare an event booth | [Maker Faire Pi Mode](MAKERFAIRE_PI.md) | [Maker Faire Pi Checklist](MAKERFAIRE_PI_CHECKLIST.md) | `iints makerfaire up` |
| Understand every command | [Command Reference](COMMAND_REFERENCE.md) | [Technical Reference](TECHNICAL_README.md) | `iints --help` |
| Maintain the SDK | [Maintainer Guide](MAINTAINER_GUIDE.md) | [Release Checklist](PUBLIC_RELEASE_CHECKLIST.md) | `tools/dev/sdk_check.sh quick` |

## How The Docs Fit Together

The public docs are split by job:

- Start pages explain the shortest route to a successful first result.
- User guides explain one workflow at a time.
- Edge guides explain Raspberry Pi, UNO Q, remote deploy, offline install, and long-running studies.
- Extension commands explain local algorithm and patient-model plugins without editing SDK source.
- Data guides explain certification, data trust, and MDMP output.
- Reference pages explain all commands and deeper architecture.
- Maintainer pages explain checks, release steps, and repo upkeep.

## Recommended External User Route

For a first-time external evaluator:

```text
Home
  -> User Guide Map
  -> Quickstart
  -> Getting Started
  -> Command Reference
  -> Scientific Workflow or Raspberry Pi Digital Patient
```

For a researcher:

```text
Scientific Workflow
  -> Study Analysis
  -> MDMP Quickstart
  -> Evidence Base
  -> Full Technical Manual
```

For an edge demo:

```text
Raspberry Pi Digital Patient
  -> Remote Deploy & Pi Connect
  -> Maker Faire Pi Mode
  -> Maker Faire Pi Checklist
  -> Arduino UNO Q Setup
```

## If You Get Lost

Run:

```bash
iints start
iints doctor --full --suggest
```

Then return to:

- [Troubleshooting](TROUBLESHOOTING.md)
- [Command Reference](COMMAND_REFERENCE.md)
- [Documentation By Role](DOCUMENTATION_INDEX.md)
