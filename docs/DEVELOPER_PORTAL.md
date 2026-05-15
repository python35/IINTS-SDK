# Developer Portal

This is the technical entry point for people who are allowed to change the IINTS-AF SDK itself.

If you need to understand the codebase, find the right subsystem, change behavior safely, run the correct checks, or prepare a release, start here.

## What This Portal Covers

- architecture and module boundaries
- exact CLI and integration behavior
- simulation, safety, data, AI, and edge-device implementation areas
- contributor workflow, checks, and release discipline
- the documentation pages that explain each subsystem in depth

## Read These First

| Need | Start Here |
| --- | --- |
| See the system visually before editing | [Visual Architecture](ARCHITECTURE_OVERVIEW.md) |
| Find every Python module and its public symbols | [API Reference](API_REFERENCE.md) |
| Understand the whole SDK architecture | [Architecture & Module Guide](COMPREHENSIVE_GUIDE.md) |
| Check exact CLI behavior and advanced commands | [CLI & Advanced Reference](TECHNICAL_README.md) |
| Contribute without weakening safety or docs | [Contribute Safely](CONTRIBUTING_SAFELY.md) |
| Maintain the repository safely | [Maintainer Guide](MAINTAINER_GUIDE.md) |
| See every public documentation surface | [Documentation Coverage Reference](PUBLIC_DOCUMENTATION.md) |
| Prepare a maintained release | [Maintainer Release Checklist](PUBLIC_RELEASE_CHECKLIST.md) |
| Need the long-form printable manual | [Full Technical Manual](manuals/IINTS-AF_SDK_Manual.md) |

## Codebase Map

| If You Are Changing | Main Code Area | Read Before Editing |
| --- | --- | --- |
| simulation loop, patient state, safety gates | `src/iints/core/` | [Architecture & Module Guide](COMPREHENSIVE_GUIDE.md) |
| CLI commands and public workflows | `src/iints/cli/` | [CLI & Advanced Reference](TECHNICAL_README.md) |
| CGM imports, registries, quality checks, realism | `src/iints/data/` | [MDMP Guide](MDMP_FULL_GUIDE.md), [Evidence Base](EVIDENCE_BASE.md) |
| reporting, posters, study analysis | `src/iints/analysis/` | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md), [Study Analysis](STUDY_ANALYSIS.md) |
| local AI preparation and explanation gates | `src/iints/ai/` | [AI Assistant](AI_ASSISTANT.md) |
| Pi, UNO Q, Jetson, long studies | `src/iints/live_patient/`, `src/iints/jetson/` | [Edge Hardware & SBC Matrix](EDGE_HARDWARE.md), [Jetson Endurance Mode](JETSON_ENDURANCE.md) |
| MDMP packaging split between repos | bundled MDMP + CI tools | [MDMP Packaging Workflow](DUAL_REPO_WORKFLOW.md) |
| docs, manuals, public release surfaces | `docs/`, `tools/research/` | [Documentation Coverage Reference](PUBLIC_DOCUMENTATION.md), [Manual Overview](OFFICIAL_MANUAL.md) |

## Before You Change Code

1. Read the matching subsystem guide from the tables above.
2. Locate the tests that already describe the behavior you are touching.
3. Keep the public CLI, generated artifacts, and docs aligned when you change a workflow.
4. Prefer the smallest relevant check first, then broaden the validation before merge.

Useful commands:

```bash
tools/dev/sdk_check.sh quick
tools/dev/sdk_check.sh edge
tools/dev/sdk_check.sh docs
tools/dev/sdk_check.sh full
```

The generated [API Reference](API_REFERENCE.md) is built from `src/iints/**/*.py`. Regenerate it after public module changes with:

```bash
python3 tools/docs/generate_api_reference.py
```

## What Must Stay True

- algorithms remain separated from deterministic safety supervision
- simulation runs remain reproducible through recorded configuration, seeds, and manifests
- data-quality claims stay tied to MDMP validation and cited evidence
- edge flows keep a clear Linux-side runtime and hardware-bridge boundary
- public docs, generated manuals, CLI help, tests, and release notes do not drift apart

## Change-Type Shortcuts

### Add Or Modify A CLI Command

- Start with [CLI & Advanced Reference](TECHNICAL_README.md)
- Update command tests under `tests/`
- Update [Command Reference](COMMAND_REFERENCE.md) when the public workflow changes

### Change Simulator Or Safety Behavior

- Start with [Architecture & Module Guide](COMPREHENSIVE_GUIDE.md)
- Inspect tests under `tests/core/`, `tests/validation/`, and relevant regression coverage
- Re-check artifact compatibility before changing outputs

### Change Data Or Realism Logic

- Start with [MDMP Guide](MDMP_FULL_GUIDE.md) and [Evidence Base](EVIDENCE_BASE.md)
- Keep registry metadata, source citations, validators, and docs synchronized
- Preserve deterministic certification behavior

### Change Edge Hardware Support

- Start with [Edge Hardware & SBC Matrix](EDGE_HARDWARE.md)
- Read the board-specific guide before changing user-facing flows
- Run `tools/dev/sdk_check.sh edge`

### Prepare A Release

- Start with [Maintainer Guide](MAINTAINER_GUIDE.md)
- Then follow [Maintainer Release Checklist](PUBLIC_RELEASE_CHECKLIST.md)
- Confirm docs, package metadata, notebooks, release notes, and manuals are aligned

## Definition Of Done

Before a code change is considered complete:

- the implementation is updated
- relevant tests are added or adjusted
- user-facing docs are updated if the workflow changed
- `flake8`, `mypy`, and the relevant pytest slice pass
- generated docs still build cleanly
- release-facing files are updated when public behavior changed

If you are unsure where to begin, use [Documentation By Role](DOCUMENTATION_INDEX.md) or [Choose Your Path](USER_GUIDE_MAP.md) to route yourself to the right level of detail.
