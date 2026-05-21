# Contribute Safely

This page is for contributors who are allowed to modify IINTS-AF SDK code and need a reliable way to do that without weakening safety, reproducibility, or public documentation.

## Core Rule

Treat the SDK as safety-relevant research software:

- algorithms may propose
- deterministic safety logic decides
- every run should stay reproducible
- documentation, tests, and public claims must move with the code

## Before You Edit

1. Start in the [Developer Portal](DEVELOPER_PORTAL.md).
2. Read the [Visual Architecture](ARCHITECTURE_OVERVIEW.md) if you are not already sure where the change belongs.
3. Open the matching module in the generated [API Reference](API_REFERENCE.md).
4. Find the nearest existing tests before writing new code.
5. Decide whether the change is public behavior, internal behavior, or release-facing behavior.

## Safety Boundaries

Do not blur these lines:

| Boundary | What Must Stay True |
| --- | --- |
| algorithm vs supervisor | A candidate algorithm never becomes final dosing authority by accident. |
| data vs claims | A dataset-quality claim stays tied to MDMP validation, provenance, and cited evidence. |
| runtime vs presentation | Demo output must not replace auditable run bundles. |
| edge Linux side vs bridge firmware | The microcontroller bridge remains thin; durable state and safety logic stay on the SDK side. |
| public API vs internals | Breaking changes are deliberate, documented, and tested. |

## Change Matrix

| Change Type | Minimum Checks |
| --- | --- |
| normal source change | `tools/dev/sdk_check.sh quick` |
| edge / Pi / UNO Q / Jetson | `tools/dev/sdk_check.sh edge` |
| docs / manuals / site navigation | `tools/dev/sdk_check.sh docs` |
| release-facing or broad refactor | `tools/dev/sdk_check.sh full` |
| public release | `IINTS_RELEASE_VERSION=X.Y.Z tools/dev/sdk_check.sh release` |

## Required Sync Work

When you change a public workflow, update all surfaces that users can touch:

- CLI help and command behavior
- tests
- task docs such as `GETTING_STARTED.md` or `COMMAND_REFERENCE.md`
- deeper technical docs when the architecture changed
- generated [API Reference](API_REFERENCE.md) when modules changed
- release notes or manual pages when public claims changed

Regenerate the API page with:

```bash
python3 tools/docs/generate_api_reference.py
```

## High-Risk Areas

Pause and review extra carefully when touching:

- `src/iints/core/safety/`
- `src/iints/core/supervisor.py`
- `src/iints/core/simulator.py`
- `src/iints/data/contracts.py`
- `src/iints/data/realism_validator.py`
- `src/iints/live_patient/`
- `src/iints/jetson/`
- public artifact schemas or manifests

For those areas, prefer:

- smaller commits
- explicit tests before refactoring
- backwards-compatible outputs unless the migration is documented
- a reviewer who understands the subsystem

## Pull Request Checklist

Before merging:

- [ ] The implementation is focused and the reason for the change is clear.
- [ ] Existing safety boundaries remain intact.
- [ ] Relevant tests were added or updated.
- [ ] `flake8`, `mypy`, and the relevant pytest slice pass.
- [ ] Public docs changed if the user-facing workflow changed.
- [ ] The API reference was regenerated if module signatures or public symbols changed.
- [ ] Docs still build if documentation changed.
- [ ] Release notes or manuals changed if public behavior or claims changed.

## What Good Looks Like

A safe contribution leaves the next maintainer with:

- source code that is easier to understand
- tests that explain the intended behavior
- docs that match the actual workflow
- artifacts that remain reproducible
- no hidden weakening of safety logic

## Read Next

- [Developer Portal](DEVELOPER_PORTAL.md)
- [Visual Architecture](ARCHITECTURE_OVERVIEW.md)
- [Maintainer Guide](MAINTAINER_GUIDE.md)
- [Documentation Coverage Reference](PUBLIC_DOCUMENTATION.md)
