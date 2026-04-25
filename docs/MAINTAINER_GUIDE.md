# Maintainer Guide

This guide is for manual SDK maintenance: the checks to run, the places to edit, and the small habits that keep IINTS-AF releasable without having to rediscover the project every time.

## The Short Version

For normal day-to-day work:

```bash
tools/dev/sdk_check.sh quick
```

For edge/Pi/UNO Q changes:

```bash
tools/dev/sdk_check.sh edge
```

For docs/manual changes:

```bash
tools/dev/sdk_check.sh docs
```

Before a release:

```bash
tools/dev/sdk_check.sh full
tools/dev/release_audit.sh 1.5.4
```

Or combine the release check:

```bash
IINTS_RELEASE_VERSION=1.5.4 tools/dev/sdk_check.sh release
```

## Maintenance Rhythm

Use this rhythm when you work manually:

1. Make one focused change.
2. Run `tools/dev/sdk_check.sh quick`.
3. If the change touches edge hardware flows, run `tools/dev/sdk_check.sh edge`.
4. If the change touches docs or manuals, run `tools/dev/sdk_check.sh docs`.
5. Commit only the relevant source/docs/test files.
6. Leave generated runtime folders, caches, and local demo output untracked.

The common local-only folders are:

- `.cache/`
- `.cache_eucys/`
- `.mplt_eucys/`
- `iints_uno_q_demo/`
- `results/`
- `site/`
- `build/`
- `dist/`

## Release Checklist

When preparing a release, update these files together:

- `pyproject.toml`
- `src/iints/__init__.py`
- `docs/releases/vX.Y.Z.md`
- `docs/releases/INDEX.md`
- `mkdocs.yml`
- `docs/UPDATING.md`
- `docs/manuals/IINTS-AF_SDK_Manual.md`
- `docs/manuals/pandoc.yaml`

Then run:

```bash
IINTS_RELEASE_VERSION=X.Y.Z tools/dev/sdk_check.sh release
```

If that passes:

```bash
git add <changed files>
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

## What Each Check Covers

`quick` checks:

- targeted high-risk tests
- `flake8` on source, tests, and tools
- `mypy src/iints/`

`edge` checks:

- long-study config and resume behavior
- edge workspace scaffolding
- Pi/UNO Q CLI paths

`docs` checks:

- `mkdocs build --strict`
- technical manual PDF rendering

`full` checks:

- full test suite
- full lint
- full type check
- docs build
- package build
- manual PDF build

`release` checks:

- everything in `full`
- version consistency across package metadata, docs, release notes, and manual metadata

## High-Value Maintenance Backlog

The SDK is healthy, so the best next work is mostly maintainability:

1. Split `src/iints/cli/cli.py` into smaller command modules.
2. Split `src/iints/live_patient/edge_ops.py` into deploy, offline bundle, study, and remote helpers.
3. Add `iints edge study-status` for long-running Pi studies.
4. Add disk-space and heartbeat checks to long studies.
5. Convert long-study YAML validation into an explicit schema.
6. Make package entry point names machine-friendly while keeping display names in UI output.
7. Add manual PDF rendering to CI.
8. Update GitHub Actions when Node runtime warnings require it.
9. Either finish Tidepool support or label it clearly as experimental.
10. Keep docs task-oriented: research, edge, data certification, and maintainer paths.

## When Something Fails

Start with the smallest relevant check. If a docs change fails the full suite, run `tools/dev/sdk_check.sh docs` first. If an edge change fails, run `tools/dev/sdk_check.sh edge`.

For release-version mistakes, run:

```bash
tools/dev/release_audit.sh X.Y.Z
```

The audit prints the exact file and expected value when a version reference is missing.
