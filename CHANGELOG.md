# Changelog

## Unreleased

- (no entries)

## v0.1.14

- Added missing type annotations for supervisor state (mypy clean).

## v0.1.13

- Added SafetyConfig and CLI wiring for tunable safety limits.
- Added parallel batch runner and scenario generator tooling.
- Added Nightscout import connector (optional dependency) and Tidepool client skeleton.
- Added tests covering safety config, generator, and Nightscout dependency handling.

## v0.1.12

- Use dedicated publish workflow for PyPI (release workflow now GitHub‑release only).

## v0.1.11

- Hotfix release for PyPI publishing (no functional changes).

## v0.1.10

- Dataset registry and CLI improvements (sample dataset, citations, integrity checks).
- PyPI auto-publish workflow for trusted publishing.

## v0.1.9

- Refreshed notebooks with clearer walkthroughs and baked outputs.
- Added branded logo support in PDF reports and notebooks.
- Fixed CLI output path defaults and rich link formatting.
- Cleaned repository by removing unused legacy folders.
- Added official dataset registry with CLI discovery and fetch helpers.
- Added bundled sample dataset, SHA-256 checks, and dataset citations.

## v0.1.3

- Added safety decision engine with dynamic IOB clamp, trend stop, and 60‑minute cap.
- Added safety decision reasoning to simulator records and audit log.
- Updated simulator to expose safety reasons for explainability tooling.

## v0.1.2

- Added dev workflow tools (Makefile, scripts, .flake8) and improved CI consistency.
- Consolidated documentation into a single authoritative guide.
- Included templates and virtual patient configs in package data.
- Updated examples to use correct `iints.*` imports and fixed indentation issues.
- Added profiling support in simulator and expanded test coverage.
- Cleaned build artifacts and repo-generated files.

## v0.1.0 (Initial Release)

- Initial SDK setup and project structure consolidation.
- Distributable Python packages (.whl, .tar.gz) generated.
- Basic CLI functionality verified.
- Sphinx documentation generated.
- GitHub Actions CI workflow implemented with build, test, linting (Flake8), and type checking (MyPy).
