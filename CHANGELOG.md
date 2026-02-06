# Changelog

## Unreleased

- Added Pydantic-based scenario and patient config validation with scenario versioning.
- Added clinic-safe story presets (pizza paradox, midnight crash) and baseline comparison tables.
- Added deterministic runs via CLI seed support and a one-line `run_simulation` API.

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
