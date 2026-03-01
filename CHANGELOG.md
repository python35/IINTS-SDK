# Changelog

## Unreleased

- (empty)

## v0.1.19

- Added meal-response filtering for OhioT1DM/AZT1D prep to drop noisy meal labels.
- Recomputed subject segments and IOB/COB per segment for more stable training windows.
- Added band-weighted loss options to emphasize hypo/hyper accuracy.
- Enabled early stopping in default research configs for better generalization.

## v0.1.18

- Multimodal predictor training pipeline with warm-start fine‑tuning and early stopping.
- Added HUPA‑UCM and OhioT1DM preparation scripts + configs.
- Added SafetyWeighted and Quantile loss modules with robust torch guards.
- Added chaos‑test algorithms, scenarios, and safety‑event callback support.
- Demo showcase supports predictor integration + optional ONNX export.
- Added ONNX export CLI and research extras for edge deployment.
- Updated docs with end‑to‑end training + export steps.

## v0.1.17

- Added Monte Carlo population evaluation with parallel runner, safety index aggregation, and PDF reporting.
- Added Bergman minimal patient model (UVA/Padova-lite) for physiologic simulations.
- Added population CLI entrypoint (`iints evaluate`) with seeded runs and confidence intervals.
- Expanded research pipeline: subject-level splits, OpenAPS-style IOB model, baseline predictors, MC dropout, and quantile loss.
- Added research-focused tests for population + Bergman model.

## v0.1.16

- Added dynamic ratio support (ISF/ICR/DIA/Basal) and scenario ratio-change events.
- Added glucose trend + 30-min prediction signal in simulator and algorithm input.
- Safety supervisor now blocks predicted hypoglycemia and caps excessive basal rates.
- Added a maintainer note inviting bug reports and contributions.

## v0.1.15

- Hardened dataset registry resource loading for Python 3.14+.
- Refreshed notebooks for 3.9 compatibility and clearer data registry flow.
- Preset CLI now defaults output paths to the working directory and fixes file links.

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
