# Changelog

## Unreleased

- Added data contract runner (`iints data contract-run`) with schema checks, constraint checks, deterministic dataset fingerprinting, and compliance score output.
- Added starter contract generator (`iints data contract-template`) for quick model-ready data gate setup.
- Added built-in unit conversion hooks in contract execution (e.g., mmol/L to mg/dL).
- Added data contract runner test coverage under `tests/data/test_contract_runner.py`.
- Added MDMP grading fields to contract reports (`mdmp_grade`, `mdmp_protocol_version`, `certified_for_medical_research`).
- Added MDMP grade helper utilities and CLI gate option (`--min-mdmp-grade`) for CI enforcement.
- Added MDMP draft documentation in `docs/MDMP.md`.
- Added MDMP certification visualizer command (`iints data mdmp-visualizer`) to generate a single-file interactive HTML audit dashboard.
- Added MDMP visualizer tests under `tests/data/test_mdmp_visualizer.py`.
- Added MDMP Auto-Guardian decorator (`iints.mdmp_gate`) to enforce grade/compliance checks before function execution.
- Added `iints init --template clinical-trial` scaffold with MDMP contract + demo dataset + audit folders.
- Added test coverage for guardians and clinical-trial project scaffold CLI.
- Added `iints study-ready` one-command workflow for researchers (run + validate + sources + summary).
- Enhanced `iints certify-run` to emit `sources_manifest.json` and `SUMMARY.md` by default.
- Added CLI/source helper utilities to export filtered evidence manifests programmatically.
- Added tests for `study-ready` forwarding and certification bundle outputs.

## v0.1.21

- Added evidence source manifest (`iints sources`) backed by peer-reviewed references in `src/iints/presets/evidence_sources.yaml`.
- Added `docs/EVIDENCE_BASE.md` and linked it from the main docs for transparent source-to-feature mapping.
- Added deterministic replay validation (`iints replay-check`) with stable output hashing.
- Added golden benchmark packs and CLI runner (`iints golden-benchmark`) for scenario-range regression checks.
- Added calibration gate profiles and gate-aware forecast evaluation (`iints research evaluate-forecast --gate-profile ...`).
- Expanded formal safety contract schema with explicit max IOB / max bolus / hypo-cutoff limits.
- Added model registry stage promotion flow (`candidate -> validated -> production`) and CLI utilities.
- Added CI ONNX parity smoke checks for predictor export and runtime drift validation.
- Added docs-as-tests CLI smoke checker (`tools/ci/check_docs_examples.py`).
- Added public API surface stability checker with tracked baseline (`tools/ci/check_api_surface.py`).
- Added governance checks for license presence, SBOM structure, dataset licensing metadata, and manifest hashing.

## v0.1.20

- Added profile-driven run validation engine (`iints validate-run`) with bundled threshold profiles (`screening`, `research_default`, `strict_safety`).
- Added `iints validation-profiles` to inspect available validation gates.
- Added `iints doctor` environment health-check with optional smoke simulation.
- Added Dual-Guard predictor wiring in CLI (`run`, `run-full`, `run-parallel`, `presets run`) via `--predictor` checkpoint option.
- Added predictor safety gates in simulator (uncertainty gate + out-of-distribution gate) with audit fields.
- Added formal safety-contract compiler/verifier command (`iints contract-verify`) and packaged default contract spec.
- Added leakage/split auditing (`iints research audit-split`) with sequence-overlap checks.
- Added calibration-first forecast evaluator (`iints research evaluate-forecast`) with band-wise error and alarm-quality metrics.
- Added scenario-bank scorecards (`iints scorecard`) and one-command certification pipeline (`iints certify-run`).
- Added edge parity checks (`iints research parity-check`) for Torch vs ONNX output drift and latency.
- Added predictor metadata capture in run config payloads for reproducibility.
- Added demo scripts under `examples/demos/` including Open-Logic architecture showcase.
- Added InputValidator telemetry in simulator outputs (`input_validator_fail_soft` and summary counters).
- Hardened predictor evaluation pipeline to enforce checkpoint-compatible feature/scaler shapes and robust meal-announcement reconstruction.
- Added OhioT1DM v2 training config (`predictor_ohio_dual_guard_v2.yaml`) with band-weighted loss + early stopping + meal preannounce support.
- Added tests for validation engine + CLI commands and updated packaging to include validation profile YAML.
- Fixed run-manifest safety/reporting bugs in CLI (`audit_summary` existence check and duplicate manifest signing call).
- Removed simulator global NumPy seeding side effect to avoid cross-run randomness coupling.
- Hardened safety core typing and enabled strict mypy gate for supervisor/input-validator modules.
- Added property-based safety tests for non-negative bounded dosing, severe-hypo hard-stop, and formal safety contract invariants.
- Added performance budget tests for supervisor and simulator latency percentiles (p95/p99) and CI gate.
- Added research metrics module with global MAE/RMSE/Bias, glycemic-band metrics, and MC-dropout 95% coverage calibration.
- Added dataset lineage metadata (schema id + dataframe fingerprint + source hash) to training/evaluation outputs and checkpoint config.
- Updated research docs and test coverage for new metrics and lineage.

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
