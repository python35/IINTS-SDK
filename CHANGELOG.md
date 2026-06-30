# Changelog

## Unreleased

## v1.5.21

Release date: 2026-06-30

### Fixed

- fixed a Matplotlib/scienceplots mathtext failure that could abort desktop baseline report generation during `fig.tight_layout()`
- disabled mathtext tick formatting in the shared IINTS plotting style to keep report labels plain and robust across macOS/Python/Matplotlib combinations
- added a fallback layout path so the clinical validation PNG is still saved even if Matplotlib tight layout fails

### Verified

- PDF regression tests: `7 passed`
- desktop baseline workflow smoke generated both `results.csv` and `clinical_report.pdf`
- full test suite: `652 passed, 2 skipped`

## v1.5.20

Release date: 2026-06-30

### Fixed

- published the complete post-rebase desktop/app release from current `main`
- ensures the terminal update UI, integrated app terminal, Cocoa startup fix, desktop import merge, AlphaFold/genomics desktop updates, and uninstall helper are released together

### Why this release exists

- `v1.5.19` was created before the final rebase landed on `main`; `v1.5.20` is the clean follow-up release from the final merged commit line

## v1.5.19

Release date: 2026-06-30

### Added

- added a cross-platform desktop update terminal flow for the PySide/Qt app
- added a Methods-tab `Developer Settings / Integrated App Terminal` panel that streams command stdout/stderr inside the app
- added a terminal walkthrough documenting macOS, Windows, and Linux update behavior

### Fixed

- fixed the macOS Cocoa desktop crash caused by PyObjC treating helper methods as Objective-C selectors
- fixed the packaged standard diabetes MDMP contract path to satisfy mypy on modern `Traversable` resources

### Verified

- desktop tests: `28 passed`
- Cocoa smoke check: `Cocoa desktop smoke OK`
- targeted mypy checks: no issues found
- full test suite: `651 passed, 2 skipped`
- docs build: `mkdocs build --strict`

## v1.5.15

Release date: 2026-06-13

### Fixed

- fixed the `AdvancedMetabolicModel.update()` signature so it remains compatible with `BergmanPatientModel.update()` under mypy
- kept `fat_intake`, `protein_intake`, `dt_minutes`, `delivered_glucagon`, and `current_time_minutes` support through keyword arguments

### Verified

- `mypy src/iints/`: no issues found in 182 source files
- advanced metabolic/scientific fidelity tests: `8 passed`


## v1.5.14

Release date: 2026-06-12

### Fixed

- fixed `AdvancedMetabolicModel.update()` compatibility with simulator and scratch-script keyword names
- exposed the advanced 18-state model through `PatientFactory`
- routed simulator illness events to illness-aware patient models
- tuned the realism validator so large meals are assessed causally while small snacks do not create false failures
- corrected 16-state/18-state documentation drift and softened over-strong clinical claims

### Verified

- full suite: `549 passed, 4 skipped`
- package build: `iints_sdk_python35-1.5.14`


## v1.5.6

Release date: 2026-05-18

### Added

- added self-contained quickstart safety improvements, run preview diagnostics, and clearer early-termination summaries
- added a more polished live-demo presenter flow for Zoom calls, clinicians, engineers, and expo booths
- added detailed physiology reference materials and EUCYS-ready physiology handouts
- added Jetson wall-clock research mode so a requested `1d` study can genuinely occupy 24 real hours
- added local AI research tooling for controller datasets, auditable imitation policies, PyTorch neural controllers, and held-out closed-loop controller evaluation
- added Jetson post-run research finalization with automatic model training and evaluation reports

### Fixed

- released the Jetson `--wall-clock` and `--finalize-research` public CLI flags that were added after `v1.5.5`
- closed the drift between current documentation and the installable PyPI wheel

### Why it matters

This patch release makes the newly documented research workflow actually installable from PyPI: the package, CLI help, Jetson docs, and local-AI workflow now describe the same public feature set.

## v1.5.5

Release date: 2026-05-17

### Added

- added a full Tidepool import path that exports standard CSV and scenario bundles
- added stricter predictor validation with external datasets, subgroup reports, uncertainty reliability, hypo-detection sensitivity, and drift checks
- added `iints onboard` as one canonical first-run path from environment check to study setup
- added richer Jetson endurance telemetry, configurable checkpoints, partial output persistence, and safer resume support
- added EUCYS-ready PDF bundles, a live demo flow, developer portal pages, generated API reference docs, and local plugin foundations
- added physiological realism validation, empirical reference profiles, realism dashboards, and empirical simulator calibration tooling

### Improved

- made simulator traces substantially more physiologically realistic using empirical residual profiles and calibrated day presets
- rebuilt and simplified the docs site structure, homepage, footer, and role-based navigation
- made Raspberry Pi and UNO Q setup easier to explain and operate for demos and jury conversations
- hardened CI, governance, versioned outputs, and runtime database cleanup

### Why it matters

This release joins the work since `v1.5.4` into one stronger SDK story: better real-data ingestion, more believable physiology, stricter model validation, clearer onboarding, and more credible long-running edge evidence.

## v1.5.2

Release date: 2026-04-06

### Added

- introduced the SBC-first edge workflow:
  - `iints edge setup`
  - `iints edge status`
  - `iints edge bundle`
  - `iints edge update`
  - `iints edge hardware-bridge`
- added a kiosk-oriented live patient presentation layer:
  - `iints patient kiosk`
  - richer dashboard cards for certification, realism review, and active scenario
  - one-click scenario shortcuts for booth and classroom demos
- added generated edge deployment scaffolding:
  - `run_edge_patient.sh`
  - `launch_kiosk.sh`
  - `update_edge_runtime.sh`
  - exported `systemd` unit files and install notes
- added UNO Q bridge export scaffolding for physical LEDs / buzzer feedback
- added public SBC support documentation with a support matrix and architecture diagram

### Improved

- split installs cleanly into workstation and SBC profiles:
  - `iints-sdk-python35[full,mdmp]`
  - `iints-sdk-python35[edge,mdmp]`
- made analysis/report imports safer so the edge profile can run without the full reporting stack installed
- expanded the live patient workspace summary to show certification, realism review, workspace size, kiosk URL, and bundle status
- improved Raspberry Pi and UNO Q docs so setup, update, and service installation are clearer and more reproducible

### Why it matters

This is the SBC release: the SDK now has a clean story for Raspberry Pi and other Linux-capable edge boards. You can scaffold an edge runtime, auto-start it as a service, present it in kiosk mode, export the runtime back to a laptop, and keep the public docs aligned with that deployment path.

## v1.5.1

Release date: 2026-03-30

### Added

- new hypothesis-driven scientific workflow layer:
  - `iints study-protocol`
  - `docs/SCIENTIFIC_WORKFLOW.md`
  - protocol bundles with `STUDY_PROTOCOL.md`, `study_design.json`, and `study_matrix.csv`
- new controlled corruption workflow for ablation studies:
  - `iints data corrupt-for-study`
  - corruption manifests for timestamp shifts, missing blocks, duplicated rows, glucose spikes, dropped meal annotations, and unit-scale errors
- new EUCYS-specific study preset support:
  - `iints scenarios export-study-pack --preset eucys`
  - `iints study-protocol --preset eucys`
  - `iints run-eucys-study`

### Improved

- `iints analyze` now reports richer scientific outputs:
  - descriptive statistics with standard deviation and 95% confidence intervals
  - failure analysis for severe hypo, early termination, and worst-run surfacing
  - optional external plausibility comparison against imported CareLink metrics
- `iints compare-study` now includes effect estimates and confidence intervals for key metrics
- study posters now include failure-analysis and external-validation cues
- docs and manuals now explain the fair-ready scientific workflow more clearly

### Why it matters

This patch shifts the SDK from “good demo tooling” toward a real experimental platform: it is now much easier to define a hypothesis, run a fixed matrix, compare clean vs corrupted evidence, and defend the results in a scientific setting.

## v1.5.0

Release date: 2026-03-26

### Added

- new `iints ai review` command for realism-oriented run critique using the local Ministral model
- automatic `review_payload.json` generation in AI-prepared run bundles and CareLink workspaces
- automatic `realism_review.md` output when reviewing a prepared run directory
- new public release checklist in `docs/PUBLIC_RELEASE_CHECKLIST.md`

### Improved

- booth demo scripts and public-facing guides now show how to use AI both for explanation and for realism review
- the AI docs, manual, README, and CareLink guidance now document the full explain/report/review flow
- prerelease validation confirmed lint, type checks, tests, docs, manuals, package build, and security scans are in a releasable state

### Why it matters

This release makes the public story stronger: the SDK can now not only simulate and explain results, but also critique whether they look realistic and give concrete feedback points before a paper, demo, or external release.

## v1.4.0

Release date: 2026-03-24

### Added

- bundled the MDMP protocol implementation directly into the SDK under `src/mdmp_core`, `src/mdmp_ai`, `src/mdmp_flavors`, and `src/mdmp_integrations`
- bundled conformance vectors and the MDMP public key into the SDK distribution so offline verification and conformance checks survive without a separate MDMP checkout
- added Apache 2.0 distribution metadata plus `NOTICE` and preserved legacy license notices for the combined SDK distribution
- added a bundled MDMP integration test to verify AI-ready artifact preparation and MDMP guard validation from the SDK alone

### Improved

- AI MDMP preparation and guard flows now describe the bundled MDMP runtime instead of pointing users at an external package install
- MDMP CI and sync checks now validate the bundled implementation directly, removing the old dual-repo dependency for normal SDK development
- T1D defaults, clinical boundary handling, data plausibility checks, and model-loading paths were hardened as part of the same release train
- install, AI, evidence, and manual docs now explain Ollama setup, medical evidence sources, and the new bundled-MDMP architecture more clearly

### Why it matters

This release makes the SDK operationally self-contained: the research, AI, and MDMP trust flows now ship together in one package, which means the separate MDMP repository is no longer required for normal SDK installs or demos.

## v1.3.2

Release date: 2026-03-24

### Added

- new `iints demo-export` command for machines that have the SDK installed but do not have the repository checkout
- bundled exportable live demo script template under `src/iints/templates/demos/live_stage_demo.py`
- new install/path guide: `docs/INSTALLATION.md`

### Improved

- the live booth script now visibly demonstrates `run_full(...)`, `generate_results_poster(...)`, and `prepare_ai_ready_artifacts(...)`
- booth wrappers now resolve the repository root automatically, making them safer to launch from another working directory
- installation, updating, getting started, and booth demo docs now explain much more clearly which commands run from which folder

### Why it matters

This patch closes the last booth-demo gap on secondary machines: users can now install the SDK, export the demo code, show the code, run it, and explain the results without needing a full repository clone.

## v1.3.1

Release date: 2026-03-23

### Added

- new live-showcase script: `examples/demos/07_live_stage_demo.py`
- new shell runner: `scripts/run_live_stage_demo.sh`

### Improved

- booth demo guidance now follows the clearer show-code -> run -> results flow
- live demo notes explicitly show how to swap the patient profile on stage
- update documentation now covers the common editable-install mistake of running `pip install -e ".[mdmp]"` outside the SDK repo root

### Why it matters

This patch makes the SDK easier to install on a second machine and easier to demonstrate live at a fair or jury table.

## v1.3.0

Release date: 2026-03-23

- Added a fair-ready booth demo workflow:
  - new `iints demo-booth` CLI command
  - new `build_booth_demo(...)` Python API
  - new `examples/demos/06_booth_demo.py` showcase script
  - new `scripts/run_booth_demo.sh` helper for fast live demos
- Added complete demo outputs for public presentations:
  - combined poster PNG
  - markdown jury talk track
  - plain-text live demo script
  - demo summary JSON and command cheat sheet
- Added a dedicated SDK update guide:
  - new `docs/UPDATING.md`
  - docs now explain how to upgrade in a virtual environment
  - docs now cover conflict repair when legacy `iints` shadows `iints-sdk-python35`
- Updated installation troubleshooting and doctor guidance so multi-machine upgrades are easier to verify.

## v1.2.0

Release date: 2026-03-21

- Added poster-ready results export for demos and juries:
  - new `iints poster` CLI command
  - new `generate_results_poster(...)` Python API
  - poster PNG + JSON summary generated directly from one to three run bundles
- Added Medtronic CareLink / MiniMed import support:
  - new `iints import-carelink` CLI command
  - `import-data --data-format carelink` now works for CareLink exports
  - import pipeline converts event logs into the standard IINTS glucose/carb/insulin timeline plus a CareLink summary JSON
- Expanded the personal CareLink workflow:
  - new `iints carelink-workbench` CLI command
  - new `build_carelink_workbench(...)` Python API
  - new `load_carelink_event_log(...)` and `import_carelink_timeline(...)` APIs for experiments with real MiniMed data
  - workbench now generates `carelink_dashboard.png`, `carelink_poster.png`, `carelink_dashboard.html`, `carelink_metrics.json`, `carelink_timeline.csv`, and AI-ready payloads for local Mistral explanations
- Updated the AI layer docs and prompts so local Mistral explanations also cover imported personal glucose datasets, not only simulation runs

## v1.1.3

- Hardened the local Ollama path for Mistral-family models:
  - `iints ai local-check` now runs a tiny generation smoke-test by default instead of only checking reachability and installed tags
  - local generation now retries once on transient disconnects before failing
  - disconnect failures now explain that the daemon likely restarted or the model was too heavy for available memory
- Improved the troubleshooting guidance for local AI inference:
  - docs now recommend stepping down to `ministral-3:3b` when `ministral-3:8b` closes the connection during generation
  - the AI guide, quickstart docs, technical docs, and manual all reflect the new smoke-test behavior

## v1.1.2

- Added `iints ai prepare <run_dir>` to generate AI-ready payloads from an existing run.
- AI commands can now point directly at a run directory and auto-resolve prepared payloads, the local development MDMP certificate, and the companion public key.
- `run`, `run-full`, and `presets run` now attempt to export AI-ready artifacts into the run bundle automatically.
- Added `iints-sdk-doctor` to diagnose legacy-package conflicts when `iints ai` is missing.
- Fixed the exported `iints.__version__` value to follow the installed SDK distribution metadata.
- Documented the repair flow for environments where an older `iints` package shadows `iints-sdk-python35`.

## v1.1.1

- Fixed the local Mistral/Ollama default model selection for the AI assistant:
  - default local model now targets the open-weight `Ministral 3` line (`ministral-3:8b`)
  - legacy `Ministral 8B` Ollama tags remain accepted as backward-compatible fallbacks
  - local health checks now surface Ollama runtime version compatibility for the open `Ministral 3` model family
- Added local model selection guidance for users with different hardware profiles:
  - new `iints ai models` command lists curated local Mistral-family options
  - docs and manuals now include PC spec recommendations for `3B`, `8B`, and `14B` variants

## v1.1.0

- Added the local research AI assistant release for Ministral via Ollama:
  - `iints ai explain`
  - `iints ai trends`
  - `iints ai anomalies`
  - `iints ai report`
  - `iints ai local-check`
- Hardened local Ministral readiness checks:
  - verifies Ollama reachability before generation
  - verifies that a compatible local Ministral tag is installed
  - resolves friendly aliases such as `ministral` to the installed Ollama tag
  - records the resolved local model identity in the AI response path
- Improved edge-device stability for local inference:
  - increased default AI timeout to `120` seconds
  - exposed `--timeout-seconds` on AI CLI commands
  - clipped oversized JSON payloads before prompt construction to reduce local inference failures
- Expanded AI test coverage for:
  - alias resolution
  - missing-model failure behavior
  - CLI local-check success/failure paths
  - prompt truncation
- Added a dedicated AI guide and updated the manuals so the release documents:
  - how the AI layer works
  - how MDMP gating works
  - how to debug the local Ollama/Ministral flow
- Fixed MDMP package floor to match published versions:
  - SDK optional extra now uses `mdmp-protocol>=0.3.0`
  - MDMP sync PyPI fallback now uses `mdmp-protocol>=0.3.0`
- Fixed MDMP sync workflow to support two sources:
  - private-repo checkout when `MDMP_REPO_TOKEN` is available
  - automatic PyPI fallback (`mdmp-protocol>=0.3.0`) when checkout is not available
- MDMP roadmap note (post `v1.1.0`): deeper protocol expansion is planned; some new features and bug-fix turnarounds may take longer while the MDMP surface is stabilized.

## v1.0.0

- First public production release of IINTS-AF SDK.
- Promoted package maturity classifier to `Production/Stable`.
- Standardized documentation to require active virtual environments across quickstarts/manual pages.
- Added branded CLI help header (ASCII IINTS logo in `iints --help`).
- Hardened MDMP integration for release stability:
  - optional dependency now uses published `mdmp-protocol>=0.2.0`
  - MDMP sync gate supports token checkout and public-package fallback
  - added automatic dependency update flow via Dependabot for MDMP package changes.
- Added release notes page for `v1.0.0` and updated docs navigation.

## v0.1.22

- Added dedicated MDMP namespace:
  - Python imports: `iints.mdmp`
  - CLI commands: `iints mdmp ...` (`template`, `validate`, `synthetic-mirror`, `visualizer`)
- Added MDMP runtime Auto-Guardians (`mdmp_gate`) for in-memory compliance gating before pipeline execution.
- Added MDMP Synthetic Mirror generation (`generate_synthetic_mirror`, `iints data synthetic-mirror`) with contract-aware validation output.
- Added MDMP certification visualizer (`iints data mdmp-visualizer`) for single-file interactive HTML audit dashboards.
- Added MDMP grading metadata and gating support (`mdmp_grade`, `mdmp_protocol_version`, `certified_for_medical_research`, `--min-mdmp-grade`).
- Added clinical-trial scaffold template via `iints init --template clinical-trial`.
- Added `iints study-ready` one-command bundle flow and enhanced `iints certify-run` outputs (`sources_manifest.json`, `SUMMARY.md`).
- Added MkDocs documentation site (`mkdocs.yml`, `docs/index.md`) and GitHub Pages deployment workflow.
- Fixed report UTC timestamp generation to timezone-aware datetime.
- Removed non-ASCII cockpit alert glyph to avoid font rendering warnings.

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
