# IINTS Neural Forecasting v1 — Phase 2 Verification Gate Report
> **Generated UTC:** `2026-11-27T19:05:48.518394+00:00`  
> **Scope:** Research-only software & methodological verification (`clinical_use=False`). Zero insulin-delivery or hardware modules modified.

## 1. Exact Commands Executed
```powershell
$env:PYTHONPATH = 'src'
.\.venv-stress\Scripts\python.exe tools/forecasting/run_phase2_verification_gate.py
```

## 2. Test Suite Execution Results
- **Pytest Exit Code:** `0`
- **Passed:** `62`
- **Failed:** `0`
- **Skipped:** `0`
- **Errors:** `0`
- **Suites Verified:** `test_phase1_phase2_baselines.py`, `test_neural_robustness_and_jetson.py`, `test_sdk2_and_research_lab.py`, `test_mathematical_proofs.py`, `test_isf_scaling.py`, `test_minimed_predictor.py`, `test_minimed_predictor_v2.py`.

## 3. Bugs & Methodological Issues Found and Fixed During Audit
1. **12-Sample / 55-Minute Elapsed Span Disambiguation (`WindowSpec`):** Explicitly separated `n_input_samples = 12` (12 five-minute bins, often informally called '60 minutes of history') from `elapsed_span_minutes = 55.0` (`t-55m` to `t`), with strict Pydantic cross-field validation and `np.searchsorted` target matching (`30.0 ± 2.5 min`).
2. **Cross-Split Temporal Overlap Guard (`DataLeakageError`):** Added an explicit per-subject check in `build_partitioned_windows()` verifying `min(test.timestamp) > max(train.timestamp)` so malformed or overlapping train/test files cannot leak.
3. **Invalid-Row Contiguity Guard (`windows.py`):** Computed `bad_rows_prefix` before filtering invalid rows so an invalid/NaN/out-of-bounds reading inside an input window is guaranteed to reject that window without silent interpolation.
4. **Small-N Statistical Qualification (`metrics.py`):** Switched subject-level Wilcoxon signed-rank to `method='exact'` for `6 <= N <= 25`, added 2,000-resample paired subject-level bootstrap 95% CIs + IQR, documented minimum attainable exact p-value `2^(1-N)` (e.g. `0.000488` at `N=12`), and suppressed p-values when `N < 6`.
5. **Hard Synthetic & Benchmark-Tier Tagging (`is_synthetic`, `benchmark_tier`):** Every report, CSV, and `ModelRegistryEntry` now carries `is_synthetic` and `benchmark_tier` (`primary_ohiot1dm` vs `synthetic_software_validation` vs `personal_exploratory_only`).

## 4. Data Leakage Audit Findings
- **Official Train/Test Isolation:** Verified (`test` windows come strictly from `source_split='test'`; overlapping timestamps raise `DataLeakageError`).
- **Chronological Train/Val Embargo:** Verified on Dataset A (`embargo_purged=204` boundary windows discarded so `min(val.window_start_ts) > max(train_fit.target_ts)` holds strictly per subject).
- **Train-Only Normalization:** Verified by adversarial poisoning test (`test_embargo_and_poisoning_data_leakage_proofs`): corrupting `val` with `-50000` / `+99999` leaves `TrainOnlyStandardScaler` `mean_mgdl` and `std_mgdl` bit-for-bit identical.
- **Ridge Hyperparameter Isolation:** `alpha` selection evaluates `val` MAE only; `test` is never passed to `fit()`.
- **Zero Future / Gap Interpolation:** Verified across injected 45-minute dropout gaps and missing/OOB readings.

## 5. Analytical Baseline Correctness Checks
- **Persistence (`iints-persistence-glucose-v1`):** Verified on two-subject linear ramps (`+2 mg/dL/step` -> exact MAE/RMSE/MedAE = `12.000 mg/dL`; `-1 mg/dL/step` -> exact MAE/RMSE/MedAE = `6.000 mg/dL`; mean subject MAE = `9.000 mg/dL`).
- **Ridge (`iints-ridge-glucose-v1`):** Verified exact shape `(N,)` output, recovery of exact 2-D linear ramp operator (`MAE < 1e-3 mg/dL`), and clean exclusion of `NaN` targets.
- **Deterministic Repeatability & Registry Integrity:** Two independent runs (`runA` vs `runB`) matched identically (`True`); checkpoint SHA-256 verification and reload matched to machine precision (`True`).

## 6. Dataset Availability & Quality
- **Official OhioT1DM Dataset:** **MISSING / BLOCKING DEPENDENCY FOR PRIMARY HUMAN BENCHMARK** — `OhioT1DM-volledig` not found locally. Per scientific protocol, NO synthetic or personal CareLink data was substituted.
- **Synthetic Software-Validation Cohort (`iints_virtual`, `is_synthetic=True`):** `12` virtual subjects, `47700` valid CGM observations, `228` simulated dropout gaps (>6.5m).

## 7. Executed Baseline Results
### 7A. Synthetic Software-Validation Pipeline (`iints_virtual` — NOT Human Clinical Evidence)
| model_id | is_synthetic | n_samples | mae | rmse | median_abs_error | clarke_ab_pct | mean_subject_mae | sd_subject_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iints-persistence-glucose-v1 | True | 12396 | 15.913 | 22.160 | 11.100 | 99.693 | 15.913 | 0.398 |
| iints-ridge-glucose-v1 | True | 12396 | 12.466 | 17.293 | 9.301 | 99.750 | 12.466 | 0.479 |

- **Paired Subject-Level Comparison (`Ridge - Persistence`, N=12 virtual subjects):** Mean ΔMAE = `-3.447 mg/dL` (95% CI `[-3.762, -3.084]`), improved `12/12`, exact Wilcoxon `p = 0.000488`.

### 7B. Official OhioT1DM Human Benchmark (`ohiot1dm`)
_No real OhioT1DM results executed yet because official XML files are not present locally._

## 8. Scientific Separation Audit & Remaining Limitations
1. **Strict Separation from Prior `Study 001` (100 Virtual Patients, MAE 8.43 mg/dL):** That earlier benchmark belongs exclusively to `iints.research` (`hovorka_graybox_v2` state-space simulator using known insulin + announced carbohydrate inputs evaluated against noiseless simulated plasma glucose). Those numbers are **separate simulation results** and must never be cited as results of univariate `iints.forecasting` or OhioT1DM.
2. **Qualification on Prior `Study 001` p-values (`4.68e-17`):** In `Study 001`, `N=100` virtual profiles are IID draws from a parametric Hovorka parameter distribution, so `N` can be made arbitrarily large in silico (`p -> 0`). By contrast, real OhioT1DM has fixed `N=12` human participants (`p_min = 2^-11 = 0.000488`), requiring cautious effect-size + CI reporting rather than p-value chasing.
3. **Univariate Scope Limitation:** Version 1 intentionally uses 12 CGM samples only (no insulin, meals, or activity); post-prandial turning points remain fundamentally uncertain without exogenous inputs.

## 9. Explicit Gate Decision
**DECISION:** **CONDITIONAL GO FOR PHASE 3 CODE DEVELOPMENT (Software, Anti-Leakage & Synthetic Baselines 100% Verified; Primary Scientific Claims Remain BLOCKED Until Official OhioT1DM XMLs Are Placed Locally)**
