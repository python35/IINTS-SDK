# IINTS Simulation SDK 2.0 & IINTS Research Lab — Architecture Guide

> **Pre-Clinical Scope:** IINTS is an in-silico research and educational platform. Neither the SDK nor bundled virtual patient adapters carry regulatory clearance, and no output may be used for clinical insulin dosing.

## 1. Fundamental Architectural Separation

| Layer | Package | Core Responsibility |
|---|---|---|
| **IINTS Simulation SDK 2.0** (*De Wetenschappelijke Motor*) | `src/iints/sdk2/` | Standardized virtual patient adapters, sensor fault & display physics, blind causal algorithm sandbox, deterministic SHA-256 seeding, Parquet + SQLite provenance. |
| **IINTS Research Lab** (*De Onderzoeksomgeving*) | `src/iints/research/` | Hypothesis-driven experiment designer, human approval gate, systematic OOD boundary generator, checkpointed batch runner, within-patient paired statistics, deterministic scientific report generator. |
| **IINTS Research Studio** (*Web Interface & API*) | `src/iints/studio/` | FastAPI HTTP service & interactive Studio UI strictly separated from the mathematical kernel. |

---

## 2. SDK 2.0 Modules (`src/iints/sdk2/`)

1. **`contracts.py`**: Pydantic v2 declarative schemas (`ExperimentSpec`, `PatientSpec`, `ScenarioSpec`, `SensorFaultSpec`, `AlgorithmSpec`, `MissingDataContract`, `EvaluationSpec`) with YAML serialization.
2. **`patients.py`** (*Module A — Virtual Patient Engine*): Interchangeable `VirtualPatientAdapter` protocol wrapping:
   - `hovorka_equilibrium`: 19-state Hovorka ODE initialized at exact fasting steady state (`equilibrium.py`),
   - `iints_ode`: Fast educational ODE (`CustomPatientModel`),
   - `bergman`: Bergman Minimal Model ODE for cross-simulator validation,
   - `simglucose`: Optional open-source external simulator adapter.
3. **`sensors.py`** (*Module B — Sensor Simulation Engine*): Explicitly separates `bg_true_mgdl`, `ig_mgdl` (interstitial lag), `cgm_clean_mgdl` (AR(1) noise + bias + drift + compression lows), `cgm_observed_mgdl`, and `cgm_displayed_mgdl` (`nan_blank` vs `stale_hold`), plus `steps_into_dropout` and `steps_since_recovery`.
4. **`forecasters.py`** (*Module C — Algorithm Sandbox*): `GlucoseForecaster` ABC + blind evaluation harness that physically strips ground-truth columns (`bg_true_mgdl`, `ig_mgdl`, `cgm_clean_mgdl`) and future timestamps before invoking `predict_horizons()`. Built-in models: `persistence_v1`, `savgol_linear_v1`, `ar2_v1`, `ridge_ml_v1`, and `hovorka_graybox_v2`.
5. **`reproducibility.py`** (*Module D — Experiment Reproducibility*): Hierarchical `derive_seed()` (guaranteeing identical patient physiology across sensor arms), `capture_provenance()` (Git commit, Python/NumPy/SciPy versions, SHA-256 artifact digests), Parquet/CSV writers, and `ExperimentStore` (`experiments.sqlite`).

---

## 3. Research Lab Modules (`src/iints/research/`)

1. **`designer.py`** (*A. Experiment Designer & Approval Gate*): Generates `ResearchPlan` specifications in `status="proposed"`. Execution is blocked (`PlanNotApprovedError`) until `approve_research_plan()` locks `plan_lock_sha256`. Includes templates for:
   - `study_001_cgm_dropout` (*Direction A — CGM Dropout Robustness Benchmark*),
   - `study_002_inclusive_stale_ui` (*Direction B — Blank vs Unflagged Stale Display Hazard*),
   - `study_003_cross_simulator` (*Direction C — Cross-Simulator Validation*).
2. **`scenarios.py`** (*B. Virtual Cohort & Boundary Generator*): Stratified cohort generator with strictly held-out `unseen` patient split (30%) plus adversarial `ood_stress` boundary scenarios (`ood_postprandial_peak_dropout`, `ood_nocturnal_descent_dropout`).
3. **`runner.py`** (*C. Checkpointed Batch Runner*): Atomic cell execution with `checkpoints.jsonl` resume support, `ThreadPoolExecutor` parallelism, `failures.jsonl` error isolation, and SQLite indexing.
4. **`statistics.py`** (*D. Within-Patient Paired Statistics*): Aggregates per virtual patient first to prevent timepoint pseudoreplication, then computes paired $\Delta\text{MAE}_i = \text{MAE}_i(\text{cond}) - \text{MAE}_i(\text{control})$, 2,000-resample bootstrap 95% CIs, Wilcoxon signed-rank p-values, and Holm-Bonferroni FWER adjustment.
5. **`reports.py`** (*E. Verifiable Report Generator*): Produces `SCIENTIFIC_REPORT.md` and 4-panel `study_dashboard.png` **100% deterministically from stored artifacts** (zero language-model-generated numbers).

---

## 4. Quick CLI Commands

```powershell
# List templates, simulators, and forecasters
iints lab templates

# Run a single declarative SDK 2.0 YAML experiment
iints lab single examples/sdk2/cgm_dropout_001.yaml --out-dir results/cgm_dropout_001

# Propose, review, approve, and run Study 001
iints lab design --template study_001_cgm_dropout --profiles 100 --out results/study_001/plan.yaml
iints lab approve results/study_001/plan.yaml --reviewer "Rune Bobbaers"
iints lab run results/study_001/plan.yaml --workers 4

# Launch interactive Research Studio web UI
iints lab studio --port 8765
```
