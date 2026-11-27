# IINTS Neural Forecasting v1 — Phase 1 Architecture Audit & Research Protocol

> **Scope & Safety Boundary:** Pre-clinical in-silico and retrospective research benchmark only. No model in `iints.forecasting` has clinical validation or regulatory clearance. Predictions carry `"clinical_use": false` and are strictly isolated from insulin delivery logic and physical pump hardware.

---

## 1. Repository Architecture & Reuse Audit

Before introducing `src/iints/forecasting/`, the existing `IINTS-SDK` (`v1.5.37`) was audited to maximize reuse and prevent duplication or breaking changes:

| Existing SDK Component | Location | Reuse Decision for `iints.forecasting` |
|---|---|---|
| **Dataset Registry (`ohio_t1dm`)** | `src/iints/data/datasets.json`, `registry.py` | **Reused directly** for citation metadata and `access="request"` enforcement (blocks unverified downloads; requires local UNC Charlotte DUA files). |
| **Safe XML Parser (`defusedxml`)** | `pyproject.toml` (`defusedxml>=0.7.1`) | **Reused directly** in `ohio_adapter.py` to parse official OhioT1DM `.xml` patient files safely against XXE/entity-expansion vulnerabilities. |
| **CareLink Section Parser** | `src/iints/analysis/carelink_sections.py` | **Reused** as an optional explicitly-labeled secondary real-CGM adapter (`dataset="carelink"`), kept strictly separate from `ohiot1dm`. |
| **SDK 2.0 Virtual Patient & Sensor Engine** | `src/iints/sdk2/` (`patients.py`, `sensors.py`) | **Reused** for **Dataset A** (`dataset="iints_virtual"`) software verification and controlled sensor-fault injection (Phase 6), never substituted silently for OhioT1DM. |
| **Clinical & Directional Forecast Grading** | `src/iints/analysis/error_grid.py`, `prediction_accuracy.py` | **Reused directly** for Clarke Error Grid A+B % and glycemic range definitions (`<70`, `70–180`, `>180 mg/dL`). |
| **Cryptographic Provenance & Hashing** | `src/iints/sdk2/reproducibility.py` | **Reused directly** (`sha256_file`, `sha256_json`, `capture_provenance`) for dataset fingerprinting and Model Registry checksums. |
| **Jetson Telemetry & Edge Memory Probes** | `src/iints/jetson/endurance.py`, `live_patient/edge_benchmark.py` | **Catalogued for Phase 7**: `collect_jetson_hardware_info()` already probes `tegrastats`, `nvidia-smi`, and `/sys/class/thermal`; `_max_rss_mb()` measures peak RSS on Linux/macOS/Windows. Not modified in Phase 1–2. |
| **Control / Dosing Algorithms** | `src/iints/core/algorithms/`, `live_patient/pico_pump.py` | **Strictly untouched & isolated** — forecasting outputs are read-only research records. |

---

## 2. Official OhioT1DM Dataset Specification & Governance

- **Reference:** Marling C, Bunescu R. *The OhioT1DM Dataset for Blood Glucose Level Prediction* (KDH@IJCAI 2018; Update 2020).
- **Cohorts & Subject IDs (12 adults with T1D on insulin pump + Medtronic Enlite CGM, ~8 weeks each):**
  - **2018 Cohort (6 subjects):** `559`, `563`, `570`, `575`, `588`, `591`
  - **2020 Cohort (6 subjects):** `540`, `544`, `552`, `567`, `584`, `596`
- **Access & Redistribution Policy:** Controlled research access via Data Use Agreement (DUA) from UNC Charlotte. Raw OhioT1DM files **must never be committed to Git** (enforced via `.gitignore`: `OhioT1DM-volledig/`) and **must never be fabricated or silently replaced by synthetic data**.
- **Official File Partitioning & XML Format:**
  - Each subject has two official files: `<id>-ws-training.xml` (`source_split="train"`) and `<id>-ws-testing.xml` (`source_split="test"`, roughly the final ~10 days).
  - Root element `<patient id="559" weight="..." insulin_type="...">` contains `<glucose_level><event ts="DD-MM-YYYY HH:MM:SS" value="151"/></glucose_level>` (timestamps in `DD-MM-YYYY HH:MM:SS`, nominal 5-minute interval, glucose in `mg/dL`).
- **Known Real-World Data Quirks Handled Explicitly by `ohio_adapter.py`:**
  1. Sensor warm-ups, dropouts, and tape/transmission gaps spanning 10 minutes to >12 hours.
  2. Occasional timestamp jitter (e.g., 4–6 min intervals instead of exact 300 s) or clock resets.
  3. Duplicate timestamps or out-of-bounds values (`< 20` or `> 600 mg/dL`).

---

## 3. Pre-Registered Forecasting Task & Anti-Leakage Protocol (v1)

### 3.1 Mathematical Problem Definition (Version 1 — Univariate CGM)
Given $L = 12$ consecutive valid CGM observations over the past 60 minutes sampled at nominal $\Delta t = 5\text{ min}$:
$$\mathbf{x}_t = \begin{bmatrix} G_{t-55}, & G_{t-50}, & \dots, & G_{t-5}, & G_t \end{bmatrix}^\top \in \mathbb{R}^{12}$$
predict future CGM glucose at horizon $H = 30\text{ min}$ ($k = 6$ intervals ahead):
$$\hat{G}_{t+30} = f(\mathbf{x}_t)$$

- **Features in v1:** Strictly historical CGM glucose only (no meal logs, no insulin doses, no subject ID embedding) so baseline temporal predictability is isolated cleanly before multimodal ablation.

### 3.2 Contiguous Window Validity & Zero-Interpolation Rule
A candidate sample $(\mathbf{x}_t, y_t)$ at index $i$ within a single `(subject_id, source_split)` stream is admitted **if and only if**:
1. All 12 input rows $j \in \{i-11, \dots, i\}$ have `is_valid == True` (finite glucose in `[20, 600] mg/dL`, non-duplicate, strictly increasing timestamp).
2. Every consecutive input step $\Delta t_m = t_{i-11+m} - t_{i-12+m}$ lies within `[3.5, 6.5] minutes`, and total input history span $t_i - t_{i-11}$ lies within `[51.0, 59.0] minutes`.
3. There exists a valid future observation $y_t = G_r$ inside the **same `source_split`** whose timestamp offset $(t_r - t_i)$ lies within `[27.5, 32.5] minutes` ($30 \pm 2.5\text{ min}$).
4. **No gap interpolation:** Missing readings are **never** linearly or spline-interpolated across gaps, because two-sided interpolation injects future post-gap glucose values into pre-gap inputs.

### 3.3 Three-Way Chronological Partitioning with Embargo Buffer
1. **Official Test Set (`test`):** Loaded exclusively from `*-ws-testing.xml` (`source_split == "test"`). Never touched during scaler fitting, model training, or hyperparameter tuning.
2. **Chronological Train / Validation Split within Official `train`:**
   - For each subject's `source_split == "train"` valid windows ordered chronologically by origin time $t_i$:
   - Let $t_{\text{cut}}$ be the origin timestamp at quantile $1 - \alpha_{\text{val}}$ (default `80%` `train_fit`, `20%` `val`).
   - **`train_fit`:** All training windows whose **target timestamp** $t_{\text{target}} \le t_{\text{cut}}$.
   - **Embargo Zone:** Windows whose earliest input timestamp $t_{i-11} \le t_{\text{cut}}$ while $t_{\text{target}} > t_{\text{cut}}$ are **purged (discarded)** so that no target value seen during `train_fit` ever appears anywhere in a `val` input window or target!
   - **`val`:** Remaining training windows with earliest input timestamp $t_{i-11} > t_{\text{cut}}$.
3. **Train-Only Preprocessing:** Mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ (`TrainOnlyStandardScaler`) and Ridge penalty $\alpha$ are computed exclusively from `train_fit` (with $\alpha$ selected solely by `val` MAE).

---

## 4. Standardized Evaluation & Reporting Protocol

- **Primary Metric:** Pooled & Per-Subject $\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^N |G_i - \hat{G}_i|$ (`mg/dL`).
- **Secondary Metrics:**
  - $\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (G_i - \hat{G}_i)^2}$ (`mg/dL`),
  - Median Absolute Error (`MedAE`, `mg/dL`),
  - Clarke Error Grid clinically acceptable rate (`A + B %`),
  - **Per-Subject Distribution:** Mean, SD, Median, IQR, and Min/Max across independent subjects ($N = 12$ on OhioT1DM) + paired Wilcoxon signed-rank test on subject-level MAEs (avoiding autocorrelation pseudoreplication across 5-minute steps),
  - **Glycemic Range Stratification (on true target $G_{t+30}$):** Hypoglycemia (`< 70 mg/dL`), Target (`70–180 mg/dL`), Hyperglycemia (`> 180 mg/dL`),
  - **Rate-of-Change Stratification:** Evaluated across falling (`< -1 mg/dL/min`), steady (`[-1, +1]`), and rising (`> +1 mg/dL/min`) transitions.

---

## 5. Phase Gate (Before Phase 3–8)

Per the pre-registered roadmap, **GRU (`gru.py`), TCN (`tcn.py`), ONNX/TensorRT Jetson worker (`deployment/`), and Research Studio UI tabs** remain gated until:
1. `ohio_adapter`, quality checks, and embargoed windowing pass all unit & anti-leakage proof tests, and
2. `PersistenceBaseline` (`Model 0`) and `RidgeCGMBaseline` (`Model 1`) produce verified, registered baseline artifacts.
