# 06 — Datasets & Evaluation Benchmarks for Closed-Loop T1D Research

Evaluating open-source closed-loop insulin algorithms credibly requires combining *in silico* virtual cohorts with real-world clinical and free-living datasets (for glucose forecasting, realism calibration, Digital Twin replay via `ReplayBG`, and offline RL).

> **Data governance rule:** Never commit private or DUA-restricted patient datasets into this repository. Keep raw downloads in local git-ignored directories (see [`.gitignore`](../.gitignore) and [`docs/DIABETES_RESEARCH_DATASETS.md`](../docs/DIABETES_RESEARCH_DATASETS.md)).

---

## Curated Dataset Matrix

| Dataset | Cohort & Duration | Signals Included | Access | Best Research Use | Link / Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OhioT1DM (2018 & 2020)** | 12 adults with T1D, 8 weeks each | CGM (5 min), basal/bolus insulin, SMBG, carbs, exercise/sleep/stress, fitness band | DUA (academic request) | Standard held-out benchmark for BG forecasting & subject-level generalization | Marling & Bunescu (2020) · [UNC Charlotte](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html) |
| **T1DEXI & T1DexiP** | Adults (T1DEXI) & pediatric (T1DexiP), 4 weeks | CGM, pump logs, meals, structured + at-home exercise, heart rate & step wearables | Public / Vivli / Jaeb request | **Primary dataset for exercise-aware closed-loop** & delayed-hypo stress testing | [Jaeb Public Datasets](https://public.jaeb.org/datasets/diabetes) |
| **OpenAPS Data Commons** | Large free-living DIY AID cohort (multi-year) | High-res CGM, `oref0`/`oref1` temp basals, SMBs, carb logs, free-living events | Community request (Open Humans) | **Best real-world open-source AID log dataset** for offline RL & `oref` replay | [openaps.org/outcomes/data-commons](https://openaps.org/outcomes/data-commons/) |
| **DCLP3 / iDCL (Control-IQ Trial)** | 168 participants, 6 months RCT | Commercial Control-IQ closed-loop CGM + pump + meal telemetry | Public download (NIDDK / Jaeb) | Ground-truth commercial MPC comparator & Digital Twin replay | Brown et al. (2019, *NEJM*) · [NIDDK](https://repository.niddk.nih.gov/studies/dclp3/) |
| **Jaeb Loop Observational Study** | 558 open-source Loop users | Real-world iOS Loop CGM + insulin + meal telemetry | Public download (Jaeb) | External validation of open-source Loop controller dynamics | Lum et al. (2021, *DTT*) · [Jaeb](https://public.jaeb.org/datasets/diabetes) |
| **AZT1D (2025)** | 25 individuals on AID systems | CGM, insulin, carbohydrate logs | Open download (Mendeley Data) | Free-living AID realism calibration & pretraining (supported by `research/prepare_azt1d.py`) | [Mendeley Data](https://data.mendeley.com/datasets/gk9m674wcx/1) |
| **HUPA-UCM** | Free-living T1D cohort | CGM, insulin, carbs, physical activity | Open download (Mendeley Data) | Multimodal forecasting & realism envelopes (supported by `research/prepare_hupa_ucm.py`) | [Mendeley Data](https://data.mendeley.com/datasets/3hbcscwz44/1) |
| **DiaTrend** | 54 individuals with T1D | Longitudinal CGM, pump bolus/basal, carb logs, pump settings | Controlled access (Synapse) | Large-scale longitudinal forecasting & settings variability | [DOI 10.7303/syn38187184](https://doi.org/10.7303/syn38187184) |
| **AIDE T1D & PEDAP** | Older adults (AIDE) & young children (PEDAP) | Clinical trial AID CGM & pump records | Public download (Jaeb) | Age-subgroup generalization (elderly & pediatric edge cases) | [Jaeb Public Datasets](https://public.jaeb.org/datasets/) |
| **T1D-UOM** | 17 adults, 12 weeks | CGM, basal/bolus, macronutrients, activity, sleep | Manual download (Zenodo) | Multimodal meal macronutrient & sleep modeling | [DOI 10.5281/zenodo.15806142](https://doi.org/10.5281/zenodo.15806142) |
| **MetaboNet** | Harmonized multi-study aggregation | Unified ML-ready schema across major T1D cohorts | Mixed (respects source DUAs) | Cross-cohort standardized external validation | [metabo-net.org](https://metabo-net.org/) |
| **Glucose-ML Collection** | Curated benchmark taxonomy | Standardized evaluation protocols across public T1D datasets | Reference map | Designing leakage-free benchmark protocols | [arXiv:2507.14077](https://arxiv.org/abs/2507.14077) |

---

## Core Dataset Papers

1. **Marling, C., & Bunescu, R. (2020).** *The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020.* **Proceedings of the 5th International Workshop on Knowledge Discovery in Healthcare Data (CEUR Workshop Proceedings)**, 2675, 71–74.
   - **URL:** [ceur-ws.org/Vol-2675/paper11.pdf](https://ceur-ws.org/Vol-2675/paper11.pdf) · **BibTeX:** `marling2020_ohiot1dm`

---

## Best Practices for University Benchmarks

1. **Subject-level splits only:** Never split the same participant across training and test sets; report cluster-level confidence intervals across subjects (`n = subjects`, not `n = windows`).
2. **Hold out an external cohort:** E.g., train/calibrate on AZT1D + HUPA-UCM (or synthetic simulations) and evaluate untouched on OhioT1DM or DCLP3.
3. **Report full clinical & safety metrics together:** Always report TIR (70–180 mg/dL), TBR (<70 and <54 mg/dL), TAR (>180 and >250 mg/dL), CV%, LBGI/HBGI, and severe-hypoglycemia / supervisor intervention rates — never cherry-pick TIR alone.
4. **Always compare against strong baselines:** Persistence baseline for forecasting; basal-bolus, PID-IFB, and Zone-MPC (or `oref0`) for closed-loop control.
