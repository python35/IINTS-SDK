# 04 — Control Theory, AI, Reinforcement Learning & Dual-Hormone Systems

Research on next-generation closed-loop insulin algorithms compares classical control architectures (PID, MPC, Fuzzy Logic) against data-driven learning approaches (Deep Reinforcement Learning, Offline RL, Foundation Models) and dual-hormone controllers.

---

## Control Paradigm Overview

| Paradigm | Strengths | Limitations | Open Research Opportunity |
| :--- | :--- | :--- | :--- |
| **PID / PID-IFB** | Lightweight, transparent, minimal tuning | Linear assumptions; struggles with unannounced meals & large delays | Benchmark baseline against adaptive controllers |
| **MPC / Zone-MPC / NMPC** | Explicitly handles constraints, PK/PD delays, and prediction horizons | Sensitive to model mismatch and intraday sensitivity drift | Online personalization via Digital Twins & fault-aware constraints |
| **Fuzzy Logic Control (FLC)** | Encodes intuitive clinical heuristics without exact ODEs | Harder to prove formal stability across edge cases | Explainable rule layers & supervisory fallback logic |
| **Heuristic PK/PD (`oref0`/`oref1`, `Loop`)** | Fully inspectable, proven in real-world & RCTs | Hand-tuned heuristics and parameter interactions | Formal analysis & stress-testing under degraded sensors/pumps |
| **Deep / Offline RL** | Learns personalized policies for unannounced meals & exercise | Out-of-distribution safety risks & simulator-to-reality gap | **Safe Hybrid RL**: learned policy bounded by a deterministic Safety Supervisor |

---

## Key Papers (Ordered by Theme)

### 1. Control-Engineering Foundations & Reviews

1. **Doyle III, F. J., Huyett, L. M., Lee, J. B., Zisser, H. C., & Dassau, E. (2014).** *Closed-Loop Artificial Pancreas Systems: Engineering the Algorithms.* **Diabetes Care**, 37(5), 1191–1197.
   - **DOI:** [10.2337/dc13-2108](https://doi.org/10.2337/dc13-2108) · **BibTeX:** `doyle2014_engineering`
   - **Why it matters:** Classic control-engineering synthesis comparing PID, MPC, and Fuzzy Logic algorithms, explaining subcutaneous delays, meal disturbances, and safety layers.

### 2. Reinforcement Learning & Safe Offline RL

2. **Tejedor, M., Woldaregay, A. Z., & Godtliebsen, F. (2020).** *Reinforcement learning application in diabetes blood glucose control: A systematic review.* **Artificial Intelligence in Medicine**, 104, 101836.
   - **DOI:** [10.1016/j.artmed.2020.101836](https://doi.org/10.1016/j.artmed.2020.101836) · **BibTeX:** `tejedor2020_rl_review`
   - **Why it matters:** Comprehensive systematic review of RL state representations, reward functions (Kovatchev risk, TIR), simulators, and safety bottlenecks in blood glucose control.

3. **Fox, I., Lee, J., Pop-Busui, R., & Wiens, J. (2020).** *Deep Reinforcement Learning for Closed-Loop Blood Glucose Control.* **Proceedings of Machine Learning Research (MLHC)**, 126, 508–536.
   - **arXiv:** [2009.09051](https://arxiv.org/abs/2009.09051) · **BibTeX:** `fox2020_drl`
   - **Why it matters:** Influential benchmark evaluating Soft Actor-Critic (SAC) policies inside `simglucose` without meal announcements (Fully Closed-Loop), compared against basal-bolus and PID baselines.

4. **Emerson, H., Guy, M., & McConville, R. (2023).** *Offline reinforcement learning for safer blood glucose control in people with type 1 diabetes.* **Journal of Biomedical Informatics**, 142, 104376.
   - **DOI:** [10.1016/j.jbi.2023.104376](https://doi.org/10.1016/j.jbi.2023.104376) · **BibTeX:** `emerson2023_offline_rl`
   - **Why it matters:** Addresses a core barrier to clinical AI — instead of risky online trial-and-error, trains conservative offline RL policies from fixed historical datasets (e.g., TD3+BC, CQL, BCQ) and benchmarks them in simulation.

### 3. Glucose Forecasting & CGM Foundation Models

5. **Li, Z., Natarajan, K., Zhang, W., Zhou, M., Lee, S. A., Zhang, Y., et al. (2026).** *GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring.* **arXiv** preprint (v2).
   - **URL:** [arxiv.org/abs/2605.30865v2](https://arxiv.org/abs/2605.30865v2) · **BibTeX:** `li2026_glucofm`
   - **Why it matters:** Explores pretrained dual-stream foundation models for CGM time series; directly connected to the forecasting experiments in IINTS-AF (`docs/CGM_FOUNDATION_MODELS.md`).

### 4. Dual-Hormone Artificial Pancreas (Insulin + Glucagon)

6. **Haidar, A., Legault, L., Dallaire, M., et al. (2013).** *Glucose-responsive insulin and glucagon delivery (dual-hormone artificial pancreas) in adults with type 1 diabetes: a randomized crossover controlled trial.* **CMAJ**, 185(4), 297–305.
   - **DOI:** [10.1503/cmaj.121265](https://doi.org/10.1503/cmaj.121265) · **BibTeX:** `haidar2013_dual_hormone`
   - **Why it matters:** Demonstrates how counter-regulatory glucagon micro-dosing complements insulin modulation to cut hypoglycemia risk, especially around exercise.

7. **Lv, D., Breton, M. D., & Farhy, L. S. (2013).** *Pharmacokinetics Modeling of Exogenous Glucagon in Type 1 Diabetes Mellitus Patients.* **Diabetes Technology & Therapeutics**, 15(11), 935–941.
   - **DOI:** [10.1089/dia.2013.0150](https://doi.org/10.1089/dia.2013.0150) · **BibTeX:** `lv2013_glucagon_pk`

8. **Haidar, A., Duval, C., Legault, L., & Rabasa-Lhoret, R. (2013).** *Pharmacokinetics of Insulin Aspart and Glucagon in Type 1 Diabetes during Closed-Loop Operation.* **Journal of Diabetes Science and Technology**, 7(6), 1507–1512.
   - **DOI:** [10.1177/193229681300700610](https://doi.org/10.1177/193229681300700610) · **BibTeX:** `haidar2013_pk`

---

## Key Open Research Gaps (2024–2026)

- **Fully Closed-Loop (FCL) without meal announcements:** Combining early meal-onset detection or adaptive controllers with ultra-rapid insulin PK models while avoiding late postprandial hypoglycemia.
- **Safety-Enveloped Learning:** Proving mathematically or empirically that wrapping a learned controller (imitation / RL / neural) inside a deterministic Safety Supervisor prevents hazardous overdosing without causing supervisor-induced hypoglycemia (see IINTS-AF's own open finding in [`docs/EXPERIMENTAL_RESULTS.md`](../docs/EXPERIMENTAL_RESULTS.md)).
- **Exercise-Aware Closed Loop:** Incorporating heart rate, step count, and accelerometry streams (e.g., from T1DEXI) directly into control decisions.
