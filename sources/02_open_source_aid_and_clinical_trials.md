# 02 — Open-Source AID Algorithms & Clinical Evidence

Open-source Automated Insulin Delivery (OS-AID) systems — developed by the patient-led *#WeAreNotWaiting* community — provide fully inspectable reference implementations of closed-loop insulin controllers that have now been evaluated in randomized controlled trials, large observational cohorts, and international clinical consensus statements.

---

## Major Open-Source AID Algorithm Families

| Ecosystem | Core Algorithm | Core Mechanism | Key Innovations | Public Source |
| :--- | :--- | :--- | :--- | :--- |
| **OpenAPS / AndroidAPS / Trio** | **`oref0` / `oref1`** | Transparent heuristic + PK/PD multi-trajectory predictor | 4 simultaneous prediction curves (IOB, COB, Zero-Temp, UAM — *Unannounced Meals*), **SMB** (*Super Micro Bolus*), **Autosens** (24h sensitivity shift), **Autotune** | [github.com/openaps/oref0](https://github.com/openaps/oref0) |
| **Loop / Tidepool Loop** | **`LoopKit`** | Single-trajectory continuous predictive controller (6h horizon) | Aggregates 4 effects: Insulin effect (IOB), Carb absorption model (COB), Glucose Momentum, and **Retrospective Correction** (real-time model-error feedback) | [github.com/LoopKit/Loop](https://github.com/LoopKit/Loop) |

---

## Landmark Papers (Ordered by Significance)

### 1. International Consensus & Clinical Guidance

1. **Braune, K., Lal, R. A., Petruželková, L., Scheiner, G., Winterdijk, P., Schmidt, S., et al. (2022).** *Open-source automated insulin delivery: international consensus statement and practical guidance for health-care professionals.* **The Lancet Diabetes & Endocrinology**, 10(1), 58–74.
   - **DOI:** [10.1016/S2213-8587(21)00267-9](https://doi.org/10.1016/S2213-8587(21)00267-9) · **BibTeX:** `braune2022_osaid_consensus`
   - **Why it matters:** The definitive international medical consensus paper on open-source AID. Reviews `oref0`/`oref1`, AndroidAPS, and Loop architectures, clinical safety data, and ethical/legal considerations.

### 2. Randomized Controlled Trial & Prospective Clinical Studies

2. **Burnside, M. J., Lewis, D. M., Crocket, H. R., Meier, R. A., Williman, J. A., Sanders, O. J., et al. (2022).** *Open-Source Automated Insulin Delivery in Type 1 Diabetes (CREATE Trial).* **New England Journal of Medicine**, 387(10), 869–881.
   - **DOI:** [10.1056/NEJMoa2203913](https://doi.org/10.1056/NEJMoa2203913) · **BibTeX:** `burnside2022_create`
   - **Why it matters:** First multicenter RCT evaluating an open-source AID algorithm (modified AndroidAPS / OpenAPS `oref1`) in children and adults vs. sensor-augmented pump therapy. Demonstrated +14 percentage points TIR improvement without severe hypoglycemia or DKA.

3. **Lum, J. W., Bailey, R. J., Barnes-Lomen, V., Naranjo, D., Hood, K. K., Lal, R. A., et al. (2021).** *A Real-World Prospective Study of the Safety and Effectiveness of the Loop Open Source Automated Insulin Delivery System.* **Diabetes Technology & Therapeutics**, 23(5), 367–375.
   - **DOI:** [10.1089/dia.2020.0535](https://doi.org/10.1089/dia.2020.0535) · **BibTeX:** `lum2021_loop`
   - **Why it matters:** Large prospective Jaeb Center observational study (558 participants) of the open-source Loop algorithm; its evidence base directly supported the FDA iAGC submission for Tidepool Loop (public dataset available via Jaeb).

### 3. Systematic Reviews & Policy Analysis

4. **Knoll, C., Peacock, S., Wäldchen, M., Cooper, D., Aulakh, S. K., Raile, K., Hussain, S., & Braune, K. (2022).** *Real-world evidence on clinical outcomes of people with type 1 diabetes using open-source and commercial automated insulin dosing systems: A systematic review.* **Diabetic Medicine**, 39(5), e14741.
   - **DOI:** [10.1111/dme.14741](https://doi.org/10.1111/dme.14741) · **BibTeX:** `knoll2022_systematic_review`
   - **Why it matters:** Direct head-to-head literature synthesis comparing real-world glycemic outcomes (TIR, TBR, HbA1c) of open-source AID (`oref0`/AAPS/Loop) against commercial AID systems.

5. **Oliver, N., Reddy, M., Marriott, C., Walker, T., & Heinemann, L. (2019).** *Open source automated insulin delivery: addressing the challenge.* **npj Digital Medicine**, 2, 124.
   - **DOI:** [10.1038/s41746-019-0202-1](https://doi.org/10.1038/s41746-019-0202-1) · **BibTeX:** `oliver2019_osaid`
   - **Why it matters:** Early landmark review mapping the technical, clinical, governance, and medico-legal challenges of patient-developed open-source closed-loop algorithms.

---

## Relevance for University Research

- Implementing clean, auditable Python reference versions of **`oref0`/`oref1`** (with SMB & UAM) and **`Loop`** (with Retrospective Correction) inside IINTS-AF alongside **PID-IFB** and **Zone-MPC** enables the first apples-to-apples *in silico* comparison of open-source vs. commercial control paradigms under identical virtual patients and injected sensor/pump faults.
