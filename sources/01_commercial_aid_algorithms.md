# 01 — Commercial AID Systems & Published Control Algorithms

Commercial Automated Insulin Delivery (AID) firmware is proprietary, but nearly every deployed closed-loop controller builds on peer-reviewed control laws published by academic groups. For open-source reference modeling, implementations must rely strictly on published scientific literature, regulatory summaries (e.g., FDA 510(k)), and public user guides.

---

## Quick Comparison Matrix

| Commercial System | Control Architecture | Cycle | Key Mechanism | Primary Academic Lineage | IINTS-AF Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Medtronic MiniMed 670G / 780G (SmartGuard)** | **PID-IFB** + autocorrection rules | 5 min | Proportional–Integral–Derivative with plasma Insulin Feedback (IFB) + frequent auto-correction boluses (780G) | Steil et al. (2006, 2011) | [`src/iints/emulation/medtronic_780g.py`](../src/iints/emulation/medtronic_780g.py) |
| **Tandem t:slim X2 (Control-IQ)** | **MPC / Zone-MPC** + rule-based safety layer | 5 min | 30-min prediction horizon, basal modulation/suspend, automated correction bolus (up to 1/hr) | UVA/Padova; Grosman et al. (2010); Brown et al. (2019) | [`src/iints/emulation/tandem_controliq.py`](../src/iints/emulation/tandem_controliq.py) |
| **CamAPS FX** | **Adaptive / Stochastic NMPC** | 8–12 min | Nonlinear MPC built on the Hovorka compartmental model with online parameter adaptation | Hovorka et al. (2004); Thabit et al. (2015); Tauschmann et al. (2018) | Hovorka model in `src/iints/core/patient/` |
| **Insulet Omnipod 5 (SmartAdjust)** | **Predictive MPC-adjacent** | 5 min | 60-min horizon, adaptive basal rate updated per pod change from Total Daily Dose (TDD) | Brown et al. (2021) | [`src/iints/emulation/omnipod_5.py`](../src/iints/emulation/omnipod_5.py) |
| **Beta Bionics iLet Bionic Pancreas** | **Adaptive PD + meal estimation** | 5 min | Initialized solely from body weight; continuous autonomous adaptation with qualitative meal announcements | Damiano / Russell et al. (2022) | Candidate for open-source reference model |

---

## Key Papers (Ordered by Algorithm Family)

### 1. PID with Insulin Feedback (Medtronic Lineage)

1. **Steil, G. M., Rebrin, K., Darwin, C., Hariri, F., & Saad, M. F. (2006).** *Feasibility of automating insulin delivery for the treatment of type 1 diabetes.* **Diabetes**, 55(12), 3344–3350.
   - **DOI:** [10.2337/db06-0419](https://doi.org/10.2337/db06-0419) · **BibTeX:** `steil2006_pid`
   - **Why it matters:** Foundational clinical demonstration of closed-loop PID insulin delivery emulating first- and second-phase beta-cell secretion.

2. **Steil, G. M., Palerm, C. C., Kurtz, N., Voskanyan, G., Roy, A., Paz, S., & Kandeel, F. R. (2011).** *The Effect of Insulin Feedback on Closed Loop Glucose Control.* **Journal of Clinical Endocrinology & Metabolism**, 96(5), 1402–1408.
   - **DOI:** [10.1210/jc.2010-2578](https://doi.org/10.1210/jc.2010-2578) · **BibTeX:** `steil2011_pid_ifb`
   - **Why it matters:** Introduces **Insulin Feedback (IFB)** — subtracting an estimate of plasma insulin from the PID command to prevent post-prandial insulin stacking and delayed hypoglycemia. Core mathematical mechanism behind Medtronic's closed-loop controller.

3. **Bergenstal, R. M., Nimri, R., Beck, R. W., et al. (2021).** *A comparison of two hybrid closed-loop systems in adolescents and young adults with type 1 diabetes (FLAIR): a multicentre, randomised, crossover trial.* **The Lancet**, 397(10270), 208–219. *(See also FDA 510(k) K193510).*
   - **DOI:** [10.1016/S0140-6736(20)32514-9](https://doi.org/10.1016/S0140-6736(20)32514-9) · **BibTeX:** `bergenstal2021_flair`
   - **Why it matters:** Clinical benchmark comparing next-generation AHCL (MiniMed 780G lineage with automated correction boluses + fuzzy-logic tuning) against earlier PID-only hybrid closed-loop control (MiniMed 670G).

4. **Choudhary, P., Kolassa, R., Keuthage, W., et al. (2022).** *Advanced hybrid closed loop therapy versus conventional treatment in adults with type 1 diabetes (ADAPT): a randomised controlled study.* **The Lancet Diabetes & Endocrinology**, 10(10), 720–731.
   - **DOI:** [10.1016/S2213-8587(22)00212-1](https://doi.org/10.1016/S2213-8587(22)00212-1) · **BibTeX:** `choudhary2022_adapt`
   - **Why it matters:** Benchmark clinical outcome envelope (TIR, TBR, HbA1c) for MiniMed 780G AHCL.

---

### 2. Model Predictive Control & Zone-MPC (Tandem Control-IQ / UVA–Harvard Lineage)

5. **Grosman, B., Dassau, E., Zisser, H. C., Jovanovič, L., & Doyle III, F. J. (2010).** *Zone Model Predictive Control: A Strategy to Minimize Hyper- and Hypoglycemic Events.* **Journal of Diabetes Science and Technology**, 4(4), 961–975.
   - **DOI:** [10.1177/193229681000400428](https://doi.org/10.1177/193229681000400428) · **BibTeX:** `grosman2010_zonempc`
   - **Why it matters:** Replaces a brittle single glycemic setpoint with an acceptable **target zone** in the MPC cost function, reducing actuator chatter and avoiding over-correction under CGM noise.

6. **Brown, S. A., Kovatchev, B. P., Raghinaru, D., et al. (2019).** *Six-Month Randomized, Multicenter Trial of Closed-Loop Control in Type 1 Diabetes.* **New England Journal of Medicine**, 381(18), 1707–1717.
   - **DOI:** [10.1056/NEJMoa1907863](https://doi.org/10.1056/NEJMoa1907863) · **BibTeX:** `brown2019_controliq_nejm`
   - **Why it matters:** Pivotal iDCL (DCLP3) RCT for Tandem Control-IQ; establishes standard 6-month clinical performance ranges and safety outcomes (public dataset available via NIDDK/Jaeb).

7. **Brown, S. A., Raghinaru, D., Emory, E., & Kovatchev, B. (2019).** *First Look at Control-IQ: A New-Generation Automated Insulin Delivery System.* **Diabetes Technology & Therapeutics**, 21(10), 592–597.
   - **DOI:** [10.1089/dia.2019.0226](https://doi.org/10.1089/dia.2019.0226) · **BibTeX:** `brown2019_controliq_dtt`
   - **Why it matters:** Concise overview of Control-IQ's modular architecture, sleep/exercise modes, and automated correction bolus behavior (see also FDA 510(k) K191289).

---

### 3. Nonlinear Adaptive MPC (Cambridge / Hovorka / CamAPS FX Lineage)

8. **Hovorka, R., Canonico, V., Chassin, L. J., et al. (2004).** *Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes.* **Physiological Measurement**, 25(4), 905–920.
   - **DOI:** [10.1088/0967-3334/25/4/010](https://doi.org/10.1088/0967-3334/25/4/010) · **BibTeX:** `hovorka2004_nmpc`
   - **Why it matters:** Defines both the **Hovorka compartmental ODE model** and the **nonlinear MPC** controller architecture that evolved into CamAPS FX.

9. **Thabit, H., Tauschmann, M., Allen, J. M., et al. (2015).** *Home Use of an Artificial Beta Cell in Type 1 Diabetes.* **New England Journal of Medicine**, 373(22), 2129–2140.
   - **DOI:** [10.1056/NEJMoa1509351](https://doi.org/10.1056/NEJMoa1509351) · **BibTeX:** `thabit2015_cambridge`
   - **Why it matters:** Landmark 12-week free-living home trial validating the Cambridge treat-to-target adaptive MPC algorithm.

10. **Tauschmann, M., Thabit, H., Bally, L., et al. (2018).** *Closed-loop insulin delivery in suboptimally controlled type 1 diabetes: a multicentre, 12-week randomised trial.* **The Lancet**, 392(10155), 1321–1329.
    - **DOI:** [10.1016/S0140-6736(18)31947-0](https://doi.org/10.1016/S0140-6736(18)31947-0) · **BibTeX:** `tauschmann2018_cambridge`
    - **Why it matters:** Demonstrates robustness of adaptive MPC across a broad age range and suboptimally controlled baseline populations.

---

### 4. Tubeless Predictive AID & Autonomous Adaptive Systems (Omnipod 5 & iLet)

11. **Brown, S. A., Forlenza, G. P., Bode, B. W., et al. (2021).** *Multicenter Trial of a Tubeless, On-Body Automated Insulin Delivery System With Customizable Glycemic Targets in Pediatric and Adult Participants With Type 1 Diabetes.* **Diabetes Care**, 44(7), 1630–1640.
    - **DOI:** [10.2337/dc21-0172](https://doi.org/10.2337/dc21-0172) · **BibTeX:** `brown2021_omnipod5`
    - **Why it matters:** Pivotal trial of Omnipod 5 (SmartAdjust); documents TDD-based adaptive basal updates and customizable target behavior (see also FDA 510(k) K203467).

12. **Bionic Pancreas Research Group; Russell, S. J., Beck, R. W., Damiano, E. R., et al. (2022).** *Multicenter, Randomized Trial of a Bionic Pancreas in Type 1 Diabetes.* **New England Journal of Medicine**, 387(13), 1161–1172.
    - **DOI:** [10.1056/NEJMoa2205225](https://doi.org/10.1056/NEJMoa2205225) · **BibTeX:** `russell2022_ilet`
    - **Why it matters:** Pivotal trial of the insulin-only iLet Bionic Pancreas showing high TIR with weight-only initialization and qualitative meal announcements — an important paradigm for low-burden open-source controller design.
