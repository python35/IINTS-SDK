# 03 — Physiology Models, Simulators, Digital Twins & Glycemic Metrics

Preclinical evaluation of closed-loop insulin algorithms depends on mechanistic physiological models, virtual patient simulators, individualized Digital Twins, and internationally standardized outcome metrics.

---

## 1. Core Glucose–Insulin Physiology & Meal Models

1. **Bergman, R. N., Ider, Y. Z., Bowden, C. R., & Cobelli, C. (1979).** *Quantitative estimation of insulin sensitivity.* **American Journal of Physiology-Endocrinology and Metabolism**, 236(6), E667–E677.
   - **DOI:** [10.1152/ajpendo.1979.236.6.E667](https://doi.org/10.1152/ajpendo.1979.236.6.E667) · **BibTeX:** `bergman1979_minimal`
   - **Role:** Minimal model of glucose effectiveness and remote insulin action; baseline ODE model in IINTS-AF.

2. **Hovorka, R., Canonico, V., Chassin, L. J., et al. (2004).** *Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes.* **Physiological Measurement**, 25(4), 905–920.
   - **DOI:** [10.1088/0967-3334/25/4/010](https://doi.org/10.1088/0967-3334/25/4/010) · **BibTeX:** `hovorka2004_nmpc`
   - **Role:** Subcutaneous insulin absorption, glucose transport/disposal/EGP compartmental ODE model; basis for the 19-state Hovorka-style model in IINTS-AF.

3. **Dalla Man, C., Rizza, R. A., & Cobelli, C. (2007).** *Meal Simulation Model of the Glucose-Insulin System.* **IEEE Transactions on Biomedical Engineering**, 54(10), 1740–1749.
   - **DOI:** [10.1109/TBME.2007.893506](https://doi.org/10.1109/TBME.2007.893506) · **BibTeX:** `dallaman2007_meal`
   - **Role:** Standard gastrointestinal meal-absorption and glucose–insulin physiological model underlying the UVA/Padova simulator.

---

## 2. In-Silico Simulators & Digital Twins

4. **Dalla Man, C., Micheletto, F., Lv, D., Breton, M., Kovatchev, B., & Cobelli, C. (2014).** *The UVA/PADOVA Type 1 Diabetes Simulator: New Features.* **Journal of Diabetes Science and Technology**, 8(1), 26–34.
   - **DOI:** [10.1177/1932296813514502](https://doi.org/10.1177/1932296813514502) · **BibTeX:** `dallaman2014_uvapadova`
   - **Role:** FDA-accepted preclinical simulator (substitute for animal trials) with virtual adult, adolescent, and child cohorts.

5. **Visentin, R., Campos-Náñez, E., Schiavon, M., et al. (2018).** *The UVA/Padova Type 1 Diabetes Simulator Goes From Single Meal to Single Day.* **Journal of Diabetes Science and Technology**, 12(2), 273–281.
   - **DOI:** [10.1177/1932296818757747](https://doi.org/10.1177/1932296818757747) · **BibTeX:** `visentin2018_uvapadova`
   - **Role:** Incorporates diurnal insulin-sensitivity variability and dawn phenomenon into multi-meal single-day simulation.

6. **Xie, J. (2018).** *Simglucose v0.2.1: A Type-1 Diabetes Simulator Implemented in Python for Reinforcement Learning Purpose.* GitHub.
   - **URL:** [github.com/jxx123/simglucose](https://github.com/jxx123/simglucose) · **BibTeX:** `xie2018_simglucose`
   - **Role:** Widely used open-source Python/Gym implementation of the 2008 UVA/Padova simulator for control and RL benchmarking.

7. **Cappon, G., Vettoretti, M., Sparacino, G., Del Favero, S., & Facchinetti, A. (2023).** *ReplayBG: A Digital Twin-Based Methodology to Identify a Personalized Model From Type 1 Diabetes Data and Simulate Glucose Concentrations to Assess Alternative Therapies.* **IEEE Transactions on Biomedical Engineering**, 70(11), 3227–3238.
   - **DOI:** [10.1109/TBME.2023.3286856](https://doi.org/10.1109/TBME.2023.3286856) · **BibTeX:** `cappon2023_replaybg`
   - **Role:** Open-source Python Digital Twin framework (`py_replay_bg`) that identifies personalized patient models from real CGM/insulin/meal logs so alternative dosing algorithms can be replayed retrospectively.

8. **Mujahid, O., Contreras, I., Beneyto, A., & Vehi, J. (2024).** *Generative deep learning for the development of a type 1 diabetes simulator.* **Communications Medicine**, 4(1), 51.
   - **DOI:** [10.1038/s43856-024-00476-0](https://doi.org/10.1038/s43856-024-00476-0) · **BibTeX:** `mujahid2024_generative_sim`
   - **Role:** Data-driven generative simulator contrasting purely mechanistic ODE approximations with empirical real-world CGM variability.

---

## 3. Disturbances, Sensor Lag & Counterregulation

9. **Rebrin, K., Sheppard, N. F., & Steil, G. M. (2010).** *Use of Subcutaneous Interstitial Fluid Glucose to Estimate Blood Glucose: Revisiting Delay and Sensor Offset.* **Journal of Diabetes Science and Technology**, 4(5), 1087–1098.
   - **DOI:** [10.1177/193229681000400507](https://doi.org/10.1177/193229681000400507) · **BibTeX:** `rebrin2010_cgm_lag`
   - **Role:** Physiological blood-to-interstitial CGM diffusion delay — a primary source of closed-loop phase lag.

10. **Visentin, R., Dalla Man, C., Kudva, Y. C., Basu, A., & Cobelli, C. (2015).** *Circadian Variability of Insulin Sensitivity: Physiological Input for In Silico Artificial Pancreas.* **Diabetes Technology & Therapeutics**, 17(1), 1–7.
    - **DOI:** [10.1089/dia.2014.0192](https://doi.org/10.1089/dia.2014.0192) · **BibTeX:** `visentin2015_circadian`
    - **Role:** Quantifies intraday insulin-sensitivity shifts across breakfast, lunch, and dinner.

11. **Riddell, M. C., Gallen, I. W., Smart, C. E., et al. (2017).** *Exercise management in type 1 diabetes: a consensus statement.* **The Lancet Diabetes & Endocrinology**, 5(5), 377–390.
    - **DOI:** [10.1016/S2213-8587(17)30014-1](https://doi.org/10.1016/S2213-8587(17)30014-1) · **BibTeX:** `riddell2017_exercise`
    - **Role:** Clinical reference for aerobic/anaerobic exercise effects on glucose uptake and delayed hypoglycemia.

12. **Alkhateeb, H., El Fathi, A., Ghanbari, M., & Haidar, A. (2021).** *Modelling glucose dynamics during moderate exercise in individuals with type 1 diabetes.* **PLOS ONE**, 16(3), e0248280.
    - **DOI:** [10.1371/journal.pone.0248280](https://doi.org/10.1371/journal.pone.0248280) · **BibTeX:** `alkhateeb2021_exercise`

13. **Romeres, D., Schiavon, M., Basu, A., Cobelli, C., Basu, R., & Dalla Man, C. (2021).** *Exercise effect on insulin-dependent and insulin-independent glucose utilization in healthy individuals and individuals with type 1 diabetes: a modeling study.* **AJP-Endocrinology and Metabolism**, 321(1), E122–E129.
    - **DOI:** [10.1152/ajpendo.00084.2021](https://doi.org/10.1152/ajpendo.00084.2021) · **BibTeX:** `romeres2021_exercise`

14. **Cryer, P. E. (2013).** *Mechanisms of Hypoglycemia-Associated Autonomic Failure in Diabetes.* **New England Journal of Medicine**, 369(4), 362–372.
    - **DOI:** [10.1056/NEJMra1215228](https://doi.org/10.1056/NEJMra1215228) · **BibTeX:** `cryer2013_haaf`
    - **Role:** Explains how recurrent hypoglycemia blunts counterregulatory defenses (HAAF).

15. **Heise, T., Stender-Petersen, K., Hövelmann, U., et al. (2017).** *Pharmacokinetic and Pharmacodynamic Properties of Faster-Acting Insulin Aspart versus Insulin Aspart Across a Clinically Relevant Dose Range in Subjects with Type 1 Diabetes Mellitus.* **Clinical Pharmacokinetics**, 56(6), 649–660.
    - **DOI:** [10.1007/s40262-016-0473-5](https://doi.org/10.1007/s40262-016-0473-5) · **BibTeX:** `heise2017_fiasp`

---

## 4. Standardized Glycemic Risk & Evaluation Metrics

16. **Kovatchev, B. P., Cox, D. J., Gonder-Frederick, L. A., & Clarke, W. L. (1997).** *Symmetrization of the Blood Glucose Measurement Scale and Its Applications.* **Diabetes Care**, 20(11), 1655–1658.
    - **DOI:** [10.2337/diacare.20.11.1655](https://doi.org/10.2337/diacare.20.11.1655) · **BibTeX:** `kovatchev1997_risk`

17. **Kovatchev, B. P., Cox, D. J., Gonder-Frederick, L. A., Young-Hyman, D., Schlundt, D., & Clarke, W. L. (1998).** *Assessment of Risk for Severe Hypoglycemia Among Adults With IDDM: Validation of the Low Blood Glucose Index.* **Diabetes Care**, 21(11), 1870–1875.
    - **DOI:** [10.2337/diacare.21.11.1870](https://doi.org/10.2337/diacare.21.11.1870) · **BibTeX:** `kovatchev1998_lbgi`
    - **Role:** Defines LBGI and HBGI, the standard asymmetric risk indices used in both clinical evaluation and RL reward design.

18. **Battelino, T., Danne, T., Bergenstal, R. M., et al. (2019).** *Clinical Targets for Continuous Glucose Monitoring Data Interpretation: Recommendations From the International Consensus on Time in Range.* **Diabetes Care**, 42(8), 1593–1603.
    - **DOI:** [10.2337/dci19-0028](https://doi.org/10.2337/dci19-0028) · **BibTeX:** `battelino2019_tir`
    - **Role:** Establishes standard targets: TIR (70–180 mg/dL) >70%, TBR (<70 mg/dL) <4%, Level 2 TBR (<54 mg/dL) <1%, TAR (>180 mg/dL) <25%, CV ≤36%.

19. **American Diabetes Association Professional Practice Committee (2026).** *6. Glycemic Goals, Hypoglycemia, and Hyperglycemic Crises* & *7. Diabetes Technology: Standards of Care in Diabetes—2026.* **Diabetes Care**, 49(Suppl. 1), S132–S165.
    - **DOIs:** [10.2337/dc26-S006](https://doi.org/10.2337/dc26-S006) · [10.2337/dc26-S007](https://doi.org/10.2337/dc26-S007) · **BibTeX:** `ada2026_goals`, `ada2026_tech`
