# Scientific Evidence & Sources

This file documents the scientific sources used to shape IINTS-AF defaults, validation targets, and report framing.

Scope:
- Pre-clinical simulation and retrospective forecasting only.
- Not a treatment recommendation engine.
- Not a medical device.

Use `iints sources` to print the same source manifest from the packaged SDK.

## How Sources Map to SDK Components

| SDK area | Why it exists | Source IDs |
|---|---|---|
| Validation targets (`TIR`, `TBR`, `TAR`, hypoglycemia framing) | Keep benchmark interpretation aligned with accepted diabetes metrics | `ada_2026_glycemic_goals`, `attd_2019_time_in_range` |
| CGM + AID context in docs/reports | Keep language aligned with current standards for technology use | `ada_2026_diabetes_technology` |
| AID benchmark envelopes | Compare algorithm behavior against realistic closed-loop outcomes | `nejm_2019_control_iq`, `adapt_2022_ahcl` |
| Meal timing / pre-bolus scenarios | Avoid unrealistic meal-response assumptions | `cobry_2010_meal_bolus_timing` |
| Insulin action profiles (rapid and ultra-rapid) | Parameterize onset/peak assumptions from pharmacology literature | `heise_2017_fiasp_pkpd`, `klaff_2020_urli_pkpd` |
| Input validation and CGM lag rationale | Keep signal handling biologically plausible | `wentholt_2004_cgm_lag` |
| Virtual patient meal dynamics | Ground meal disturbance dynamics in established models | `dalla_man_2007_meal_model` |
| Simulator realism/validation framing | Align in-silico model evaluation with accepted simulator literature | `visentin_2018_uvapadova` |
| Exercise stress scenarios | Use consensus guidance for exercise-related glucose behavior | `riddell_2017_exercise_consensus` |
| Forecast training data provenance | Use publicly documented dataset references | `marling_2020_ohiot1dm` |

## Source List

1. `ada_2026_glycemic_goals`  
   ADA Professional Practice Committee. *Glycemic Goals and Hypoglycemia: Standards of Care in Diabetes—2026*.  
   DOI: [10.2337/dc26-S006](https://doi.org/10.2337/dc26-S006)

2. `ada_2026_diabetes_technology`  
   ADA Professional Practice Committee. *Diabetes Technology: Standards of Care in Diabetes—2026*.  
   DOI: [10.2337/dc26-S007](https://doi.org/10.2337/dc26-S007)

3. `attd_2019_time_in_range`  
   Battelino T, Danne T, Bergenstal RM, et al. *Clinical Targets for Continuous Glucose Monitoring Data Interpretation*. Diabetes Care. 2019.  
   DOI: [10.2337/dci19-0028](https://doi.org/10.2337/dci19-0028)

4. `nejm_2019_control_iq`  
   Brown SA, et al. *Six-Month Randomized, Multicenter Trial of Closed-Loop Control in Type 1 Diabetes*. N Engl J Med. 2019.  
   DOI: [10.1056/NEJMoa1907863](https://doi.org/10.1056/NEJMoa1907863)

5. `adapt_2022_ahcl`  
   Benhamou PY, et al. *Advanced hybrid closed loop therapy versus conventional treatment in adults with type 1 diabetes (ADAPT)*. Lancet Diabetes Endocrinol. 2022.  
   DOI: [10.1016/S2213-8587(22)00212-1](https://doi.org/10.1016/S2213-8587(22)00212-1)

6. `cobry_2010_meal_bolus_timing`  
   Cobry E, et al. *Timing of Meal Insulin Boluses to Achieve Optimal Postprandial Glycemic Control*. J Diabetes Sci Technol. 2010.  
   DOI: [10.1177/193229681000400404](https://doi.org/10.1177/193229681000400404)

7. `heise_2017_fiasp_pkpd`  
   Heise T, et al. *A Faster-Onset Formulation of Insulin Aspart*. Clin Pharmacokinet. 2017.  
   DOI: [10.1007/s40262-017-0510-8](https://doi.org/10.1007/s40262-017-0510-8)

8. `klaff_2020_urli_pkpd`  
   Klaff LJ, et al. *Ultra Rapid Lispro Demonstrates Accelerated Pharmacokinetics and Pharmacodynamics*. Diabetes Obes Metab. 2020.  
   DOI: [10.1111/dom.14049](https://doi.org/10.1111/dom.14049)

9. `wentholt_2004_cgm_lag`  
   Wentholt IME, et al. *How glucose sensors can facilitate therapy in diabetes management*. Diabetes Technol Ther. 2004.  
   DOI: [10.1089/dia.2004.6.615](https://doi.org/10.1089/dia.2004.6.615)

10. `dalla_man_2007_meal_model`  
    Dalla Man C, Rizza RA, Cobelli C. *Meal simulation model of the glucose-insulin system*. IEEE Trans Biomed Eng. 2007.  
    DOI: [10.1109/TBME.2007.893506](https://doi.org/10.1109/TBME.2007.893506)

11. `visentin_2018_uvapadova`  
    Visentin R, et al. *The University of Virginia/Padova Type 1 Diabetes Simulator Matches the 2014 DMMS.R*. J Diabetes Sci Technol. 2018.  
    DOI: [10.1177/1932296818757747](https://doi.org/10.1177/1932296818757747)

12. `riddell_2017_exercise_consensus`  
    Riddell MC, et al. *Exercise management in type 1 diabetes: a consensus statement*. Lancet Diabetes Endocrinol. 2017.  
    DOI: [10.1016/S2213-8587(17)30014-1](https://doi.org/10.1016/S2213-8587(17)30014-1)

13. `marling_2020_ohiot1dm`  
    Marling C, Bunescu R. *The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020*. CEUR Workshop Proceedings.  
    Paper: [ceur-ws.org/Vol-2675/paper2.pdf](http://ceur-ws.org/Vol-2675/paper2.pdf)

## Reproducibility Note

For report reproducibility, pair this file with:
- run metadata (`run_metadata.json`)
- manifest hashes (`run_manifest.json`)
- dataset lineage fields in research outputs (`source_file_sha256`, split metadata).
