# Scientific Evidence & Source Legend

This file documents the scientific and technical sources used to shape IINTS-AF defaults, validation targets, report framing, local AI setup guidance, and best-effort device emulation notes.

Scope:
- Pre-clinical simulation and retrospective forecasting only.
- Not a treatment recommendation engine.
- Not a medical device.

Use `iints sources` to print the packaged source manifest from the installed SDK.

For the full public source index, including dataset registry sources and documentation-only implementation references, see [Complete Source Library](SOURCE_LIBRARY.md).

## How To Read This Legend

There are two source buckets in this project:

1. **Packaged evidence sources**
   These are the core medical and dataset references shipped with the SDK and exposed through `iints sources`.

2. **Documentation-only implementation sources**
   These are the additional references used in the guides for:
   - Ollama setup
   - local open Mistral model selection
   - best-effort device emulation context

That split is intentional:
- the packaged manifest stays focused on repeatable research evidence
- this page stays readable as the full legend for humans

## Category Legend

| Category | Meaning |
|---|---|
| `guideline` | formal standards or standards-of-care style guidance |
| `consensus` | expert consensus used for interpretation targets |
| `trial` | clinical or comparative outcome study |
| `pharmacology` | insulin PK/PD reference |
| `sensor` | CGM behavior, lag, or sensor interpretation source |
| `model` | mathematical or simulator foundation source |
| `dataset` | public dataset provenance source |
| `runtime` | technical runtime or installation reference for local AI |
| `model_card` | model-family reference for local Mistral choices |
| `regulatory` | manufacturer or regulator-facing system reference |
| `technical_manual` | user guide or product documentation |
| `clinical_trial` | named trial registry or trial program page |

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
| Simulator realism/validation framing | Align in-silico model evaluation with accepted simulator literature | `visentin_2018_uvapadova`, `mujahid_2024_generative_t1d_simulator` |
| Exercise stress scenarios | Use consensus guidance for exercise-related glucose behavior | `riddell_2017_exercise_consensus` |
| Forecast training data provenance | Use publicly documented dataset references | `marling_2020_ohiot1dm` |
| AGP-style research reports | Present dense glucose traces with time-in-ranges, modal-day percentile bands, and daily profiles | `idc_2025_agp_report_overview`, `attd_2019_time_in_range` |
| Local AI setup and model selection docs | Keep Ollama and open Mistral setup instructions grounded in official docs | `ollama_linux_install`, `mistral_2025_ministral_3_announcement`, `mistral_2025_ministral_3_3b`, `mistral_2025_ministral_3_8b`, `mistral_2025_ministral_3_14b` |
| Device emulation notes | Provide best-effort references behind 780G / Control-IQ / Omnipod 5 approximations | `bergenstal_2020_780g`, `fda_k193510_780g`, `medtronic_780g_user_guide`, `brown_2019_control_iq_dtt`, `fda_k191289_control_iq`, `idcl_nct03563313`, `control_iq_user_guide`, `assert_omnipod_5`, `onset_omnipod_5`, `fda_k203467_omnipod5`, `omnipod5_user_guide` |

## Packaged Medical And Dataset Sources

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

12. `mujahid_2024_generative_t1d_simulator`
    Mujahid O, Contreras I, Beneyto A, Vehi J. *Generative deep learning for the development of a type 1 diabetes simulator*. Communications Medicine. 2024.
    DOI: [10.1038/s43856-024-00476-0](https://doi.org/10.1038/s43856-024-00476-0)

13. `riddell_2017_exercise_consensus`
    Riddell MC, et al. *Exercise management in type 1 diabetes: a consensus statement*. Lancet Diabetes Endocrinol. 2017.  
    DOI: [10.1016/S2213-8587(17)30014-1](https://doi.org/10.1016/S2213-8587(17)30014-1)

14. `marling_2020_ohiot1dm`
    Marling C, Bunescu R. *The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020*. CEUR Workshop Proceedings.  
    Paper: [ceur-ws.org/Vol-2675/paper2.pdf](http://ceur-ws.org/Vol-2675/paper2.pdf)

15. `idc_2025_agp_report_overview`
    HealthPartners Institute / International Diabetes Center. *Guide to Understanding the Ambulatory Glucose Profile (AGP) Report*. 2025.  
    PDF: [healthpartners.com](https://www.healthpartners.com/institute/wp-content/uploads/2025/05/Ambulatory-Glucose-Profile-Report-Overview.pdf)

## Documentation-Only Local AI Setup Sources

These are the official references used in the guides for installing Ollama, understanding the local Ministral 3 family, and explaining why the SDK recommends different model sizes for different hardware.

1. `ollama_linux_install`  
   Ollama. *Linux Installation Documentation*.  
   URL: [docs.ollama.com/linux](https://docs.ollama.com/linux)

2. `mistral_2025_ministral_3_announcement`  
   Mistral AI. *Introducing Mistral 3*.  
   URL: [mistral.ai/news/mistral-3](https://mistral.ai/news/mistral-3)

3. `mistral_2025_ministral_3_3b`  
   Mistral AI Docs. *Ministral 3 3B*.  
   URL: [docs.mistral.ai/models/ministral-3-3b-25-12](https://docs.mistral.ai/models/ministral-3-3b-25-12)

4. `mistral_2025_ministral_3_8b`  
   Mistral AI Docs. *Ministral 3 8B*.  
   URL: [docs.mistral.ai/models/ministral-3-8b-25-12](https://docs.mistral.ai/models/ministral-3-8b-25-12)

5. `mistral_2025_ministral_3_14b`  
   Mistral AI Docs. *Ministral 3 14B*.  
   URL: [docs.mistral.ai/models/ministral-3-14b-25-12](https://docs.mistral.ai/models/ministral-3-14b-25-12)

6. `mistral_2026_adjustable_reasoning`  
   Mistral AI Docs. *Adjustable Reasoning*.  
   URL: [docs.mistral.ai/studio-api/conversations/reasoning/adjustable](https://docs.mistral.ai/capabilities/reasoning/adjustable)

7. `mistral_2026_small_4`  
   Mistral AI Docs. *Mistral Small 4*.  
   URL: [docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03](https://docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03)

8. `mistral_2026_medium_35`  
   Mistral AI Docs. *Mistral Medium 3.5*.  
   URL: [docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04](https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04)

## Documentation-Only Device Emulation References

These references are used for the best-effort emulator notes and are not claims of exact proprietary algorithm reproduction.

### Medtronic MiniMed 780G

1. `bergenstal_2020_780g`  
   Bergenstal RM, et al. *Safety of a Hybrid Closed-Loop Insulin Delivery System in Patients With Type 1 Diabetes*.  
   DOI: [10.1056/NEJMoa2003479](https://doi.org/10.1056/NEJMoa2003479)

2. `fda_k193510_780g`  
   U.S. FDA. *510(k) K193510 - MiniMed 780G System*.  
   URL: [accessdata.fda.gov](https://www.accessdata.fda.gov/)

3. `medtronic_780g_user_guide`  
   Medtronic Diabetes. *MiniMed 780G User Guide / Product Documentation*.  
   URL: [medtronicdiabetes.com](https://www.medtronicdiabetes.com/)

### Tandem Control-IQ

4. `brown_2019_control_iq_dtt`  
   Brown SA, et al. *Performance of the Tandem t:slim X2 insulin pump with Control-IQ technology in the International Diabetes Closed-Loop trial*.  
   DOI: [10.1089/dia.2019.0226](https://doi.org/10.1089/dia.2019.0226)

5. `fda_k191289_control_iq`  
   U.S. FDA. *510(k) K191289 - Control-IQ System*.  
   URL: [accessdata.fda.gov](https://www.accessdata.fda.gov/)

6. `idcl_nct03563313`  
   ClinicalTrials.gov. *International Diabetes Closed Loop Trial*.  
   URL: [clinicaltrials.gov/ct2/show/NCT03563313](https://clinicaltrials.gov/ct2/show/NCT03563313)

7. `control_iq_user_guide`  
   Tandem Diabetes Care. *Control-IQ User Guide / Product Documentation*.  
   URL: [tandemdiabetes.com](https://www.tandemdiabetes.com/)

### Omnipod 5

8. `assert_omnipod_5`  
   Insulet / Omnipod. *ASSERT Trial - Omnipod 5*.  
   URL: [omnipod.com/assert-trial](https://www.omnipod.com/assert-trial)

9. `onset_omnipod_5`  
   Insulet / Omnipod. *ONSET Trial - Omnipod 5 in Type 2 Diabetes*.  
   URL: [omnipod.com/onset-trial](https://www.omnipod.com/onset-trial)

10. `fda_k203467_omnipod5`  
    U.S. FDA. *510(k) K203467 - Omnipod 5 System*.  
    URL: [accessdata.fda.gov](https://www.accessdata.fda.gov/)

11. `omnipod5_user_guide`  
    Insulet / Omnipod. *Omnipod 5 User Guide / Product Documentation*.  
    URL: [omnipod.com](https://www.omnipod.com/)

## Reproducibility Note

For report reproducibility, pair this file with:
- run metadata (`run_metadata.json`)
- manifest hashes (`run_manifest.json`)
- dataset lineage fields in research outputs (`source_file_sha256`, split metadata).

If you want the machine-readable packaged subset, export it with:

```bash
iints sources --output-json results/source_manifest.json
```
