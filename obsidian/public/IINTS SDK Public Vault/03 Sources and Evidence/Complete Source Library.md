---
tags:
  - iints/sources
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Complete Source Library

This is the curated source list currently recorded in the SDK manifests and documentation. It is split into three practical buckets:

- **Packaged evidence**: shipped with the SDK and exposed through `iints sources`.
- **Dataset registry**: datasets the SDK can reference, import, fetch, or use for realism/training workflows.
- **Documentation-only references**: AI setup, device-emulation context, Pico pump safety, and hardware documentation.

> [!note] Scope
> This table is a traceability map for SDK users. It says what evidence is referenced by the project; it does not claim clinical validation or medical-device approval.

| Bucket | ID | Category | SDK component | Title | DOI / URL | Source note |
| --- | --- | --- | --- | --- | --- | --- |
| packaged_evidence | ada_2026_glycemic_goals | guideline | validation_targets | Glycemic Goals and Hypoglycemia: Standards of Care in Diabetes—2026 | 10.2337/dc26-S006 | [[ada_2026_glycemic_goals]] |
| packaged_evidence | ada_2026_diabetes_technology | guideline | cgm_and_aid_context | Diabetes Technology: Standards of Care in Diabetes—2026 | 10.2337/dc26-S007 | [[ada_2026_diabetes_technology]] |
| packaged_evidence | attd_2019_time_in_range | consensus | metrics_targets | Clinical Targets for Continuous Glucose Monitoring Data Interpretation | 10.2337/dci19-0028 | [[attd_2019_time_in_range]] |
| packaged_evidence | nejm_2019_control_iq | trial | aid_benchmarking | Six-Month Randomized, Multicenter Trial of Closed-Loop Control in Type 1 Diabetes | 10.1056/NEJMoa1907863 | [[nejm_2019_control_iq]] |
| packaged_evidence | adapt_2022_ahcl | trial | aid_benchmarking | Advanced hybrid closed loop therapy versus conventional treatment in adults with type 1 diabetes (ADAPT) | 10.1016/S2213-8587(22)00212-1 | [[adapt_2022_ahcl]] |
| packaged_evidence | cobry_2010_meal_bolus_timing | trial | prebolus_timing | Timing of Meal Insulin Boluses to Achieve Optimal Postprandial Glycemic Control | 10.1177/193229681000400404 | [[cobry_2010_meal_bolus_timing]] |
| packaged_evidence | heise_2017_fiasp_pkpd | pharmacology | insulin_action_profiles | A Faster-Onset Formulation of Insulin Aspart | 10.1007/s40262-017-0510-8 | [[heise_2017_fiasp_pkpd]] |
| packaged_evidence | klaff_2020_urli_pkpd | pharmacology | insulin_action_profiles | Ultra Rapid Lispro Demonstrates Accelerated Pharmacokinetics and Pharmacodynamics | 10.1111/dom.14049 | [[klaff_2020_urli_pkpd]] |
| packaged_evidence | wentholt_2004_cgm_lag | sensor | cgm_lag_and_validator | How glucose sensors can facilitate therapy in diabetes management | 10.1089/dia.2004.6.615 | [[wentholt_2004_cgm_lag]] |
| packaged_evidence | dalla_man_2007_meal_model | model | virtual_patient_dynamics | Meal simulation model of the glucose-insulin system | 10.1109/TBME.2007.893506 | [[dalla_man_2007_meal_model]] |
| packaged_evidence | visentin_2018_uvapadova | model | simulation_validation | The University of Virginia/Padova Type 1 Diabetes Simulator Matches the 2014 DMMS.R | 10.1177/1932296818757747 | [[visentin_2018_uvapadova]] |
| packaged_evidence | mujahid_2024_generative_t1d_simulator | model | simulation_realism | Generative deep learning for the development of a type 1 diabetes simulator | 10.1038/s43856-024-00476-0 | [[mujahid_2024_generative_t1d_simulator]] |
| packaged_evidence | riddell_2017_exercise_consensus | consensus | exercise_stress_scenarios | Exercise management in type 1 diabetes: a consensus statement | 10.1016/S2213-8587(17)30014-1 | [[riddell_2017_exercise_consensus]] |
| packaged_evidence | marling_2020_ohiot1dm | dataset | predictor_training_data | OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020 | http://ceur-ws.org/Vol-2675/paper2.pdf | [[marling_2020_ohiot1dm]] |
| dataset_registry | sample | dataset | data_import_realism_training | IINTS Sample CGM (Bundled) |  | [[sample]] |
| dataset_registry | ohio_t1dm | dataset | data_import_realism_training | OhioT1DM Dataset | https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html | [[ohio_t1dm]] |
| dataset_registry | diatrend | dataset | data_import_realism_training | DiaTrend Dataset | https://doi.org/10.7303/syn38187184 | [[diatrend]] |
| dataset_registry | t1d_uom | dataset | data_import_realism_training | T1D-UOM Longitudinal Multimodal Dataset | 10.5281/zenodo.15806142 | [[t1d_uom]] |
| dataset_registry | t1d_granada | dataset | data_import_realism_training | T1DiabetesGranada Dataset | 10.5281/zenodo.10050944 | [[t1d_granada]] |
| dataset_registry | aide_t1d | dataset | data_import_realism_training | AIDE T1D Public Dataset | https://public.jaeb.org/datasets/ | [[aide_t1d]] |
| dataset_registry | pedap | dataset | data_import_realism_training | PEDAP Public Dataset | https://public.jaeb.org/datasets/ | [[pedap]] |
| dataset_registry | azt1d | dataset | data_import_realism_training | AZT1D: A Real-World Dataset for Type 1 Diabetes | 10.17632/gk9m674wcx.1 | [[azt1d]] |
| dataset_registry | hupa_ucm | dataset | data_import_realism_training | HUPA-UCM Diabetes Dataset | 10.17632/3hbcscwz44.1 | [[hupa_ucm]] |
| dataset_registry | openaps_data_commons | dataset | data_import_realism_training | OpenAPS Data Commons | https://openaps.org/outcomes/data-commons/ | [[openaps_data_commons]] |
| dataset_registry | tidepool_bigdata | dataset | data_import_realism_training | Tidepool Big Data Donation | https://www.tidepool.org/bigdata | [[tidepool_bigdata]] |
| dataset_registry | niddk_central | dataset | data_import_realism_training | NIDDK Central Repository | https://repository.niddk.nih.gov/ | [[niddk_central]] |
| dataset_registry | t1d_exchange | dataset | data_import_realism_training | T1D Exchange Clinic Registry | https://datacatalog.med.nyu.edu/dataset/10129 | [[t1d_exchange]] |
| docs_only_local_ai | ollama_linux_install | runtime | local_ai_setup | Ollama Linux Installation Documentation | https://docs.ollama.com/linux | [[ollama_linux_install]] |
| docs_only_local_ai | mistral_2025_ministral_3_announcement | model_card | local_ai_model_selection | Mistral AI: Introducing Mistral 3 | https://mistral.ai/news/mistral-3 | [[mistral_2025_ministral_3_announcement]] |
| docs_only_local_ai | mistral_2025_ministral_3_3b | model_card | local_ai_model_selection | Mistral AI Docs: Ministral 3 3B | https://docs.mistral.ai/models/ministral-3-3b-25-12 | [[mistral_2025_ministral_3_3b]] |
| docs_only_local_ai | mistral_2025_ministral_3_8b | model_card | local_ai_model_selection | Mistral AI Docs: Ministral 3 8B | https://docs.mistral.ai/models/ministral-3-8b-25-12 | [[mistral_2025_ministral_3_8b]] |
| docs_only_local_ai | mistral_2025_ministral_3_14b | model_card | local_ai_model_selection | Mistral AI Docs: Ministral 3 14B | https://docs.mistral.ai/models/ministral-3-14b-25-12 | [[mistral_2025_ministral_3_14b]] |
| docs_only_device_emulation | bergenstal_2020_780g | trial | medtronic_780g_emulation_context | Safety of a Hybrid Closed-Loop Insulin Delivery System in Patients With Type 1 Diabetes | 10.1056/NEJMoa2003479 | [[bergenstal_2020_780g]] |
| docs_only_device_emulation | fda_k193510_780g | regulatory | medtronic_780g_emulation_context | U.S. FDA 510(k) K193510 - MiniMed 780G System | https://www.accessdata.fda.gov/ | [[fda_k193510_780g]] |
| docs_only_device_emulation | medtronic_780g_user_guide | technical_manual | medtronic_780g_emulation_context | Medtronic MiniMed 780G User Guide / Product Documentation | https://www.medtronicdiabetes.com/ | [[medtronic_780g_user_guide]] |
| docs_only_device_emulation | brown_2019_control_iq_dtt | trial | tandem_control_iq_emulation_context | Performance of Tandem t:slim X2 with Control-IQ technology in the IDCL trial | 10.1089/dia.2019.0226 | [[brown_2019_control_iq_dtt]] |
| docs_only_device_emulation | fda_k191289_control_iq | regulatory | tandem_control_iq_emulation_context | U.S. FDA 510(k) K191289 - Control-IQ System | https://www.accessdata.fda.gov/ | [[fda_k191289_control_iq]] |
| docs_only_device_emulation | idcl_nct03563313 | clinical_trial | tandem_control_iq_emulation_context | ClinicalTrials.gov: International Diabetes Closed Loop Trial | https://clinicaltrials.gov/ct2/show/NCT03563313 | [[idcl_nct03563313]] |
| docs_only_device_emulation | control_iq_user_guide | technical_manual | tandem_control_iq_emulation_context | Tandem Control-IQ User Guide / Product Documentation | https://www.tandemdiabetes.com/ | [[control_iq_user_guide]] |
| docs_only_device_emulation | assert_omnipod_5 | clinical_trial | omnipod_5_emulation_context | Insulet / Omnipod ASSERT Trial - Omnipod 5 | https://www.omnipod.com/assert-trial | [[assert_omnipod_5]] |
| docs_only_device_emulation | onset_omnipod_5 | clinical_trial | omnipod_5_emulation_context | Insulet / Omnipod ONSET Trial - Omnipod 5 in Type 2 Diabetes | https://www.omnipod.com/onset-trial | [[onset_omnipod_5]] |
| docs_only_device_emulation | fda_k203467_omnipod5 | regulatory | omnipod_5_emulation_context | U.S. FDA 510(k) K203467 - Omnipod 5 System | https://www.accessdata.fda.gov/ | [[fda_k203467_omnipod5]] |
| docs_only_device_emulation | omnipod5_user_guide | technical_manual | omnipod_5_emulation_context | Insulet / Omnipod 5 User Guide / Product Documentation | https://www.omnipod.com/ | [[omnipod5_user_guide]] |
| docs_only_physiology | bergman_1979_minimal_model | model | physiology_reference | Quantitative estimation of insulin sensitivity | 10.1152/ajpendo.1979.236.6.E667 | [[bergman_1979_minimal_model]] |
| docs_only_hardware_safety | fda_infusion_pump_software_safety | regulatory | pico_pump_lab_safety | FDA Infusion Pump Software Safety Research | https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-software-safety-research-fda | [[fda_infusion_pump_software_safety]] |
| docs_only_hardware_safety | fda_infusion_pumps | regulatory | pico_pump_lab_safety | FDA Infusion Pumps | https://www.fda.gov/medical-devices/general-hospital-devices-and-supplies/infusion-pumps | [[fda_infusion_pumps]] |
| docs_only_hardware_safety | raspberry_pi_pico_docs | technical_manual | pico_pump_lab_firmware | Raspberry Pi Pico-series microcontrollers documentation | https://www.raspberrypi.com/documentation/microcontrollers/raspberry-pi-pico.html | [[raspberry_pi_pico_docs]] |

## Machine-Readable Files

- [[Source Library.csv]]
- [[Discovered URL and DOI Index.csv]]
