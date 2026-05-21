---
tags:
  - iints/sources
  - iints/physiology
  - iints/safety
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Physiology and Safety Sources

> [!important] User framing
> These sources explain the physiological ranges and safety language in the SDK. They support simulation and review, not real-world treatment decisions.

| ID | Category | SDK component | Title | DOI / URL | Note |
| --- | --- | --- | --- | --- | --- |
| ada_2026_glycemic_goals | guideline | validation_targets | Glycemic Goals and Hypoglycemia: Standards of Care in Diabetes—2026 | 10.2337/dc26-S006 | [[ada_2026_glycemic_goals]] |
| ada_2026_diabetes_technology | guideline | cgm_and_aid_context | Diabetes Technology: Standards of Care in Diabetes—2026 | 10.2337/dc26-S007 | [[ada_2026_diabetes_technology]] |
| attd_2019_time_in_range | consensus | metrics_targets | Clinical Targets for Continuous Glucose Monitoring Data Interpretation | 10.2337/dci19-0028 | [[attd_2019_time_in_range]] |
| cobry_2010_meal_bolus_timing | trial | prebolus_timing | Timing of Meal Insulin Boluses to Achieve Optimal Postprandial Glycemic Control | 10.1177/193229681000400404 | [[cobry_2010_meal_bolus_timing]] |
| heise_2017_fiasp_pkpd | pharmacology | insulin_action_profiles | A Faster-Onset Formulation of Insulin Aspart | 10.1007/s40262-017-0510-8 | [[heise_2017_fiasp_pkpd]] |
| klaff_2020_urli_pkpd | pharmacology | insulin_action_profiles | Ultra Rapid Lispro Demonstrates Accelerated Pharmacokinetics and Pharmacodynamics | 10.1111/dom.14049 | [[klaff_2020_urli_pkpd]] |
| wentholt_2004_cgm_lag | sensor | cgm_lag_and_validator | How glucose sensors can facilitate therapy in diabetes management | 10.1089/dia.2004.6.615 | [[wentholt_2004_cgm_lag]] |
| dalla_man_2007_meal_model | model | virtual_patient_dynamics | Meal simulation model of the glucose-insulin system | 10.1109/TBME.2007.893506 | [[dalla_man_2007_meal_model]] |
| visentin_2018_uvapadova | model | simulation_validation | The University of Virginia/Padova Type 1 Diabetes Simulator Matches the 2014 DMMS.R | 10.1177/1932296818757747 | [[visentin_2018_uvapadova]] |
| mujahid_2024_generative_t1d_simulator | model | simulation_realism | Generative deep learning for the development of a type 1 diabetes simulator | 10.1038/s43856-024-00476-0 | [[mujahid_2024_generative_t1d_simulator]] |
| riddell_2017_exercise_consensus | consensus | exercise_stress_scenarios | Exercise management in type 1 diabetes: a consensus statement | 10.1016/S2213-8587(17)30014-1 | [[riddell_2017_exercise_consensus]] |
| bergman_1979_minimal_model | model | physiology_reference | Quantitative estimation of insulin sensitivity | 10.1152/ajpendo.1979.236.6.E667 | [[bergman_1979_minimal_model]] |
| fda_infusion_pump_software_safety | regulatory | pico_pump_lab_safety | FDA Infusion Pump Software Safety Research | https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-software-safety-research-fda | [[fda_infusion_pump_software_safety]] |
| fda_infusion_pumps | regulatory | pico_pump_lab_safety | FDA Infusion Pumps | https://www.fda.gov/medical-devices/general-hospital-devices-and-supplies/infusion-pumps | [[fda_infusion_pumps]] |

## Numbers Users Should Recognize

| Concept | Typical SDK meaning | Where to look |
| --- | --- | --- |
| `TIR 70-180 mg/dL` | Time in target glucose range | [[ada_2026_glycemic_goals]], [[attd_2019_time_in_range]] |
| `TBR <70 mg/dL` | Hypoglycemia burden flag | [[ada_2026_glycemic_goals]], [[attd_2019_time_in_range]] |
| `ISF mg/dL/U` | Insulin sensitivity assumption in profiles | [[bergman_1979_minimal_model]] plus SDK profile docs |
| meal rise and lag | Meal-response realism checks | [[dalla_man_2007_meal_model]], [[cobry_2010_meal_bolus_timing]] |
| CGM lag | Sensor delay / signal interpretation | [[wentholt_2004_cgm_lag]] |
| bench-only pump boundary | Why upload tools must not drive real insulin | [[fda_infusion_pump_software_safety]], [[fda_infusion_pumps]] |
