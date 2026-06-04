# Complete Source Library

This page is the public source index for IINTS-AF. It collects the references used to shape SDK defaults, physiology assumptions, safety framing, dataset guidance, local AI workflows, and documentation.

!!! warning "Research-only boundary"
    These sources support transparent simulation and educational AI research. They are not a claim that IINTS-AF is clinically validated or certified for treatment decisions.

## How To Use This Page

- Use this page when a reviewer asks: "Where did these assumptions come from?"
- Use [Evidence Base](EVIDENCE_BASE.md) for the shorter explanation of how sources map to SDK components.
- Use [Diabetes Research Datasets](DIABETES_RESEARCH_DATASETS.md) for dataset acquisition and local AI planning.
- Use `iints sources --output-json results/source_manifest.json` to export the packaged evidence subset from an installed SDK.
- Use `iints data research-plan --output-dir data_packs/research_dataset_plan` to export the dataset-source plan.

## Packaged Scientific Sources

These are the machine-readable evidence sources shipped in `src/iints/presets/evidence_sources.yaml` and exposed through `iints sources`.

| ID | Category | SDK component | Title | Link | Why it matters |
| --- | --- | --- | --- | --- | --- |
| `ada_2026_glycemic_goals` | guideline | validation_targets | Glycemic Goals and Hypoglycemia: Standards of Care in Diabetes—2026 | [DOI 10.2337/dc26-S006](https://doi.org/10.2337/dc26-S006) | Reference for TIR/TBR/TAR target framing and hypoglycemia thresholds in validation profiles. |
| `ada_2026_diabetes_technology` | guideline | cgm_and_aid_context | Diabetes Technology: Standards of Care in Diabetes—2026 | [DOI 10.2337/dc26-S007](https://doi.org/10.2337/dc26-S007) | Reference for CGM and AID operational context used in report language and safeguards. |
| `attd_2019_time_in_range` | consensus | metrics_targets | Clinical Targets for Continuous Glucose Monitoring Data Interpretation | [DOI 10.2337/dci19-0028](https://doi.org/10.2337/dci19-0028) | Basis for TIR/TBR/TAR interpretation in analytics and scorecards. |
| `nejm_2019_control_iq` | trial | aid_benchmarking | Six-Month Randomized, Multicenter Trial of Closed-Loop Control in Type 1 Diabetes | [DOI 10.1056/NEJMoa1907863](https://doi.org/10.1056/NEJMoa1907863) | Reference envelope for realistic AID performance ranges in benchmark narratives. |
| `adapt_2022_ahcl` | trial | aid_benchmarking | Advanced hybrid closed loop therapy versus conventional treatment in adults with type 1 diabetes (ADAPT) | [DOI 10.1016/S2213-8587(22)00212-1](https://doi.org/10.1016/S2213-8587(22)00212-1) | Adds modern AHCL outcomes for realistic comparator ranges. |
| `cobry_2010_meal_bolus_timing` | trial | prebolus_timing | Timing of Meal Insulin Boluses to Achieve Optimal Postprandial Glycemic Control | [DOI 10.1177/193229681000400404](https://doi.org/10.1177/193229681000400404) | Reference for pre-bolus timing defaults and scenario annotations. |
| `heise_2017_fiasp_pkpd` | pharmacology | insulin_action_profiles | A Faster-Onset Formulation of Insulin Aspart | [DOI 10.1007/s40262-017-0510-8](https://doi.org/10.1007/s40262-017-0510-8) | Reference for rapid insulin onset assumptions in therapy profiles. |
| `klaff_2020_urli_pkpd` | pharmacology | insulin_action_profiles | Ultra Rapid Lispro Demonstrates Accelerated Pharmacokinetics and Pharmacodynamics | [DOI 10.1111/dom.14049](https://doi.org/10.1111/dom.14049) | Reference for ultra-rapid insulin variants and action-curve configuration. |
| `wentholt_2004_cgm_lag` | sensor | cgm_lag_and_validator | How glucose sensors can facilitate therapy in diabetes management | [DOI 10.1089/dia.2004.6.615](https://doi.org/10.1089/dia.2004.6.615) | Background reference for physiologic CGM lag and fail-soft validator behavior. |
| `dalla_man_2007_meal_model` | model | virtual_patient_dynamics | Meal simulation model of the glucose-insulin system | [DOI 10.1109/TBME.2007.893506](https://doi.org/10.1109/TBME.2007.893506) | Reference foundation for physiologic meal disturbance modeling. |
| `mandelbrot_1968_fractional_brownian` | model | cgm_noise_model | Fractional Brownian Motions, Fractional Noises and Applications | [DOI 10.1137/1010093](https://doi.org/10.1137/1010093) | Mathematical anchor for long-memory, fractional-noise-style CGM noise approximations. |
| `campbell_1985_dawn_phenomenon` | physiology | circadian_egp_model | Pathogenesis of the Dawn Phenomenon in Patients with Insulin-Dependent Diabetes Mellitus | [DOI 10.1056/NEJM198506063122302](https://doi.org/10.1056/NEJM198506063122302) | Reference for dawn-phenomenon physiology involving nocturnal growth hormone surges, glucose production, and impaired utilization. |
| `richter_2013_glut4_exercise` | physiology | exercise_glut4_model | Exercise, GLUT4, and skeletal muscle glucose uptake | [DOI 10.1152/physrev.00038.2012](https://doi.org/10.1152/physrev.00038.2012) | Reference for exercise/contraction-stimulated GLUT4 translocation and insulin-independent skeletal-muscle glucose uptake. |
| `naslund_1999_glp1_gastric_emptying` | physiology | glp1_gastric_emptying_model | GLP-1 slows solid gastric emptying and inhibits insulin, glucagon, and PYY release in humans | [DOI 10.1152/ajpregu.1999.277.3.R910](https://doi.org/10.1152/ajpregu.1999.277.3.R910) | Reference for GLP-1-associated slowing of gastric emptying used to document nonlinear meal absorption feedback. |
| `bergman_1979_minimal_model` | model | virtual_patient_dynamics | Quantitative estimation of insulin sensitivity | [DOI 10.1152/ajpendo.1979.236.6.E667](https://doi.org/10.1152/ajpendo.1979.236.6.E667) | Foundation for the Bergman-style minimal-model insulin sensitivity and remote insulin-action states. |
| `hovorka_2004_nmpc_t1d` | model | hovorka_patient_model | Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes | [DOI 10.1088/0967-3334/25/4/010](https://doi.org/10.1088/0967-3334/25/4/010) | Reference for the Hovorka-style compartmental glucose, insulin, and insulin-action structure used in research-mode physiology. |
| `gerich_1979_counterregulation` | physiology | endogenous_counterregulation | Hormonal mechanisms of recovery from insulin-induced hypoglycemia in man | [DOI 10.1152/ajpendo.1979.236.4.E380](https://doi.org/10.1152/ajpendo.1979.236.4.E380) | Primary human physiology reference for glucagon and epinephrine roles in acute recovery from insulin-induced hypoglycemia. |
| `cryer_2013_haaf_mechanisms` | review | haaf_memory_model | Mechanisms of Hypoglycemia-Associated Autonomic Failure in Diabetes | [DOI 10.1056/NEJMra1215228](https://doi.org/10.1056/NEJMra1215228) | Reference for HAAF as a recurrent-hypoglycemia memory mechanism that blunts counterregulation and symptom awareness. |
| `cryer_2013_haaf_diabetes` | review | haaf_memory_model | Hypoglycemia-associated autonomic failure in diabetes | [DOI 10.1016/B978-0-444-53491-0.00023-7](https://doi.org/10.1016/B978-0-444-53491-0.00023-7) | Reference for antecedent hypoglycemia, sleep, and exercise lowering later autonomic and symptomatic responses. |
| `lv_2013_exogenous_glucagon_pk` | pharmacology | exogenous_glucagon_pkpd | Pharmacokinetics modeling of exogenous glucagon in type 1 diabetes mellitus patients | [DOI 10.1089/dia.2013.0150](https://doi.org/10.1089/dia.2013.0150) | Foundation for dual-hormone glucagon depot-to-plasma transport models in research simulations. |
| `haidar_2013_dual_hormone_ap` | trial | dual_hormone_context | Glucose-responsive insulin and glucagon delivery in adults with type 1 diabetes | [DOI 10.1503/cmaj.121265](https://doi.org/10.1503/cmaj.121265) | Clinical research context for dual-hormone insulin-plus-glucagon closed-loop systems. |
| `haidar_2013_insulin_glucagon_pk` | pharmacology | exogenous_glucagon_pkpd | Pharmacokinetics of Insulin Aspart and Glucagon in Type 1 Diabetes during Closed-Loop Operation | [DOI 10.1177/193229681300700610](https://doi.org/10.1177/193229681300700610) | Reference for subcutaneous insulin and glucagon pharmacokinetics during closed-loop operation. |
| `hummel_2018_renal_glucose_handling` | physiology | renal_glucose_clearance | Physiology of renal glucose handling via SGLT1, SGLT2 and GLUT2 | [DOI 10.1007/s00125-018-4656-5](https://doi.org/10.1007/s00125-018-4656-5) | Reference for renal glucose reabsorption physiology and transporter context behind smooth renal-threshold modeling. |
| `defronzo_2013_renal_reabsorption_splay` | physiology | renal_glucose_clearance | Characterization of Renal Glucose Reabsorption in Response to Dapagliflozin | [DOI 10.2337/dc13-0387](https://doi.org/10.2337/dc13-0387) | Reference for threshold, transport maximum, and splay concepts used when documenting renal glucose clearance. |
| `visentin_2018_uvapadova` | model | simulation_validation | The University of Virginia/Padova Type 1 Diabetes Simulator Matches the 2014 DMMS.R | [DOI 10.1177/1932296818757747](https://doi.org/10.1177/1932296818757747) | Reference for simulator validation methodology and in-silico credibility. |
| `mujahid_2024_generative_t1d_simulator` | model | simulation_realism | Generative deep learning for the development of a type 1 diabetes simulator | [DOI 10.1038/s43856-024-00476-0](https://doi.org/10.1038/s43856-024-00476-0) | Reference for modern data-driven T1D simulator realism framing and the limits of purely physiological approximations. |
| `riddell_2017_exercise_consensus` | consensus | exercise_stress_scenarios | Exercise management in type 1 diabetes: a consensus statement | [DOI 10.1016/S2213-8587(17)30014-1](https://doi.org/10.1016/S2213-8587(17)30014-1) | Reference for exercise stress scenario defaults and safety interpretation. |
| `marling_2020_ohiot1dm` | dataset | predictor_training_data | OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020 | [source](http://ceur-ws.org/Vol-2675/paper2.pdf) | Primary public dataset used for subject-level predictor training and evaluation. |
| `idc_2025_agp_report_overview` | consensus | agp_style_reporting | Guide to Understanding the Ambulatory Glucose Profile (AGP) Report | [source](https://www.healthpartners.com/institute/wp-content/uploads/2025/05/Ambulatory-Glucose-Profile-Report-Overview.pdf) | Reference for AGP-style report layout: time-in-ranges, glucose metrics, modal-day percentile profile, and daily glucose profiles. |

## Dataset Sources

These are the dataset references registered in `src/iints/data/datasets.json`. Some are bundled or public-download sources; others require manual download, approval, or a data-use agreement. The SDK records them for provenance but does not bypass source access rules.

| Dataset ID | Access | Source | Name | Link | Citation / use |
| --- | --- | --- | --- | --- | --- |
| `sample` | bundled | IINTS-AF | IINTS Sample CGM (Bundled) | n/a | Bundled full-day demo trace with meal and bolus annotations for quickstarts, reporting, and offline demos. |
| `ohio_t1dm` | request | UNC Charlotte Smart Health Lab | OhioT1DM Dataset | [source](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html) | Eight weeks each for 12 people with type 1 diabetes; includes CGM, fingersticks, basal/bolus insulin, meals, exercise, sleep, work, stress, illness, and fitness-band physiology. |
| `diatrend` | request | Dartmouth College / Scientific Data | DiaTrend Dataset | [source](https://doi.org/10.7303/syn38187184) | Controlled-access CGM and insulin pump dataset from 54 people with type 1 diabetes, with carb logs, bolus data, and selected pump settings. |
| `t1d_uom` | manual | University of Manchester / Zenodo | T1D-UOM Longitudinal Multimodal Dataset | [source](https://doi.org/10.5281/zenodo.15806142) | Twelve-week multimodal dataset from 17 people with type 1 diabetes, including CGM, basal and bolus insulin, meal macronutrients, physical activity, and sleep. |
| `t1d_granada` | request | University of Granada / Zenodo | T1DiabetesGranada Dataset | [source](https://doi.org/10.5281/zenodo.10050944) | Longitudinal flash glucose, demographic, diagnostic, and biochemical dataset spanning four years and 736 people with type 1 diabetes. |
| `aide_t1d` | public-download | Jaeb Center for Health Research | AIDE T1D Public Dataset | [source](https://public.jaeb.org/datasets/) | Automated Insulin Delivery in Elderly with Type 1 Diabetes (AIDE T1D) public dataset. |
| `pedap` | public-download | Jaeb Center for Health Research | PEDAP Public Dataset | [source](https://public.jaeb.org/datasets/) | Pediatric Artificial Pancreas (PEDAP) public dataset (Release 5). |
| `azt1d` | manual | Mendeley Data | AZT1D: A Real-World Dataset for Type 1 Diabetes | [source](https://data.mendeley.com/datasets/gk9m674wcx/1) | Real-world T1D dataset (CGM + insulin + meals) from 25 individuals on AID systems. |
| `hupa_ucm` | manual | Mendeley Data | HUPA-UCM Diabetes Dataset | [source](https://data.mendeley.com/datasets/3hbcscwz44/1) | Free-living T1D dataset with CGM, insulin, meals, and activity data. |
| `openaps_data_commons` | request | OpenAPS | OpenAPS Data Commons | [source](https://openaps.org/outcomes/data-commons/) | Community-contributed open APS data commons. |
| `tidepool_bigdata` | request | Tidepool | Tidepool Big Data Donation | [source](https://www.tidepool.org/bigdata) | Large-scale real-world diabetes data donation program. |
| `niddk_central` | request | NIDDK | NIDDK Central Repository | [source](https://repository.niddk.nih.gov/) | NIH/NIDDK centralized repository of clinical diabetes studies. |
| `t1d_exchange` | request | Jaeb Center / T1D Exchange | T1D Exchange Clinic Registry | [source](https://datacatalog.med.nyu.edu/dataset/10129) | Large clinical registry for type 1 diabetes research. |
| `d1namo` | manual | Zenodo / AISLab HES-SO | D1NAMO Open Dataset | [source](https://zenodo.org/records/1421616) | Exploratory multimodal feature research after local checksum and manual schema review. |
| `t1dexi` | manual | Jaeb Center / Vivli / Public Study Websites | Type 1 Diabetes Exercise Initiative (T1DEXI) | [source](https://public.jaeb.org/datasets/diabetes) | Primary external dataset for exercise and activity stress testing of local AI models. |
| `t1dexip` | manual | Jaeb Center / Public Study Websites | Type 1 Diabetes Exercise Initiative Pediatric Study (T1DexiP) | [source](https://public.jaeb.org/datasets/diabetes) | Hold-out pediatric generalization dataset, not a direct training replacement for adult cohorts. |
| `dclp3_idcl` | public-download | NIDDK Central Repository / Jaeb Center | International Diabetes Closed Loop Trial (DCLP3 / iDCL) | [source](https://repository.niddk.nih.gov/studies/dclp3/) | External validation and benchmark-style comparisons after conversion to the IINTS standard schema. |
| `jaeb_loop` | public-download | Jaeb Center / Public Study Websites | Loop Observational Study Public Dataset | [source](https://public.jaeb.org/datasets/diabetes) | External real-world AID validation after local schema conversion and subject-level splitting. |
| `metabonet` | mixed | MetaboNet / arXiv | MetaboNet Consolidated T1D Dataset | [source](https://metabo-net.org/) | Best long-term target for broad external validation once local access and terms are documented. |
| `glucose_ml` | collection | arXiv / Glucose-ML authors | Glucose-ML Dataset Collection | [source](https://arxiv.org/abs/2507.14077) | Use as a research map to choose external datasets and benchmark protocols; do not treat it as one homogeneous training set. |

## Documentation-Only Implementation Sources

These references are used in narrative docs, setup guides, local-AI explanation, and best-effort device-emulation notes. They are intentionally documented separately from the packaged scientific manifest.

### Local AI Runtime And Model Setup

| ID | Source | Link | Used for |
| --- | --- | --- | --- |
| `ollama_linux_install` | Ollama Linux installation documentation | [docs.ollama.com/linux](https://docs.ollama.com/linux) | Local LLM setup guidance |
| `mistral_2025_ministral_3_announcement` | Mistral AI, Introducing Mistral 3 | [mistral.ai/news/mistral-3](https://mistral.ai/news/mistral-3) | Local model-family context |
| `mistral_2025_ministral_3_3b` | Mistral AI model docs | [Ministral 3 3B](https://docs.mistral.ai/models/ministral-3-3b-25-12) | Small local model guidance |
| `mistral_2025_ministral_3_8b` | Mistral AI model docs | [Ministral 3 8B](https://docs.mistral.ai/models/ministral-3-8b-25-12) | Mid-size local model guidance |
| `mistral_2025_ministral_3_14b` | Mistral AI model docs | [Ministral 3 14B](https://docs.mistral.ai/models/ministral-3-14b-25-12) | Larger local model guidance |
| `mistral_2026_adjustable_reasoning` | Mistral AI documentation | [Adjustable reasoning](https://docs.mistral.ai/capabilities/reasoning/adjustable) | Serverless replacement settings and `reasoning_effort` |
| `mistral_2026_small_4` | Mistral AI model docs | [Mistral Small 4](https://docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03) | Cloud small-model migration target |
| `mistral_2026_medium_35` | Mistral AI model docs | [Mistral Medium 3.5](https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04) | Cloud strong-model migration target |

### Device Emulation And Pump-Context Sources

| ID | Source | Link | Used for |
| --- | --- | --- | --- |
| `bergenstal_2020_780g` | Bergenstal et al., hybrid closed-loop safety study | [DOI 10.1056/NEJMoa2003479](https://doi.org/10.1056/NEJMoa2003479) | 780G context and benchmark narrative |
| `fda_k193510_780g` | FDA 510(k) K193510 | [FDA accessdata](https://www.accessdata.fda.gov/) | 780G regulatory/device reference |
| `medtronic_780g_user_guide` | Medtronic Diabetes product documentation | [medtronicdiabetes.com](https://www.medtronicdiabetes.com/) | User-guide terminology and mode descriptions |
| `brown_2019_control_iq_dtt` | Brown et al., Control-IQ performance paper | [DOI 10.1089/dia.2019.0226](https://doi.org/10.1089/dia.2019.0226) | Control-IQ benchmark context |
| `fda_k191289_control_iq` | FDA 510(k) K191289 | [FDA accessdata](https://www.accessdata.fda.gov/) | Control-IQ regulatory/device reference |
| `idcl_nct03563313` | ClinicalTrials.gov iDCL trial record | [NCT03563313](https://clinicaltrials.gov/ct2/show/NCT03563313) | Closed-loop trial context |
| `control_iq_user_guide` | Tandem Diabetes Care product documentation | [tandemdiabetes.com](https://www.tandemdiabetes.com/) | Control-IQ user-facing terminology |
| `assert_omnipod_5` | Insulet / Omnipod ASSERT trial page | [omnipod.com/assert-trial](https://www.omnipod.com/assert-trial) | Omnipod 5 context |
| `onset_omnipod_5` | Insulet / Omnipod ONSET trial page | [omnipod.com/onset-trial](https://www.omnipod.com/onset-trial) | Omnipod 5 type 2 diabetes context |
| `fda_k203467_omnipod5` | FDA 510(k) K203467 | [FDA accessdata](https://www.accessdata.fda.gov/) | Omnipod 5 regulatory/device reference |
| `omnipod5_user_guide` | Insulet / Omnipod product documentation | [omnipod.com](https://www.omnipod.com/) | User-guide terminology and system context |

### Hardware And Safety-Engineering Context

| ID | Source | Link | Used for |
| --- | --- | --- | --- |
| `fda_infusion_pump_software_safety_research` | FDA infusion pump software safety research | [FDA](https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-software-safety-research-fda) | Pico pump lab framing: simulate first, verify before hardware |
| `eu_ai_pact` | European Commission AI Pact information | [digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/policies/ai-pact) | EU AI governance framing for research artifacts |

## Source Maintenance Rules

- Add scientific sources to `src/iints/presets/evidence_sources.yaml` when they affect SDK assumptions, validation targets, or report interpretation.
- Add datasets to `src/iints/data/datasets.json` when they affect research data planning or local AI workflows.
- Add docs-only setup/device links here and in [Evidence Base](EVIDENCE_BASE.md) when they explain setup or product context but are not part of the packaged evidence manifest.
- Prefer DOI links for papers and official source pages for datasets, regulators, and vendor documentation.
- Do not cite a source as clinical validation unless the SDK actually performs a clinical validation protocol.

## Export Commands

```bash
iints sources --output-json results/source_manifest.json
iints data research-plan --output-dir data_packs/research_dataset_plan
iints evidence build --run normal=results/live_demo/results/01_normal_run --output-dir results/evidence_bundle
```
