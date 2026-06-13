---
tags:
  - iints/sources
  - iints/edge
  - iints/local-ai
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Local AI and Hardware Sources

This page collects the sources behind user-facing local AI, Jetson, device-emulation, and Pico pump lab guidance.

| ID | Category | SDK component | Title | URL | Note |
| --- | --- | --- | --- | --- | --- |
| ollama_linux_install | runtime | local_ai_setup | Ollama Linux Installation Documentation | https://docs.ollama.com/linux | [[ollama_linux_install]] |
| mistral_2025_ministral_3_announcement | model_card | local_ai_model_selection | Mistral AI: Introducing Mistral 3 | https://mistral.ai/news/mistral-3 | [[mistral_2025_ministral_3_announcement]] |
| mistral_2025_ministral_3_3b | model_card | local_ai_model_selection | Mistral AI Docs: Ministral 3 3B | https://docs.mistral.ai/models/ministral-3-3b-25-12 | [[mistral_2025_ministral_3_3b]] |
| mistral_2025_ministral_3_8b | model_card | local_ai_model_selection | Mistral AI Docs: Ministral 3 8B | https://docs.mistral.ai/models/ministral-3-8b-25-12 | [[mistral_2025_ministral_3_8b]] |
| mistral_2025_ministral_3_14b | model_card | local_ai_model_selection | Mistral AI Docs: Ministral 3 14B | https://docs.mistral.ai/models/ministral-3-14b-25-12 | [[mistral_2025_ministral_3_14b]] |
| bergenstal_2020_780g | trial | medtronic_780g_emulation_context | Safety of a Hybrid Closed-Loop Insulin Delivery System in Patients With Type 1 Diabetes | https://doi.org/10.1056/NEJMoa2003479 | [[bergenstal_2020_780g]] |
| fda_k193510_780g | regulatory | medtronic_780g_emulation_context | U.S. FDA 510(k) K193510 - MiniMed 780G System | https://www.accessdata.fda.gov/ | [[fda_k193510_780g]] |
| medtronic_780g_user_guide | technical_manual | medtronic_780g_emulation_context | Medtronic MiniMed 780G User Guide / Product Documentation | https://www.medtronicdiabetes.com/ | [[medtronic_780g_user_guide]] |
| brown_2019_control_iq_dtt | trial | tandem_control_iq_emulation_context | Performance of Tandem t:slim X2 with Control-IQ technology in the IDCL trial | https://doi.org/10.1089/dia.2019.0226 | [[brown_2019_control_iq_dtt]] |
| fda_k191289_control_iq | regulatory | tandem_control_iq_emulation_context | U.S. FDA 510(k) K191289 - Control-IQ System | https://www.accessdata.fda.gov/ | [[fda_k191289_control_iq]] |
| idcl_nct03563313 | clinical_trial | tandem_control_iq_emulation_context | ClinicalTrials.gov: International Diabetes Closed Loop Trial | https://clinicaltrials.gov/ct2/show/NCT03563313 | [[idcl_nct03563313]] |
| control_iq_user_guide | technical_manual | tandem_control_iq_emulation_context | Tandem Control-IQ User Guide / Product Documentation | https://www.tandemdiabetes.com/ | [[control_iq_user_guide]] |
| assert_omnipod_5 | clinical_trial | omnipod_5_emulation_context | Insulet / Omnipod ASSERT Trial - Omnipod 5 | https://www.omnipod.com/assert-trial | [[assert_omnipod_5]] |
| onset_omnipod_5 | clinical_trial | omnipod_5_emulation_context | Insulet / Omnipod ONSET Trial - Omnipod 5 in Type 2 Diabetes | https://www.omnipod.com/onset-trial | [[onset_omnipod_5]] |
| fda_k203467_omnipod5 | regulatory | omnipod_5_emulation_context | U.S. FDA 510(k) K203467 - Omnipod 5 System | https://www.accessdata.fda.gov/ | [[fda_k203467_omnipod5]] |
| omnipod5_user_guide | technical_manual | omnipod_5_emulation_context | Insulet / Omnipod 5 User Guide / Product Documentation | https://www.omnipod.com/ | [[omnipod5_user_guide]] |
| fda_infusion_pump_software_safety | regulatory | pico_pump_lab_safety | FDA Infusion Pump Software Safety Research | https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-software-safety-research-fda | [[fda_infusion_pump_software_safety]] |
| fda_infusion_pumps | regulatory | pico_pump_lab_safety | FDA Infusion Pumps | https://www.fda.gov/medical-devices/general-hospital-devices-and-supplies/infusion-pumps | [[fda_infusion_pumps]] |
| raspberry_pi_pico_docs | technical_manual | pico_pump_lab_firmware | Raspberry Pi Pico-series microcontrollers documentation | https://www.raspberrypi.com/documentation/microcontrollers/raspberry-pi-pico.html | [[raspberry_pi_pico_docs]] |

## User Boundary

- Local AI can help summarize, inspect, and assist research workflows.
- It must not autonomously dose insulin.
- Pico and pump tooling stays **bench-only** until formal verification, hardware safety review, and regulatory pathways exist.
