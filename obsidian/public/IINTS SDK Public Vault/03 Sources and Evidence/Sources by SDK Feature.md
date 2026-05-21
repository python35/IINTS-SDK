---
tags:
  - iints/sources
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Sources by SDK Feature

Use this when you are explaining IINTS to a doctor, engineer, teacher, or jury member and they ask: **which source supports this part of the SDK?**

| SDK feature | What the user sees | Supporting sources |
| --- | --- | --- |
| Validation targets | TIR/TBR/TAR, hypo framing, report gates | [[ada_2026_glycemic_goals]], [[attd_2019_time_in_range]] |
| CGM and AID context | Language around CGM/AID use and report limitations | [[ada_2026_diabetes_technology]], [[wentholt_2004_cgm_lag]] |
| AID benchmark envelopes | Realistic closed-loop outcome context | [[nejm_2019_control_iq]], [[adapt_2022_ahcl]] |
| Meal scenarios | Meal absorption and pre-bolus timing assumptions | [[dalla_man_2007_meal_model]], [[cobry_2010_meal_bolus_timing]] |
| Insulin action | Rapid and ultra-rapid insulin action profiles | [[heise_2017_fiasp_pkpd]], [[klaff_2020_urli_pkpd]] |
| Exercise stress | Exercise-related glucose behavior in scenarios | [[riddell_2017_exercise_consensus]] |
| Simulator realism | In-silico model realism and validation framing | [[visentin_2018_uvapadova]], [[mujahid_2024_generative_t1d_simulator]], [[bergman_1979_minimal_model]] |
| Forecast training data | CGM and multimodal training/evaluation data provenance | [[marling_2020_ohiot1dm]], [[ohio_t1dm]], [[azt1d]], [[hupa_ucm]], [[diatrend]], [[t1d_uom]] |
| Data import and certification | Where real CGM/pump data sources come from | [[tidepool_bigdata]], [[openaps_data_commons]], [[aide_t1d]], [[pedap]], [[t1d_exchange]] |
| Pico pump lab | Bench-only firmware/package/upload safety boundary | [[fda_infusion_pump_software_safety]], [[fda_infusion_pumps]], [[raspberry_pi_pico_docs]] |
| Local AI and Jetson | Ollama and Mistral local model setup | [[ollama_linux_install]], [[mistral_2025_ministral_3_3b]], [[mistral_2025_ministral_3_8b]], [[mistral_2025_ministral_3_14b]] |
| Device emulation notes | Best-effort public context for 780G / Control-IQ / Omnipod 5 emulators | [[bergenstal_2020_780g]], [[brown_2019_control_iq_dtt]], [[assert_omnipod_5]], [[onset_omnipod_5]] |

## One-Sentence Explanation

IINTS combines published diabetes metrics, public dataset provenance, simulator literature, and explicit bench-safety references so users can run reproducible pre-clinical simulations without pretending the SDK is a treatment product.
