# SDK bibliography verification notes

Date: 2026-09-09. Existing user files `paper.bib`, `paper.md` and `CITATION.cff` were not overwritten. This audit changes neither the equations nor the packaged evidence registry.

## Repository sources reviewed

- `src/iints/presets/evidence_sources.yaml`: packaged scientific references.
- `docs/FORMULA_REGISTRY.md` and `docs/PHYSIOLOGY_REFERENCE.md`: equation and physiology references.
- `docs/REGENERATIVE_ISLET_RESEARCH.md`: transplantation, islet and proteomics context.
- `src/iints/data/datasets.json`: acquisition catalogue, not a record of completed experiments.
- `src/iints_desktop/evidence_connectors.py`: connector names, links and integration status.
- `src/iints/research/glucofm.py`: explicitly referenced GlucoFM preprint version.
- `docs/SOURCE_LIBRARY.md`, `docs/EVIDENCE_BASE.md` and `paper.bib`: existing source lists.
- `pyproject.toml`: declared direct Python dependencies, including optional groups.

This is a repository-source audit, not an exhaustive historical or experimental-use audit.
No official CGM-JEPA paper or checkpoint citation was inferred solely from a module name.

## Important corrections

- OhioT1DM: use CEUR Vol. 2675, **paper11.pdf**, pages 71–74, **5th** workshop. The registry pointed to paper2; another dataset record called it the 6th workshop.
- OpenCOR: correct DOI **10.3389/fphys.2015.00026**. The old DOI names a different CellML/OpenCMISS paper.
- Cobry: correct DOI **10.1089/dia.2009.0112**, Diabetes Technology & Therapeutics 12(3):173–177. The old DOI names a diabetic-foot wound-monitor article.
- Renal glucose handling: the DOI metadata names **Ghezzi, Loo and Wright**, not the Hummel author list in the registry.
- ADAPT: the DOI metadata names Choudhary as first author, rather than Benhamou in the registry.
- Visentin: use the actual title **The UVA/Padova Type 1 Diabetes Simulator Goes From Single Meal to Single Day** and DOI metadata, not the mixed title/pages in the registry.
- The faster-aspart and ultra-rapid-lispro title/DOI combinations were not accepted. A separately labelled verified Heise paper is provided as replacement reading, not represented as historical use.

## Entries not accepted as verified citations

| Existing key | Identifier | Reason |
| --- | --- | --- |
| `heise_2017_fiasp_pkpd` | `10.1007/s40262-017-0510-8` | Existing title/DOI combination is not reliable; excluded from the submission bibliography. |
| `klaff_2020_urli_pkpd` | `10.1111/dom.14049` | Existing title/DOI combination is not reliable; excluded from the submission bibliography. |
| `wentholt_2004_cgm_lag` | `10.1089/dia.2004.6.615` | Existing title/DOI combination is not reliable; excluded from the submission bibliography. |

## Verification ledger

“Crossref metadata” verifies bibliographic identity, not the full paper or its implementation. API rate limits were retried. Resource and dataset catalogue entries retain a lower verification level explicitly. No patient-level data were read or exported.

| Number | Citation key | Evidence for record |
| --- | --- | --- |
| 1 | `ada_2026_glycemic_goals` | Crossref metadata, 2026-09-09 |
| 2 | `ada_2026_diabetes_technology` | Crossref metadata, 2026-09-09 |
| 3 | `attd_2019_time_in_range` | Crossref metadata, 2026-09-09 |
| 4 | `nejm_2019_control_iq` | Crossref metadata, 2026-09-09 |
| 5 | `adapt_2022_ahcl` | Crossref metadata, 2026-09-09 |
| 6 | `cobry_2010_meal_bolus_timing` | Crossref metadata, 2026-09-09 |
| 7 | `dalla_man_2007_meal_model` | Crossref metadata, 2026-09-09 |
| 8 | `visentin_2018_uvapadova` | Crossref metadata, 2026-09-09 |
| 9 | `mujahid_2024_generative_t1d_simulator` | Crossref metadata, 2026-09-09 |
| 10 | `riddell_2017_exercise_consensus` | Crossref metadata, 2026-09-09 |
| 11 | `idc_2025_agp_report_overview` | Repository record; web-page metadata not independently re-audited |
| 12 | `bergman_1979_minimal_model` | Crossref metadata, 2026-09-09 |
| 13 | `hovorka_2004_nmpc_t1d` | Crossref metadata, 2026-09-09 |
| 14 | `gerich_1979_counterregulation` | Crossref metadata, 2026-09-09 |
| 15 | `cryer_2013_haaf_mechanisms` | Crossref metadata, 2026-09-09 |
| 16 | `cryer_2013_haaf_diabetes` | Crossref metadata, 2026-09-09 |
| 17 | `lv_2013_exogenous_glucagon_pk` | Crossref metadata, 2026-09-09 |
| 18 | `haidar_2013_dual_hormone_ap` | Crossref metadata, 2026-09-09 |
| 19 | `haidar_2013_insulin_glucagon_pk` | Crossref metadata, 2026-09-09 |
| 20 | `hummel_2018_renal_glucose_handling` | Crossref metadata, 2026-09-09 |
| 21 | `defronzo_2013_renal_reabsorption_splay` | Crossref metadata, 2026-09-09 |
| 22 | `mandelbrot_1968_fractional_brownian` | Crossref metadata, 2026-09-09 |
| 23 | `campbell_1985_dawn_phenomenon` | Crossref metadata, 2026-09-09 |
| 24 | `richter_2013_glut4_exercise` | Crossref metadata, 2026-09-09 |
| 25 | `naslund_1999_glp1_gastric_emptying` | Crossref metadata, 2026-09-09 |
| 26 | `formula_source_4` | Crossref metadata, 2026-09-09 |
| 27 | `formula_source_5` | Crossref metadata, 2026-09-09 |
| 28 | `formula_source_6` | Crossref metadata, 2026-09-09 |
| 29 | `formula_source_7` | Crossref metadata, 2026-09-09 |
| 30 | `formula_source_8` | Crossref metadata, 2026-09-09 |
| 31 | `formula_source_10` | Crossref metadata, 2026-09-09 |
| 32 | `formula_source_11` | Crossref metadata, 2026-09-09 |
| 33 | `formula_source_12` | Crossref metadata, 2026-09-09 |
| 34 | `formula_source_13` | Crossref metadata, 2026-09-09 |
| 35 | `formula_source_14` | Crossref metadata, 2026-09-09 |
| 36 | `dallaman2014` | Crossref metadata, 2026-09-09 |
| 37 | `kovatchev1997` | Crossref metadata, 2026-09-09 |
| 38 | `kovatchev1998` | Crossref metadata, 2026-09-09 |
| 39 | `heise2017_corrected_context` | Crossref metadata, 2026-09-09 |
| 40 | `johansen1984` | Crossref metadata, 2026-09-09 |
| 41 | `peng2022` | Crossref metadata, 2026-09-09 |
| 42 | `numpy2020` | Crossref metadata, 2026-09-09 |
| 43 | `scipy2020` | Crossref metadata, 2026-09-09 |
| 44 | `matplotlib2007` | Crossref metadata, 2026-09-09 |
| 45 | `pinn2019` | Crossref metadata, 2026-09-09 |
| 46 | `neldermead1965` | Crossref metadata, 2026-09-09 |
| 47 | `glucofm2026v2` | arXiv version page, 2026-09-09 |
| 48 | `bindingdb_2016` | Crossref metadata, 2026-09-09 |
| 49 | `ncbi_clinvar_classification_representation` | Repository record; web-page metadata not independently re-audited |
| 50 | `jumper2021` | Crossref metadata, 2026-09-09 |
| 51 | `varadi2024` | Crossref metadata, 2026-09-09 |
| 52 | `reichman2025` | Crossref metadata, 2026-09-09 |
| 53 | `carlsson2025` | Crossref metadata, 2026-09-09 |
| 54 | `balboa2022` | Crossref metadata, 2026-09-09 |
| 55 | `augsornworawat2023` | Crossref metadata, 2026-09-09 |
| 56 | `maxquant2008` | Crossref metadata, 2026-09-09 |
| 57 | `diann2020` | Crossref metadata, 2026-09-09 |
| 58 | `sintov2022` | Crossref metadata, 2026-09-09 |
| 59 | `resource_alphafold-db` | Link and status from SDK connector registry; not independent use verification |
| 60 | `resource_ensembl-vep-alphamissense` | Link and status from SDK connector registry; not independent use verification |
| 61 | `resource_open-targets` | Link and status from SDK connector registry; not independent use verification |
| 62 | `resource_reactome` | Link and status from SDK connector registry; not independent use verification |
| 63 | `resource_rcsb-pdb` | Link and status from SDK connector registry; not independent use verification |
| 64 | `resource_uniprot` | Link and status from SDK connector registry; not independent use verification |
| 65 | `resource_human-protein-atlas` | Link and status from SDK connector registry; not independent use verification |
| 66 | `resource_gtex` | Link and status from SDK connector registry; not independent use verification |
| 67 | `resource_chembl` | Link and status from SDK connector registry; not independent use verification |
| 68 | `resource_clinpgx-pharmgkb` | Link and status from SDK connector registry; not independent use verification |
| 69 | `resource_opencor-physiome` | Link and status from SDK connector registry; not independent use verification |
| 70 | `resource_bindingdb` | Link and status from SDK connector registry; not independent use verification |
| 71 | `resource_string-db` | Link and status from SDK connector registry; not independent use verification |
| 72 | `resource_clinvar` | Link and status from SDK connector registry; not independent use verification |
| 73 | `resource_fair4rs` | Link and status from SDK connector registry; not independent use verification |
| 74 | `resource_pubmed` | Link and status from SDK connector registry; not independent use verification |
| 75 | `resource_clinicaltrials-gov` | Link and status from SDK connector registry; not independent use verification |
| 76 | `resource_zenodo` | Link and status from SDK connector registry; not independent use verification |
| 77 | `pride` | Repository-linked resource; project-level metadata not fully revalidated |
| 78 | `proteomexchange` | Repository-linked resource; project-level metadata not fully revalidated |
| 79 | `pxd001539` | Repository-linked resource; project-level metadata not fully revalidated |
| 80 | `pxd064528` | Repository-linked resource; project-level metadata not fully revalidated |
| 81 | `spectronaut` | Repository-linked resource; project-level metadata not fully revalidated |
| 82 | `trial_zimislecel` | Repository-linked resource; project-level metadata not fully revalidated |
| 83 | `sbml_2019_l3v2_core` | Crossref metadata, 2026-09-09 |
| 84 | `libroadrunner_2015` | Crossref metadata, 2026-09-09 |
| 85 | `copasi_2006` | Crossref metadata, 2026-09-09 |
| 86 | `opencor_2015` | Crossref metadata, 2026-09-09 |
| 87 | `fmi_3_0_2_standard` | Repository record; web-page metadata not independently re-audited |
| 88 | `fmpy_0_3` | Repository record; web-page metadata not independently re-audited |
| 89 | `myvariant_query_api` | Repository record; web-page metadata not independently re-audited |
| 90 | `fair2016` | Crossref metadata, 2026-09-09 |
| 91 | `resource_ro-crate` | Link and status from SDK connector registry; not independent use verification |
| 92 | `resource_biomodels` | Link and status from SDK connector registry; not independent use verification |
| 93 | `resource_libroadrunner` | Link and status from SDK connector registry; not independent use verification |
| 94 | `resource_copasi` | Link and status from SDK connector registry; not independent use verification |
| 95 | `resource_sed-ml` | Link and status from SDK connector registry; not independent use verification |
| 96 | `resource_sbml` | Link and status from SDK connector registry; not independent use verification |
| 97 | `marling_2020_ohiot1dm` | CEUR proceedings and paper, 2026-09-09 |
| 98 | `dataset_sample` | Dataset registry entry; publication metadata not independently revalidated |
| 99 | `dataset_diatrend` | Dataset registry entry; publication metadata not independently revalidated |
| 100 | `dataset_t1d_uom` | Dataset registry entry; publication metadata not independently revalidated |
| 101 | `dataset_t1d_granada` | Dataset registry entry; publication metadata not independently revalidated |
| 102 | `dataset_aide_t1d` | Dataset registry entry; publication metadata not independently revalidated |
| 103 | `dataset_azt1d` | Dataset registry entry; publication metadata not independently revalidated |
| 104 | `dataset_hupa_ucm` | Dataset registry entry; publication metadata not independently revalidated |
| 105 | `dataset_openaps_data_commons` | Dataset registry entry; publication metadata not independently revalidated |
| 106 | `dataset_tidepool_bigdata` | Dataset registry entry; publication metadata not independently revalidated |
| 107 | `dataset_niddk_central` | Dataset registry entry; publication metadata not independently revalidated |
| 108 | `dataset_t1d_exchange` | Dataset registry entry; publication metadata not independently revalidated |
| 109 | `dataset_d1namo` | Dataset registry entry; publication metadata not independently revalidated |
| 110 | `dataset_t1dexi` | Dataset registry entry; publication metadata not independently revalidated |
| 111 | `dataset_dclp3_idcl` | Dataset registry entry; publication metadata not independently revalidated |
| 112 | `dataset_metabonet` | Dataset registry entry; publication metadata not independently revalidated |
| 113 | `dataset_glucose_ml` | Dataset registry entry; publication metadata not independently revalidated |
| 114 | `python` | Implementation/project documentation reference; no new access-date claim |
| 115 | `rust` | Implementation/project documentation reference; no new access-date claim |
| 116 | `tauri` | Implementation/project documentation reference; no new access-date claim |
| 117 | `pyside` | Implementation/project documentation reference; no new access-date claim |
| 118 | `ollama` | Implementation/project documentation reference; no new access-date claim |
| 119 | `mistral` | Implementation/project documentation reference; no new access-date claim |
| 120 | `huggingface` | Implementation/project documentation reference; no new access-date claim |
| 121 | `pytorch` | Implementation/project documentation reference; no new access-date claim |
| 122 | `pandas` | Implementation/project documentation reference; no new access-date claim |
| 123 | `plotly` | Implementation/project documentation reference; no new access-date claim |
| 124 | `scienceplots` | Implementation/project documentation reference; no new access-date claim |
| 125 | `seaborn` | Implementation/project documentation reference; no new access-date claim |
| 126 | `fpdf2` | Implementation/project documentation reference; no new access-date claim |
| 127 | `pymol` | Implementation/project documentation reference; no new access-date claim |
| 128 | `openaps` | Implementation/project documentation reference; no new access-date claim |
| 129 | `simglucose` | Implementation/project documentation reference; no new access-date claim |
| 130 | `raspberry` | Implementation/project documentation reference; no new access-date claim |
| 131 | `arduino` | Implementation/project documentation reference; no new access-date claim |
| 132 | `jetson` | Implementation/project documentation reference; no new access-date claim |
| 133 | `mkdocs` | Implementation/project documentation reference; no new access-date claim |
| 134 | `material` | Implementation/project documentation reference; no new access-date claim |
| 135 | `mermaid` | Implementation/project documentation reference; no new access-date claim |
| 136 | `pytest` | Implementation/project documentation reference; no new access-date claim |
| 137 | `hypothesis` | Implementation/project documentation reference; no new access-date claim |
| 138 | `git` | Implementation/project documentation reference; no new access-date claim |
| 139 | `github` | Implementation/project documentation reference; no new access-date claim |

## Before submission

1. Remove sources not used in the submitted work, or retain them only in a clearly labelled supplementary catalogue.
2. Match each scientific claim to a supporting passage, not merely to a relevant title.
3. Record dataset accession, sample selection, release, licence and held-out split from actual experiment manifests.
4. Record model/checkpoint versions and software environment.
5. Resolve the remaining historical manufacturer/regulatory and literature references before claiming a complete historical audit.
