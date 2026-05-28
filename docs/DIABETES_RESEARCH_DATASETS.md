# Diabetes Research Datasets

This page is the IINTS-AF dataset map for local AI diabetes research. It lists the external datasets that are useful for simulation realism, glucose forecasting, and bench-only controller research.

!!! warning "Research-only boundary"
    These datasets can support pre-clinical simulation, model development, and documentation. They do not make IINTS-AF a medical device and must not be used for real insulin dosing.

## One-Command Plan

Generate the local acquisition plan:

```bash
iints data research-plan --output-dir data_packs/research_dataset_plan
```

This writes:

| Artifact | Purpose |
| --- | --- |
| `DATASET_ACQUISITION_PLAN.md` | Human-readable acquisition checklist |
| `research_dataset_matrix.csv` | Spreadsheet-style source matrix |
| `dataset_registry_snapshot.json` | Exact registry metadata used for the plan |
| `SOURCE_CITATIONS.bib` | BibTeX citations for reports and papers |
| `research_dataset_plan_manifest.json` | Machine-readable manifest |

## Recommended Order

| Priority | Dataset | Best role in IINTS |
| ---: | --- | --- |
| 1 | `hupa_ucm` | Multimodal predictor training with glucose, insulin, carbs, steps, heart rate, calories, and sleep |
| 2 | `azt1d` | AID-oriented predictor training with detailed bolus and device-mode context |
| 3 | `t1d_uom` | Newer multimodal longitudinal validation set with nutrition, activity, and sleep |
| 4 | `ohio_t1dm` | Classic external benchmark for glucose prediction |
| 5 | `dclp3_idcl` | Closed-loop clinical-trial benchmark and external validation |
| 6 | `jaeb_loop` | Real-world AID/Loop validation source |
| 7 | `t1dexi` / `t1dexip` | Exercise-aware glucose and hypoglycemia-risk research |
| 8 | `d1namo` | Large multimodal/wearable research archive |
| 9 | `openaps_data_commons` | Community AID data, useful after access approval and careful provenance review |
| 10 | `metabonet` / `glucose_ml` | Dataset-selection and cross-dataset benchmark references |

## Access Classes

| Access | Meaning | SDK behavior |
| --- | --- | --- |
| `bundled` | Included with the SDK | `iints data fetch sample` works offline |
| `public-download` | Public URL is known | `iints data fetch <id>` requires pinned hashes or `--no-verify` |
| `manual` | User must download through source page or approved form | SDK writes instructions and expects local files |
| `request` | Requires approval or data-use agreement | SDK records source metadata but does not bypass access rules |
| `mixed` / `collection` | Meta-dataset or mixed access | Use as a benchmark map, not one homogeneous raw dataset |

## Current SDK Support

The SDK has dedicated preparation commands for the first three practical public pipelines:

```bash
iints research prepare-hupa
iints research prepare-azt1d
iints research prepare-ohio
```

For sources without a dedicated converter yet, use the generic importer after extracting the source data:

```bash
iints import-data \
  --input-csv data_packs/public/<dataset_id>/raw/<file>.csv \
  --output-dir data_packs/public/<dataset_id>/standard \
  --data-format generic
```

Then blend only prepared, leakage-safe datasets:

```bash
iints research blend-datasets \
  --source hupa=data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv \
  --source azt1d=data_packs/public/azt1d/processed/azt1d_merged.csv \
  --source ohio=data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv \
  --output data_packs/processed/iints_research_blend.csv \
  --manifest data_packs/processed/iints_research_blend_manifest.json
```

## AI Training Rules

Use this split:

| Model type | Data source | Why |
| --- | --- | --- |
| Glucose predictor | Real-world datasets | Learns glucose dynamics from measured data |
| Controller policy | Safety-supervised simulator/Jetson runs | Learns auditable research actions under known safety constraints |
| Local LLM assistant | Reports, model cards, MDMP payloads | Explains and reviews evidence; does not dose insulin |

Never train an autonomous insulin controller directly from mixed public data without a safety contract, subject-level split, MDMP review, and simulator-only validation.

## Provenance Checklist

Before using any dataset in a model card or EUCYS report:

- Record source URL, DOI, access date, version, and license/access terms.
- Keep raw data read-only under `data_packs/public/<dataset_id>/raw`.
- Save converted data under `data_packs/public/<dataset_id>/processed`.
- Keep `source_dataset` and `subject_id` in every row.
- Split train/validation/test by subject.
- Run `iints data realism-check` and `iints data certify`.
- Put the `research_dataset_plan_manifest.json` next to the model output.

## Sources

- [D1NAMO on Zenodo](https://zenodo.org/records/1421616) publishes the open multimodal D1NAMO archive and its license metadata.
- [Jaeb Public Diabetes Datasets](https://public.jaeb.org/datasets/diabetes) lists T1DEXI, DCLP3/iDCL, Loop, PEDAP, AIDE T1D, and other CGM-containing studies.
- [NIDDK DCLP3/iDCL study page](https://repository.niddk.nih.gov/studies/dclp3/) describes the iDCL/DCLP3 trial and points to public dataset access.
- [T1D-UOM Scientific Data paper](https://www.nature.com/articles/s41597-025-05695-1) describes the longitudinal multimodal dataset and Zenodo repository.
- [AZT1D arXiv paper](https://arxiv.org/abs/2506.14789) describes the AID-oriented AZT1D dataset and its Mendeley DOI.
- [MetaboNet arXiv paper](https://arxiv.org/abs/2601.11505) describes the consolidated T1D dataset and mixed-access public/DUA model.
- [Glucose-ML arXiv paper](https://arxiv.org/abs/2507.14077) summarizes a benchmark collection of longitudinal diabetes datasets for robust AI work.
