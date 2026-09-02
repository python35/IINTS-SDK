# CGM Foundation Models

IINTS-AF contains an **independent method reproduction** of the GlucoFM v2
architecture described by Metwally et al. It is not Google's source code, does
not include official Google weights, and must not be presented as the official
GlucoFM implementation.

The purpose of this module is reproducible research into CGM representations:
pretraining on local research data, extracting traceable embeddings, and
evaluating models on the same held-out subjects.

> Research and education only. These models are not medical devices and must
> not be used for insulin dosing, diagnosis, or real-time patient care.

Primary references:

- [GlucoFM v2 paper, arXiv:2605.30865v2](https://arxiv.org/abs/2605.30865v2)
- [Google Research overview of GlucoFM](https://research.google/blog/glucofm-foundation-model-for-continuous-glucose-monitoring/)

## Implementation Status

| Capability | IINTS status |
| --- | --- |
| Paper-aligned state/event encoder | Implemented independently |
| Mask-preserving 24-hour alignment | Implemented |
| MCR and temporal-dynamics pretraining | Implemented |
| EMA target encoder | Implemented |
| Secure trained-checkpoint loading | Implemented |
| Subject-disjoint validation split | Required by default |
| Official Google source or weights | Not available in IINTS |
| Clinical validation | Not established |

## Input Contract

Each sample represents 24 hours on a fixed five-minute grid:

- `288` grid positions;
- a glucose-value stream;
- a binary physical-observation mask;
- a circadian grid index anchored to the first timestamp.

Irregular timestamped data is binned onto this grid. Duplicate readings in one
position are averaged. Missing positions remain missing: the loader does not
dense-interpolate them and does not turn imputed values into observations.
Timestamp-free input is accepted only when it already contains exactly 288
chronological positions.

## Architecture

The reproduction follows the paper's main v2 dimensions:

1. A causal, mask-aware Gaussian filter separates a slow state component from
   the event residual. Its sigma is learnable, constrained to `2-12` grid
   positions, initialized at `6`, and uses at most `36` past positions.
2. State and event streams are each divided into `24` patches of `12` positions
   (one hour per patch).
3. State patch features combine waveform, first differences, and summary
   statistics before projection to `64D`.
4. Event patch features combine waveform, rate of change, and summary
   statistics before projection to `64D`.
5. The streams are fused into `24 x 128D` tokens.
6. The encoder uses three Transformer layers, four attention heads, and a
   feed-forward width of `256`.
7. Global mean pooling over the 24 tokens gives one `128D` daily embedding.

This corrects an older IINTS prototype that described a `256D` embedding and
used random weights during extraction. Research-facing extraction now requires
a trained checkpoint.

## Pretraining

```bash
iints research glucofm-pretrain \
  --source data/processed/cgm.csv \
  --glucose-column glucose_mgdl \
  --timestamp-column timestamp \
  --subject-column subject_id \
  --epochs 120 \
  --output-dir models/glucofm-reproduction
```

The command writes:

- `glucofm_encoder.pt`: trained encoder checkpoint;
- `glucofm_pretraining_state.pt`: resumable optimizer/model state;
- `training_history.csv`: epoch-level losses and Gaussian sigma;
- `window_manifest.csv`: window coverage and subject provenance;
- `training_report.json`: configuration, hashes, split type, and limitations.

The default validation split is subject-disjoint. A single-subject window split
is available only through `--allow-single-subject` and is explicitly labelled a
software smoke test, not model evidence.

## Embedding Extraction

```bash
iints research glucofm-embed \
  --input-file data/example_day.csv \
  --checkpoint models/glucofm-reproduction/glucofm_encoder.pt \
  --glucose-column glucose_mgdl \
  --timestamp-column timestamp \
  --output-file results/glucofm/day_embedding.csv
```

The output contains `z_0` through `z_127`. A separate provenance JSON records
the checkpoint SHA-256, training metadata, observed-grid coverage, selected
columns, and input path. Untrained random weights are rejected by CLI and app
workflows.

## PPGR Evaluation

The ordinary PPGR benchmark uses subject-grouped splitting. Its tabular context
ridge is not called GlucoFM. To include GlucoFM, supply both a trained checkpoint
and measured 24-hour pre-meal histories:

```bash
iints research ppgr-benchmark \
  --meals-file data/cgmacros_meals.csv \
  --subjects-file data/cgmacros_subjects.csv \
  --glucofm-checkpoint models/glucofm-reproduction/glucofm_encoder.pt \
  --output-dir results/ppgr
```

The meal table must contain `subject_id` and a JSON-array column named
`pre_meal_cgm_json` (or its sensor-specific equivalent). Without these inputs,
the report contains only the carb, macronutrient, and measured-covariate
baselines and makes no GlucoFM claim.

## Evidence-Only Arena

The Foundation Arena no longer contains built-in scores. It compares evaluation
artifacts only when they share one benchmark ID, metric units/directions, and a
group-disjoint split:

```bash
iints research foundation-arena \
  --result results/evaluations/glucofm.json \
  --result results/evaluations/jepa.json \
  --output-dir results/foundation_arena
```

Each artifact follows schema `iints.foundation-arena.evaluation.v1` and records:

- model architecture, implementation kind, latent dimension, and checkpoint hash;
- benchmark, cohort, task, split, seed, group count, and sample count;
- measured values with units and whether higher or lower is better.

If benchmark IDs differ, group-disjoint evaluation is false, or no common metric
exists, comparison is blocked rather than silently mixing incomparable results.

## Literature Results Versus Local Results

Published GlucoFM values belong to the paper's cohorts and protocols. For
example, the v2 paper reports a task-average PR-AUC of `58.8` and a full-context
PPGR trajectory MAE of `21.88 mg/dL`. These are **literature references**, not
IINTS benchmark outputs. IINTS does not preload them into charts or rank them
against locally evaluated models.

## Jetson Training Boundary

`iints research glucose-model jetson-train-hf` fine-tunes the native IINTS
forecast checkpoint format (`predictor.pt`). It is not a generic adapter for
GlucoFM, CGM-JEPA, or arbitrary Hugging Face backbones. Use
`glucofm-pretrain` for the GlucoFM reproduction. Keeping these paths separate
prevents a downloaded model from being misidentified as a compatible warm start.

## Remaining Research Work

- reproduce downstream paper tasks under documented cohort contracts;
- run repeated subject-grouped folds and confidence intervals;
- evaluate calibration, hypoglycemia sensitivity, and subgroup robustness;
- compare against strong persistence and supervised baselines;
- publish dataset cards and licenses without exposing private patient records;
- seek external replication before making general performance claims.
