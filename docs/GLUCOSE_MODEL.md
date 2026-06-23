# IINTS Glucose Forecast Model

The IINTS glucose model workflow is for building a dedicated research-only model that reads glucose time-series context and predicts future glucose trends.

It is designed for:

- CGM trend forecasting
- 30/60/120 minute glucose prediction experiments
- hypoglycemia and hyperglycemia risk research
- uncertainty and calibration studies
- local AI experiments using OhioT1DM, AZT1D, HUPA-UCM, simulator exports, and Jetson endurance runs
- Hugging Face model packaging without publishing private/raw dataset rows

It is not designed for:

- real-world treatment decisions
- insulin or glucagon dosing authority
- replacing a deterministic safety supervisor
- uploading gated/private patient data to public repositories

## Mental Model

The workflow has four stages:

```text
prepared glucose datasets
        ↓
iints research glucose-model build-dataset
        ↓
normalized training pack + manifest + config
        ↓
iints research glucose-model train
        ↓
predictor.pt + training_report.json
        ↓
iints research glucose-model export-hf
        ↓
Hugging Face-ready model folder
```

The dedicated model line is called:

```text
iints-glucose-forecast-v0
```

This is intentionally a numeric time-series model, not a language model. LLMs can explain runs, summarize results, or manage research artifacts, but the glucose forecaster itself should be trained and evaluated as a physiological time-series predictor.

The default training profile is **band-PINN first**: the generated config uses `loss: band_pinn`, which combines hypo/hyper range weighting with penalties for impossible glucose bounds, unrealistic rate-of-change, and suspicious IOB/COB logic.

## Build A Training Pack

Use one or more prepared datasets. Keep full/private OhioT1DM data outside git.

```bash
export OHIO_T1DM_ROOT="/path/to/OhioT1DM-volledig"

PYTHONPATH=src python3 research/prepare_ohio_t1dm.py \
  --input "$OHIO_T1DM_ROOT" \
  --splits train \
  --output data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --report data_packs/public/ohio_t1dm_full/processed/ohio_train_quality_report.json
```

Then normalize the training sources into the glucose-model contract:

```bash
iints research glucose-model build-dataset \
  --input data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --input results/realism_learning_10k/research/predictor_training.csv \
  --labels ohio_full,sim_10k \
  --profile long \
  --history-minutes 360 \
  --horizon-minutes 120 \
  --output-dir models/iints-glucose-forecast-v0/dataset
```

This writes:

```text
models/iints-glucose-forecast-v0/dataset/
├── glucose_training_dataset.csv
├── glucose_dataset_manifest.json
├── glucose_model_config.yaml
└── MODEL_INTENT.md
```

## OhioT1DM On Jetson Without Pushing Raw Data

Do not commit the full raw OhioT1DM folder to GitHub. Keep it on a local SSD,
Jetson disk, or another access-controlled storage location. The repository
contains the preparation code and training commands, not the gated/raw dataset.

If the raw folder is available on the Jetson, prepare it locally:

```bash
iints research prepare-ohio \
  --input-dir /path/to/OhioT1DM-volledig \
  --splits train \
  --output data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --report data_packs/public/ohio_t1dm_full/processed/ohio_train_quality_report.json
```

Then build the normalized glucose-model dataset:

```bash
iints research glucose-model build-dataset \
  --input data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --labels ohio_full \
  --profile long \
  --history-minutes 360 \
  --horizon-minutes 120 \
  --output-dir models/iints-glucose-forecast-v0/dataset
```

The generated processed files live under gitignored folders (`data_packs/` and
`models/`). They are available to the Jetson for training, but they are not
accidentally published to GitHub.

## Train The Model

For a quick smoke test:

```bash
iints research glucose-model train \
  --data models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --config models/iints-glucose-forecast-v0/dataset/glucose_model_config.yaml \
  --output-dir models/iints-glucose-forecast-v0 \
  --epochs 2
```

For a serious long local run:

```bash
iints research glucose-model train \
  --data models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --config models/iints-glucose-forecast-v0/dataset/glucose_model_config.yaml \
  --output-dir models/iints-glucose-forecast-v0 \
  --epochs 220 \
  --batch-size 256 \
  --export-hf
```

For the current recommended OhioT1DM fine-tune path, start from the best
band-weighted 120-minute checkpoint and continue with the combined band-PINN
objective:

```bash
iints research glucose-model train \
  --data models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --config research/configs/predictor_ohio_band_pinn_v3.yaml \
  --output-dir models/iints-glucose-forecast-v0-band-pinn-v3 \
  --warm-start models/iints-glucose-forecast-v0-ohio-safe-band/predictor.pt \
  --export-hf
```

Promote this model only if the comparison report improves the 120-minute
trade-off: MAE/RMSE should stay competitive while physiological violations,
missed hypoglycemia, and suspicious IOB/COB trajectories decrease.

The training output includes:

```text
models/iints-glucose-forecast-v0/
├── predictor.pt
├── training_report.json
├── glucose_model_config.resolved.yaml
└── huggingface/
```

## Evaluate Against Held-Out Data

Use external data as a separate benchmark whenever possible:

```bash
PYTHONPATH=src python3 research/evaluate_predictor.py \
  --data data_packs/public/ohio_t1dm_full/processed/ohio_test.csv \
  --model models/iints-glucose-forecast-v0/predictor.pt \
  --config models/iints-glucose-forecast-v0/glucose_model_config.resolved.yaml \
  --reference-data models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --out results/iints_glucose_forecast_v0_eval.json \
  --mc-samples 30
```

Minimum metrics to inspect:

- MAE and RMSE by forecast horizon
- band-wise error for hypo, target, and hyper ranges
- missed hypoglycemia rate
- false hypoglycemia alarm rate
- uncertainty calibration if MC dropout is used
- subject-level split and leakage audit
- external dataset performance, not just internal validation

## Compare MSE, Band-Weighted, PINN, And Band-PINN Models

After training multiple candidates, compare them with one command:

```bash
iints research glucose-model compare \
  --data data_packs/public/ohio_t1dm_full/processed/ohio_test.csv \
  --config models/iints-glucose-forecast-v0/dataset/glucose_model_config.yaml \
  --model mse=models/glucose_mse/predictor.pt \
  --model band=models/glucose_band/predictor.pt \
  --model pinn=models/glucose_pinn/predictor.pt \
  --model band_pinn=models/iints-glucose-forecast-v0/predictor.pt \
  --mc-samples 30 \
  --output-dir results/glucose_model_comparison
```

This writes:

```text
results/glucose_model_comparison/
├── comparison_report.json
├── comparison_report.md
├── horizon_metrics.csv
├── physiological_violation_metrics.csv
├── hypo_detection_metrics.csv
├── model_card_metrics.json
└── figures/
```

The key idea is simple: do not promote a model just because MAE improved. A useful diabetes model must also reduce missed hypoglycemia, avoid overconfident uncertainty, and reduce physiologically impossible predictions.

The comparison gate checks:

- MAE, RMSE, bias, and within-range error
- per-horizon metrics across the forecast window
- missed hypoglycemia rate and false hypo alarms
- impossible glucose predictions below 20 mg/dL or above 600 mg/dL
- unrealistic predicted rate-of-change
- suspicious rise with high IOB and no COB
- suspicious drop with COB and no IOB

For the scientific reasoning behind these outputs, see [Interpreting Glucose Forecast Results](comparison_interpretation.md).
That page explains why the lowest MSE is not automatically the best research model, why PINN can be preferable, and why long-horizon forecasts are harder.

## Export For Hugging Face

After training:

```bash
iints research glucose-model export-hf \
  --model-dir models/iints-glucose-forecast-v0 \
  --dataset-manifest models/iints-glucose-forecast-v0/dataset/glucose_dataset_manifest.json \
  --comparison-dir results/glucose_model_comparison \
  --repo-id IINTS/iints-glucose-forecast-v0 \
  --output-dir models/iints-glucose-forecast-v0/huggingface
```

The export folder contains:

```text
huggingface/
├── README.md
├── PUBLISHING.md
├── privacy.md
├── limitations.md
├── config.json
├── glucose_model_config.yaml
├── predictor.pt
├── training_report.json
├── dataset_manifest.public.json
├── comparison_report.md
├── comparison_interpretation.md
├── comparison_report.json
├── horizon_metrics.csv
├── physiological_violation_metrics.csv
├── hypo_detection_metrics.csv
├── model_card_metrics.json
└── examples/
    ├── inference_example.py
    └── sample_glucose_trace.csv
```

The public manifest redacts local source paths and raw file hashes. This is important for gated datasets such as OhioT1DM.
The comparison files are optional, but strongly recommended before publishing because they show why the model is judged by physiology-aware safety gates, not only by MAE.

## Continue Training On Jetson From Hugging Face

If your model already exists on Hugging Face, use the Jetson as a conservative
fine-tuning worker. The SDK downloads the current model, trains candidates with
warm-start, compares the candidate against the current local champion, and only
promotes the candidate when a physiology-aware composite score improves.

Login once on the Jetson:

```bash
HF_HOME="$PWD/.cache/huggingface" hf auth login
```

Run one safe smoke trial:

```bash
iints research glucose-model jetson-train-hf \
  --repo-id IINTS/iints-glucose-forecast-v0 \
  --dataset models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --dataset-manifest models/iints-glucose-forecast-v0/dataset/glucose_dataset_manifest.json \
  --work-dir models/jetson_hf_training \
  --max-trials 1 \
  --epochs 2 \
  --batch-size 64 \
  --upload-mode none
```

If that succeeds, start a longer run:

```bash
nohup iints research glucose-model jetson-train-hf \
  --repo-id IINTS/iints-glucose-forecast-v0 \
  --dataset models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --dataset-manifest models/iints-glucose-forecast-v0/dataset/glucose_dataset_manifest.json \
  --work-dir models/jetson_hf_training \
  --max-trials 0 \
  --epochs 8 \
  --batch-size 64 \
  --timeout-minutes 45 \
  --cooldown-seconds 20 \
  --upload-mode none \
  > jetson_hf_training.log 2>&1 &
```

Monitor progress:

```bash
tail -f jetson_hf_training.log
cat models/jetson_hf_training/jetson_hf_leaderboard.csv
ls models/jetson_hf_training/champion
```

When you are ready to send a candidate to Hugging Face, prefer a pull request:

```bash
iints research glucose-model jetson-train-hf \
  --repo-id IINTS/iints-glucose-forecast-v0 \
  --dataset models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --work-dir models/jetson_hf_training \
  --max-trials 1 \
  --epochs 8 \
  --batch-size 64 \
  --upload-mode pr
```

This command does not upload raw OhioT1DM rows. It uploads the champion model
bundle only when a candidate is promoted, with the research-only model card,
privacy notes, limitations, comparison artifacts, and example inference script.

## Private-First Upload

Upload privately first:

```bash
cd models/iints-glucose-forecast-v0/huggingface
hf upload IINTS/iints-glucose-forecast-v0 . . --type model --private
```

Before making it public, verify:

- no raw OhioT1DM rows are included
- no private local file paths are visible
- the model card clearly says research-only and not for treatment
- evaluation includes held-out subjects
- limitations and uncertainty are documented

## Feature Contract

The v0 feature contract includes:

```text
glucose_actual_mgdl
glucose_trend_mgdl_min
patient_iob_units
patient_cob_grams
delivered_insulin_units
carb_intake_grams
effective_isf
effective_icr
effective_basal_rate_u_per_hr
exercise_intensity
stress_intensity
steps
heart_rate
time_of_day_sin
time_of_day_cos
glucagon_mg
haaf_memory
```

Missing optional features are filled with conservative defaults during dataset preparation. Glucose is required.

## Research Boundary

This model should be treated as a forecast signal only. In IINTS controller experiments, the deterministic supervisor remains the final authority. The model can inform analysis and simulation, but it must never bypass safety constraints.
