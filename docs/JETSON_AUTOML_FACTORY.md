# Jetson / Hugging Face Glucose Training Factory

The recommended Jetson flow is `iints research glucose-model jetson-train-hf`.
It treats the Jetson as a research-only fine-tuning worker for your Hugging
Face glucose-forecast model. It downloads the current model, trains candidates
with warm-start, compares each candidate against the current local champion,
and promotes only candidates that improve a physiology-aware composite score.

It is not a treatment system and does not perform online medical control.

## Preconditions

Run from the SDK repository with an activated Python 3.10+ environment:

```bash
source .venv/bin/activate
iints --help
```

Install and login to the modern Hugging Face CLI:

```bash
hf --version
HF_HOME="$PWD/.cache/huggingface" hf auth login
```

You also need a normalized glucose training dataset:

```text
models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv
```

If you use OhioT1DM, keep the raw `OhioT1DM-volledig/` folder outside GitHub.
Copy it to the Jetson via SSD, `rsync`, or another access-controlled method,
then prepare it locally:

```bash
iints research prepare-ohio \
  --input-dir /path/to/OhioT1DM-volledig \
  --splits train \
  --output data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --report data_packs/public/ohio_t1dm_full/processed/ohio_train_quality_report.json

iints research glucose-model build-dataset \
  --input data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --labels ohio_full \
  --profile long \
  --history-minutes 360 \
  --horizon-minutes 120 \
  --output-dir models/iints-glucose-forecast-v0/dataset
```

## Safe First Run

Run one trial first:

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

If it succeeds, inspect:

```bash
cat models/jetson_hf_training/jetson_hf_leaderboard.csv
ls models/jetson_hf_training/champion
```

## Overnight Run

```bash
nohup iints research glucose-model jetson-train-hf \
  --repo-id IINTS/iints-glucose-forecast-v0 \
  --dataset models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --dataset-manifest models/iints-glucose-forecast-v0/dataset/glucose_dataset_manifest.json \
  --work-dir models/jetson_hf_training \
  --max-trials 0 \
  --timeout-minutes 45 \
  --cooldown-seconds 20 \
  --epochs 8 \
  --batch-size 64 \
  --upload-mode none \
  > jetson_hf_training.log 2>&1 &
```

Stop safely with `Ctrl+C` if running in the foreground, or stop the process if
using `nohup`. The leaderboard and champion folder are preserved.

## What Gets Logged

- `models/jetson_hf_training/jetson_hf_leaderboard.csv`: every successful or failed trial
- `models/jetson_hf_training/hf_base/`: downloaded Hugging Face starting model
- `models/jetson_hf_training/trials/<trial_id>/trial_config.yaml`: exact config used
- `models/jetson_hf_training/trials/<trial_id>/train_stdout_stderr.log`: training log
- `models/jetson_hf_training/trials/<trial_id>/comparison/`: current champion vs candidate metrics
- `models/jetson_hf_training/champion/predictor.pt`: best accepted local checkpoint
- `models/jetson_hf_training/champion/huggingface/`: HF-ready champion bundle

Champion selection uses a lower-is-better composite:

```text
MAE + physiology_weight * physiology_violation_pct + hypo_weight * missed_hypo_rate_pct
```

The default is intentionally conservative. A model is not promoted just because
the raw MAE looks slightly better if it creates more impossible physiology or
worse hypoglycemia behavior.

## Uploading Back To Hugging Face

Default behavior is local-only. To upload a promoted champion as a pull request:

```bash
iints research glucose-model jetson-train-hf \
  --repo-id IINTS/iints-glucose-forecast-v0 \
  --dataset models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --work-dir models/jetson_hf_training \
  --max-trials 1 \
  --upload-mode pr
```

Use `--upload-mode direct` only after reviewing the generated model card,
privacy notes, limitations, and comparison metrics.

## Jetson Notes

The command sets conservative thread limits for subprocesses and compares every
candidate before promotion. On a Nano, start with `--batch-size 64`; increase
only after you have verified stable thermals and memory.

The older `scratch/jetson_automl_trainer.py` script is still useful for local
experiments, but the HF-first command is the preferred path when your model
already lives on Hugging Face.
