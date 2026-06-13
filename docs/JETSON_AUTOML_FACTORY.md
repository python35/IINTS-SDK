# Jetson AutoML Factory

The Jetson AutoML Factory is a research-only script for continuously training
IINTS glucose-forecast model candidates on edge hardware. It mutates a small
set of hyperparameters, trains one candidate in a subprocess, logs every trial,
and promotes only the best valid model to `models/jetson_champion/`.

It is not a treatment system and does not perform online medical control.

## Preconditions

Run from the SDK repository with an activated Python 3.10+ environment:

```bash
source .venv/bin/activate
iints --help
```

You also need a normalized glucose training dataset, normally:

```text
models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.parquet
```

## Safe First Run

Run one trial first:

```bash
python scratch/jetson_automl_trainer.py --max-trials 1 --timeout-minutes 45
```

If it succeeds, inspect:

```bash
cat jetson_leaderboard.csv
ls models/jetson_champion
```

## Overnight Run

```bash
python scratch/jetson_automl_trainer.py \
  --timeout-minutes 45 \
  --cooldown-seconds 10 \
  --epochs 15 \
  --batch-size 128
```

Stop safely with `Ctrl+C`. The leaderboard and champion folder are preserved.

## What Gets Logged

- `jetson_leaderboard.csv`: every successful or failed trial
- `models/jetson_trials/<trial_id>/trial_config.yaml`: exact config used
- `models/jetson_trials/<trial_id>/train_stdout_stderr.log`: subprocess log
- `models/jetson_champion/predictor.pt`: best candidate checkpoint
- `models/jetson_champion/champion_metadata.json`: score and provenance

Champion selection uses the first available lower-is-better metric:

1. `test_rmse`
2. `val_loss_final`
3. `test_mae`
4. `train_loss_final`

## Jetson Notes

The script sets conservative thread limits and runs each training job in a child
process so memory is reclaimed after every trial. On a Nano, keep
`--batch-size 128` unless you have verified stable thermals and memory.
