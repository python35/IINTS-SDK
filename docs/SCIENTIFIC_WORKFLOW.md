# Scientific Workflow

Use this workflow when you want reproducible experimental comparisons rather than a single illustrative run.

## Goal

A strong SDK study compares the same algorithms under controlled conditions:
- same scenario set
- same seeds
- clearly defined condition changes
- measurable differences in safety and glycemic metrics

This gives you a clean hypothesis-and-evidence loop.

## Recommended Command Sequence

### 1) Export The Study Pack

```bash
iints scenarios export-study-pack --output-dir scenarios/study_pack
```

For the fixed EUCYS-oriented matrix, use:

```bash
iints scenarios export-study-pack --preset eucys --output-dir scenarios/eucys_pack
```

### 2) Write A Reproducible Protocol Bundle

```bash
iints study-protocol --preset eucys --output-dir results/study_protocol
```

This writes:
- `STUDY_PROTOCOL.md`
- `study_design.json`
- `study_matrix.csv`

The `eucys` preset fixes:
- the seed list
- the official scenario pack
- a clean certified arm
- a corrupted uncertified arm
- a supervisor-off ablation arm

### 3) Generate Corrupted Data For The Comparison Arm

```bash
iints data corrupt-for-study data/demo/diabetes_cgm.csv \
  --output-csv data/demo/diabetes_cgm_corrupted.csv \
  --mode timestamp_shift \
  --mode missing_block \
  --mode glucose_spikes
```

### 4) Run The Study Scenarios

```bash
for seed in 1 2 3 4 5; do
  iints run-full \
    --algo algorithms/example_algorithm.py \
    --seed "$seed" \
    --duration 1440 \
    --output-dir "results/study_clean/run_$seed"
done
```

### 5) Analyze The Result Folder

```bash
iints analyze results/study_clean \
  --output-json results/study_clean/study_summary.json \
  --output-markdown results/study_clean/study_summary.md \
  --output-csv results/study_clean/evidence_table.csv \
  --output-evidence-markdown results/study_clean/evidence_table.md
```

### 6) Add An External Plausibility Check If Available

If you have a CareLink workbench, compare the simulated metrics against imported real-world traces:

```bash
iints analyze results/study_clean \
  --output-json results/study_clean/study_summary.json \
  --carelink-metrics results/personal_carelink/carelink_metrics.json
```

### 7) Compare Study Arms

```bash
iints compare-study results/study_clean results/study_corrupted \
  --output-json results/study_comparison.json \
  --output-markdown results/study_comparison.md
```

### 8) Build A Study Poster

```bash
iints poster-study results/study_clean/study_summary.json \
  --output-path results/study_clean/study_poster.png
```

### 9) Run The Full Fixed Matrix Automatically

```bash
iints run-eucys-study \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_study \
  --carelink-metrics results/personal_carelink/carelink_metrics.json
```

That command exports the EUCYS preset, runs the clean, corrupted, and supervisor-off arms, then writes summaries and comparisons automatically.

## What Makes This Scientific

A solid study includes:
- a predefined protocol with explicit hypotheses
- shared seeds across conditions
- controlled corruption operators instead of vague “bad data”
- more than one outcome metric
- descriptive statistics with confidence intervals
- failure analysis, not just best-case runs
- optional plausibility checks against imported real-world traces

## What `iints analyze` Adds

The study summary includes:
- aggregate means for TIR, hypo, hyper, glucose, CV, and GMI
- standard deviation and 95% confidence intervals
- certified vs uncertified split
- failure analysis for terminated, severe-hypo, supervisor-heavy, and worst-TIR runs
- optional external validation against `carelink_metrics.json`

## Controlled Corruption Modes

`iints data corrupt-for-study` supports:
- `timestamp_shift`
- `missing_block`
- `duplicate_rows`
- `glucose_spikes`
- `drop_meal_annotations`
- `unit_scale_error`

Each corrupted export also writes a manifest JSON so the modifications are documented explicitly.

## AI Review

After preparing a run or CareLink workbench, you can add a realism review layer:

```bash
iints ai review results/<run_id> --model ministral-3:3b
```

The review is structured into:
- realism verdict
- realistic patterns
- suspicious patterns
- priority fixes
- next validation steps

That makes the model useful as a critique layer, not just a narrative layer.
