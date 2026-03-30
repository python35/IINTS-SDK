# Scientific Workflow

Use this workflow when you want the SDK to support a real experimental claim, not just a demo.

## Goal

The strongest public IINTS story is:

- same scenarios
- same seeds
- same algorithms
- different data quality or safety conditions
- measurable deltas in TIR, hypo exposure, interventions, and realism review

That gives you a clear hypothesis-and-evidence loop.

## Recommended Command Sequence

1. Export the official study pack:

```bash
iints scenarios export-study-pack --output-dir scenarios/study_pack
```

For a fixed fair-ready matrix, use:

```bash
iints scenarios export-study-pack --preset eucys --output-dir scenarios/eucys_pack
```

2. Write a reproducible study protocol bundle:

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

3. Generate deliberately corrupted data for the uncertified arm:

```bash
iints data corrupt-for-study data/demo/diabetes_cgm.csv \
  --output-csv data/demo/diabetes_cgm_corrupted.csv \
  --mode timestamp_shift \
  --mode missing_block \
  --mode glucose_spikes
```

4. Run your scenarios:

```bash
for seed in 1 2 3 4 5; do
  iints run-full \
    --algo algorithms/example_algorithm.py \
    --seed "$seed" \
    --duration 1440 \
    --output-dir "results/study_clean/run_$seed"
done
```

5. Analyze the study:

```bash
iints analyze results/study_clean \
  --output-json results/study_clean/study_summary.json \
  --output-markdown results/study_clean/study_summary.md \
  --output-csv results/study_clean/evidence_table.csv \
  --output-evidence-markdown results/study_clean/evidence_table.md
```

6. If you have a CareLink workbench, add an external plausibility check:

```bash
iints analyze results/study_clean \
  --output-json results/study_clean/study_summary.json \
  --carelink-metrics results/personal_carelink/carelink_metrics.json
```

7. Compare clean vs corrupted or supervisor-on vs supervisor-off:

```bash
iints compare-study results/study_clean results/study_corrupted \
  --output-json results/study_comparison.json \
  --output-markdown results/study_comparison.md
```

8. Build a study poster:

```bash
iints poster-study results/study_clean/study_summary.json \
  --output-path results/study_clean/study_poster.png
```

Or let the SDK execute the full fixed matrix for you:

```bash
iints run-eucys-study \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_study \
  --carelink-metrics results/personal_carelink/carelink_metrics.json
```

That command exports the EUCYS preset, runs the clean/corrupted/supervisor-off arms, and writes summaries plus comparisons automatically.

## What Makes This Scientific

- A predefined protocol with explicit hypotheses
- Controlled corruption operators instead of vague “bad data”
- Shared seeds across conditions
- More than one metric
- Descriptive statistics with 95% confidence intervals
- Failure analysis, not just best-case runs
- Optional plausibility comparison against imported real-world CGM traces

## What `iints analyze` Now Adds

- aggregate means for TIR, hypo, hyper, glucose, CV, and GMI
- standard deviation and 95% confidence intervals
- certified vs uncertified split
- failure analysis:
  - terminated early runs
  - severe hypo runs
  - supervisor-heavy runs
  - worst TIR runs
- optional external validation against `carelink_metrics.json`

## Controlled Corruption Modes

`iints data corrupt-for-study` supports:

- `timestamp_shift`
- `missing_block`
- `duplicate_rows`
- `glucose_spikes`
- `drop_meal_annotations`
- `unit_scale_error`

Each corrupted export also writes a manifest JSON so you can document exactly what was changed.

## AI Review

After you prepare a run or CareLink workbench, use:

```bash
iints ai review results/<run_id> --model ministral-3:3b
```

The review is structured into:

- realism verdict
- what looks realistic
- what looks suspicious
- priority fixes
- what to improve next
- follow-up validation checks

That makes the model useful as a critique layer, not just a narration layer.
