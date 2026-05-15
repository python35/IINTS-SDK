# Study Analysis

Use this page when you already have multiple run folders and need one defensible summary for review, comparison, or presentation.

**Read before:** [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) if you have not created a study bundle yet.

**Read next:** [Evidence Base](EVIDENCE_BASE.md) when you need to explain what the metrics mean.

## What The Analyzer Produces

`iints analyze` scans run folders with `results.csv` and summarizes:

- mean time in range and out-of-range exposure
- severe hypo and hyper exposure
- supervisor interventions
- mean glucose and coefficient of variation
- descriptive statistics including standard deviation and 95% confidence intervals
- baseline comparisons when available
- certification splits when certification JSON exists
- failure analysis for worst runs, severe hypo, and early termination
- optional plausibility deltas against CareLink-style real-world metrics

## Fastest Useful Flow

```bash
for seed in 1 2 3 4 5 6 7 8 9 10; do
  iints run-full \
    --algo algorithms/example_algorithm.py \
    --seed "$seed" \
    --duration 1440 \
    --output-dir "results/study/run_$seed"
done

iints analyze results/study \
  --output-json results/study_summary.json \
  --output-markdown results/study_summary.md \
  --output-csv results/evidence_table.csv \
  --output-evidence-markdown results/evidence_table.md
```

## Common Follow-Up Commands

### Compare two study arms

```bash
iints compare-study results/study_certified results/study_uncertified \
  --output-json results/study_comparison.json \
  --output-markdown results/study_comparison.md
```

### Build a poster

```bash
iints poster-study results/study_summary.json \
  --output-path results/study_poster.png
```

### Start from a written protocol

```bash
iints study-protocol --output-dir results/study_protocol
iints run-study --experiment results/study_protocol/study_experiment.yaml
```

### Create a controlled corrupted arm

```bash
iints data corrupt-for-study data/demo/diabetes_cgm.csv \
  --output-csv data/demo/diabetes_cgm_corrupted.csv \
  --mode timestamp_shift \
  --mode missing_block \
  --mode glucose_spikes
```

## Output Files

- `study_summary.json`: machine-readable aggregate summary
- `study_summary.md`: readable narrative summary
- `evidence_table.csv`: poster- and paper-ready rows
- `evidence_table.md`: markdown table for docs or slides
- `external_validation`: optional real-world plausibility deltas
- `failure_analysis`: worst runs and safety-heavy cases
- `aggregate_stats`: descriptive statistics and confidence intervals

Each run entry records:

- run id
- scenario
- algorithm
- TIR 70-180
- supervisor interventions
- certification grade
- baseline delta when available
- quality badges such as `strong_tir`, `stable_variability`, or `supervisor_heavy`

## Official Study Pack

```bash
iints scenarios export-study-pack --output-dir scenarios/study_pack
```

This writes reusable scenario JSON files, a manifest with the recommended seed list, and a short README with the batch pattern.

## Why This Matters

Study analysis is the step that turns “we ran the simulator” into evidence you can defend:

- repeated-run performance rather than one cherry-picked trace
- explicit safety behavior across scenarios
- candidate-versus-baseline comparison
- certified-versus-uncertified comparison when both exist
- a written study protocol instead of an undocumented benchmark

## Where To Go Next

| If you want to... | Continue with |
| --- | --- |
| build the study from scratch | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| certify or corrupt data deliberately | [MDMP Quickstart](MDMP_QUICKSTART.md) |
| create presentation assets | [Booth Demo Guide](BOOTH_DEMO.md) |
| explain source claims | [Evidence Base](EVIDENCE_BASE.md) |
