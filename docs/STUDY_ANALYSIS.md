# Study Analysis

Use `iints analyze` when you have multiple run folders and want one summary for posters, demos, or review.

**Before this page:** [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) if you have not created a study bundle yet.

**After this page:** [Evidence Base](EVIDENCE_BASE.md) if you need to explain what your metrics and sources mean.

## What It Does

`iints analyze` scans a study directory for run folders containing `results.csv` and aggregates:

- mean time in range (`tir_70_180`)
- mean hypo and hyper exposure (`tir_below_70`, `tir_below_54`, `tir_above_180`, `tir_above_250`)
- mean supervisor interventions
- mean glucose and CV
- descriptive statistics including standard deviation and 95% confidence intervals
- baseline comparison rows when available
- certification split (`certified` vs `uncertified`) when certification JSON exists
- failure analysis for worst runs, severe hypo, and early terminations
- optional external plausibility comparison against `carelink_metrics.json`

## Recommended Flow

Run a small study:

```bash
for seed in 1 2 3 4 5 6 7 8 9 10; do
  iints run-full \
    --algo algorithms/example_algorithm.py \
    --seed "$seed" \
    --duration 1440 \
    --output-dir "results/study/run_$seed"
done
```

Then aggregate it:

```bash
iints analyze results/study \
  --output-json results/study_summary.json \
  --output-markdown results/study_summary.md \
  --output-csv results/evidence_table.csv \
  --output-evidence-markdown results/evidence_table.md \
  --carelink-metrics results/personal_carelink/carelink_metrics.json
```

Compare two studies directly:

```bash
iints compare-study results/study_certified results/study_uncertified \
  --output-json results/study_comparison.json \
  --output-markdown results/study_comparison.md
```

Build a poster from the study:

```bash
iints poster-study results/study_summary.json \
  --output-path results/study_poster.png
```

Build the full expo bundle:

```bash
iints demo-expo --output-dir results/expo_demo
```

Write a reproducible study protocol before you start:

```bash
iints study-protocol --output-dir results/study_protocol
```

Then run the exact same plan from the generated experiment file:

```bash
iints run-study --experiment results/study_protocol/study_experiment.yaml
```

Generate a controlled corrupted dataset for the uncertified arm:

```bash
iints data corrupt-for-study data/demo/diabetes_cgm.csv \
  --output-csv data/demo/diabetes_cgm_corrupted.csv \
  --mode timestamp_shift \
  --mode missing_block \
  --mode glucose_spikes
```

## Output

- `study_summary.json`: machine-readable aggregate summary
- `study_summary.md`: easy-to-share narrative summary
- `evidence_table.csv`: poster/paper-ready evidence rows
- `evidence_table.md`: markdown table for docs or slides
- `external_validation`: optional deltas vs CareLink-style real-world metrics
- `failure_analysis`: worst runs and safety-heavy run counts
- `aggregate_stats`: descriptive stats and confidence intervals

Each run entry includes:

- run id
- scenario name
- algorithm
- TIR 70-180
- supervisor interventions
- certification grade
- delta versus baseline reference when baseline comparison exists
- quality badges such as `strong_tir`, `stable_variability`, or `supervisor_heavy`

## Official Study Pack

Export the built-in public study pack:

```bash
iints scenarios export-study-pack --output-dir scenarios/study_pack
```

That writes:

- reusable scenario JSON files
- `study_pack_manifest.json` with the recommended seed list
- a small README with the batch-loop pattern

## Why This Matters

This is the command that turns “we ran the simulator” into evidence you can show:

- average performance over many runs
- safety behavior across scenarios
- comparison against baseline algorithms
- clinician-style `Clinical Baseline` comparison in the default protocol bundle
- comparison between certified and uncertified data when both are present
- a written protocol that explains the hypothesis and study matrix
- a deliberate corruption workflow instead of vague “bad data”

## Where To Go Next

| If you want to... | Continue with |
|---|---|
| build the study from scratch | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| certify or corrupt data deliberately | [MDMP Quickstart](MDMP_QUICKSTART.md) |
| create poster-ready assets | [Booth Demo & Presentation Flow](BOOTH_DEMO.md) |
| understand source claims | [Evidence Base](EVIDENCE_BASE.md) |
| browse every analysis command | [Command Reference](COMMAND_REFERENCE.md) |
