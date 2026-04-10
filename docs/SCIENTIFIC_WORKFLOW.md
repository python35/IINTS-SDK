# Scientific Workflow

Use this workflow when you want a real benchmark study instead of a single illustrative run.

## Goal

The SDK now supports a reproducible study engine built around:
- fixed profile sets
- fixed scenario families
- explicit study arms
- shared seeds across all comparisons
- automatic summaries, comparisons, and poster assets

That means we can compare a candidate algorithm against baseline controllers, safety conditions, and corrupted-data ablations without hand-stitching folders afterward.

## The Main Commands

The scientific workflow now centers on five commands:
- `iints study-protocol`
- `iints run-study`
- `iints analyze`
- `iints compare-study`
- `iints poster-study`

`iints run-eucys-study` still exists, but it is now the fixed shortcut for the EUCYS-oriented preset rather than the only research workflow.

## Recommended Flow

### 1. Write The Protocol Bundle

Start by freezing the benchmark design:

```bash
iints study-protocol \
  --preset eucys \
  --profile-set clinic_safe_core \
  --output-dir results/study_protocol
```

This writes:
- `STUDY_PROTOCOL.md`
- `study_design.json`
- `study_matrix.csv`
- `algorithms.json`

The protocol bundle records:
- the research question
- the hypotheses
- the profile set
- the baseline registry
- the study arms
- the scenario matrix
- the seed policy
- the corruption plan
- the recommended follow-up commands

### 2. Run The Generic Study Engine

Use `run-study` for the reusable benchmark path:

```bash
iints run-study \
  --algo algorithms/example_algorithm.py \
  --preset default \
  --profile-set clinic_safe_core \
  --seeds 1,2,3,4,5 \
  --output-dir results/study_bundle
```

This automatically creates:
- `protocol/`
- `scenarios/`
- `study_clean/`
- `study_corrupted/`
- `study_supervisor_off/`
- `comparisons/`

Each run is nested under:

```text
<arm_dir>/<algorithm_id>/<profile_id>/<scenario_slug>_seed_<seed>/
```

and carries explicit metadata for:
- `study_preset`
- `study_arm`
- `condition_group`
- `algorithm_id`
- `algorithm_role`
- `profile_id`
- `scenario_slug`
- `seed`
- `supervisor_enabled`
- `corruption_modes`

### 3. Use The Fixed EUCYS Shortcut When You Want The Full Competition Matrix

If you want the fixed EUCYS-oriented matrix, use:

```bash
iints run-eucys-study \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_study
```

This keeps the official study structure:
- `clean_certified`
- `corrupted_uncertified`
- `supervisor_off_ablation`

and uses the fixed scenario families:
- `baseline_day`
- `meal_challenge`
- `exercise_challenge`
- `supervisor_override`

The EUCYS shortcut also writes:
- `EUCYS_SUMMARY.md`
- `EUCYS_RESULTS_TABLE.csv`
- `EUCYS_FIGURE_MANIFEST.json`
- `EUCYS_LIMITATIONS.md`

### 4. Re-Analyze A Study Folder On Demand

If you already have a bundle or want to re-run the analysis with new options:

```bash
iints analyze results/study_bundle/study_clean \
  --output-json results/study_bundle/study_clean/study_summary.json \
  --output-markdown results/study_bundle/study_clean/study_summary.md \
  --output-csv results/study_bundle/study_clean/evidence_table.csv \
  --output-evidence-markdown results/study_bundle/study_clean/evidence_table.md
```

### 5. Compare Two Study Arms

For example, compare clean certified data against corrupted uncertified data:

```bash
iints compare-study \
  results/study_bundle/study_clean \
  results/study_bundle/study_corrupted \
  --output-json results/study_bundle/comparisons/clean_vs_corrupted.json \
  --output-markdown results/study_bundle/comparisons/clean_vs_corrupted.md
```

### 6. Generate A Poster Summary

```bash
iints poster-study \
  results/study_bundle/study_clean/study_summary.json \
  --output-path results/study_bundle/study_clean/study_poster.png
```

When the richer study fields are present, the poster includes:
- a baseline-vs-candidate comparison panel
- a safety outcomes panel
- a profile heatmap panel

## What `study-protocol` Encodes

The protocol bundle now acts as the authoritative benchmark design writer.

It defines:
- profile set metadata
- candidate and baseline algorithms
- study arms
- scenario families
- metrics
- seed policy
- corruption operators
- reproducibility checklist

By default, the `clinic_safe_core` profile set contains:
- `clinic_safe_baseline`
- `clinic_safe_stress_meal`
- `clinic_safe_hypo_prone`
- `clinic_safe_hyper_challenge`
- `clinic_safe_pizza`
- `clinic_safe_midnight`

By default, the baseline registry includes:
- `PID Controller`
- `Standard Pump`
- `Correction Bolus`

You can disable those defaults or add more comparison algorithms:

```bash
iints study-protocol \
  --output-dir results/custom_protocol \
  --no-include-default-baselines \
  --extra-algorithms "My Published Baseline,My Legacy Controller"
```

## What `analyze` Adds

The study summary remains backward compatible, but now also adds:
- `by_algorithm`
- `by_profile`
- `by_arm`
- `by_scenario`
- `safety_summary`
- `pairwise_baseline_deltas`

If the run outputs contain prediction and uncertainty columns, the summary also adds:
- `calibration_summary`
- `uncertainty_summary`

That uncertainty summary now also includes an `uncertainty_vs_error` block so you can see whether larger predicted uncertainty actually lines up with larger forecast error.

That means the generated study JSON can support:
- cohort-level overview
- subgroup analysis
- candidate-vs-baseline deltas
- safety-first reporting
- uncertainty-aware benchmarking

## Optional External Plausibility Check

If you have a CareLink workbench or reference metrics export, you can still compare simulated metrics against a real-world plausibility reference:

```bash
iints analyze results/study_bundle/study_clean \
  --output-json results/study_bundle/study_clean/study_summary.json \
  --carelink-metrics results/personal_carelink/carelink_metrics.json
```

This is treated as an external plausibility check, not as a clinical efficacy claim.

## Controlled Corruption Modes

`iints data corrupt-for-study` supports:
- `timestamp_shift`
- `missing_block`
- `duplicate_rows`
- `glucose_spikes`
- `drop_meal_annotations`
- `unit_scale_error`

The study protocol also records those corruption operators in the bundle so the ablation logic stays explicit.

## What Makes This Scientific

A strong SDK study now includes:
- a predefined protocol with explicit hypotheses
- a fixed study matrix
- repeated seeds across conditions
- multiple patient profiles
- candidate-vs-baseline comparisons
- supervisor-on vs supervisor-off comparisons
- certified vs uncertified comparisons
- descriptive statistics with confidence intervals
- safety summaries, not only best-case metrics
- optional calibration and uncertainty reporting
- optional external plausibility checks

## Fastest EUCYS Path

If you want the shortest end-to-end competition workflow:

```bash
iints run-eucys-study \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_study
```

Then inspect:
- `results/eucys_study/EUCYS_SUMMARY.md`
- `results/eucys_study/EUCYS_RESULTS_TABLE.csv`
- `results/eucys_study/study_clean/study_poster.png`
- `results/eucys_study/comparisons/clean_vs_corrupted.json`

That gives you one deterministic package for:
- protocol review
- benchmark evidence
- safety comparison
- poster figures
- EUCYS-ready summary artifacts
