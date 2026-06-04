# IINTS-AF Research Track (AI Predictor)

This folder contains the **AI research pipeline** for training a glucose predictor model
that plugs into the IINTS-AF safety workflow.

## Plain Summary
- The predictor forecasts glucose.
- The predictor does **not** dose insulin.
- Safety decisions remain deterministic in the simulator supervisor.

If this is your first time, read `docs/PLAIN_LANGUAGE_GUIDE.md` before this file.

If you are preparing a competition or formal research submission, start with:
- `research/EUCYS_FINAL_WORKFLOW.md`
- `research/EUCYS_REPORT.md`
- `research/EUCYS_SUBMISSION_CHECKLIST.md`

Shortest final-submission path:

```bash
tools/research/run_eucys_final.sh \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10

# update research/EUCYS_REPORT.md with the final numbers

tools/research/render_eucys_report_pdf.sh
```

## Goals
- Train a **prediction model** (not a controller) that forecasts BG 30–120 minutes ahead.
- Use synthetic simulator data for bootstrapping, then fine-tune on real-world datasets.
- Keep Safety Supervisor deterministic; predictor only provides foresight.

## Install (Research Extras)
```bash
pip install iints-sdk-python35[research]
```

## Data Format
We standardize training data to **Parquet** with at least these columns:
- `glucose_actual_mgdl`
- `patient_iob_units`
- `patient_cob_grams`
- `effective_isf`
- `effective_icr`
- `effective_basal_rate_u_per_hr`
- `glucose_trend_mgdl_min`

If you want the strongest current real-data predictor track, blend multiple
prepared datasets instead of training on one small cohort in isolation:

```bash
iints research blend-datasets \
  --source azt1d=data_packs/public/azt1d/processed/azt1d_merged.csv \
  --source hupa=data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv \
  --output data_packs/processed/predictor_blend.csv \
  --manifest data_packs/processed/predictor_blend_manifest.json
```

Use OhioT1DM as a held-out external benchmark whenever possible rather than
silently mixing every source into the training set.

For CGM-only datasets (no insulin/carbs), use:
```bash
python research/prepare_aide_cgm.py \
  --input data_packs/public/aide_t1d/Data\\ Tables/AIDEDeviceCGM.txt \
  --output data_packs/public/aide_t1d/processed/aide_cgm.csv
```
and train with `research/configs/predictor_cgm_only.yaml`.

Full AIDE training run:
```bash
PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/public/aide_t1d/processed/aide_cgm.csv \
  --config research/configs/predictor_cgm_only.yaml \
  --out models/aide_predictor_full
```

AZT1D (CGM + insulin + carbs) preparation and training:
```bash
PYTHONPATH=src python3 research/prepare_azt1d.py \
  --input "data_packs/public/azt1d/AZT1D 2025/CGM Records" \
  --output data_packs/public/azt1d/processed/azt1d_merged.csv

PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/public/azt1d/processed/azt1d_merged.csv \
  --config research/configs/predictor_azt1d.yaml \
  --out models/azt1d_predictor_full
```

OhioT1DM (CGM + insulin + carbs) preparation and training:
```bash
PYTHONPATH=src python3 research/prepare_ohio_t1dm.py \
  --input data_packs/public/OhioT1DM \
  --output data_packs/public/OhioT1DM/processed/ohio_merged.csv

PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/public/OhioT1DM/processed/ohio_merged.csv \
  --config research/configs/predictor_ohio_dual_guard_v2.yaml \
  --out models/ohio_dual_guard
```
Note: The bundled Ohio pack in this repo contains only a few subjects. For a
research‑grade model, add more subjects and/or pretrain on synthetic/AZT1D,
then fine‑tune on OhioT1DM.

Full local OhioT1DM XML release:
```bash
export OHIO_T1DM_ROOT="/path/to/OhioT1DM-volledig"

PYTHONPATH=src python3 research/prepare_ohio_t1dm.py \
  --input "$OHIO_T1DM_ROOT" \
  --splits train \
  --output data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --report data_packs/public/ohio_t1dm_full/processed/ohio_train_quality_report.json

PYTHONPATH=src python3 research/prepare_ohio_t1dm.py \
  --input "$OHIO_T1DM_ROOT" \
  --splits test \
  --output data_packs/public/ohio_t1dm_full/processed/ohio_test.csv \
  --report data_packs/public/ohio_t1dm_full/processed/ohio_test_quality_report.json

PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --config research/configs/predictor_ohio_dual_guard_v2.yaml \
  --out models/ohio_t1dm_full

PYTHONPATH=src python3 research/evaluate_predictor.py \
  --data data_packs/public/ohio_t1dm_full/processed/ohio_test.csv \
  --model models/ohio_t1dm_full/predictor.pt \
  --reference-data data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --subgroup-column subject_id \
  --subgroup-column dataset_year \
  --out results/ohio_t1dm_full_eval.json
```

Keep the raw Ohio XML files and processed CSVs out of git. The repo ignores
`data_packs/public/`, `data_packs/**/processed/`, `models/`, and `results/`.

Recommended v2 recipe details:
- Subject-level holdout split (leakage-safe).
- Band-weighted loss (extra emphasis for hypo/hyper ranges).
- Meal pre-announcement feature reconstruction (`meal_announcement_grams`).
- Early stopping for stability and better generalization.

HUPA-UCM (CGM + insulin + carbs + activity) preparation:
```bash
PYTHONPATH=src python3 research/prepare_hupa_ucm.py \
  --input data_packs/public/hupa_ucm \
  --output data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv
```

“Enormous” model recipe (AZT1D → HUPA fine‑tune):
```bash
# 1) Pretrain on AZT1D (multimodal config; extra channels are zero-filled)
PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/public/azt1d/processed/azt1d_merged.csv \
  --config research/configs/predictor_multimodal_dual_guard.yaml \
  --out models/pretrain_azt1d

# 2) Fine-tune on HUPA-UCM using warm-start (lower LR + early stopping)
PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv \
  --config research/configs/predictor_multimodal_dual_guard_finetune.yaml \
  --warm-start models/pretrain_azt1d/predictor.pt \
  --out models/hupa_finetuned
```

Export to ONNX (edge/Jetson):
```bash
PYTHONPATH=src python3 research/export_predictor.py \
  --model models/hupa_finetuned_v2/predictor.pt \
  --out models/hupa_finetuned_v2/predictor.onnx
```

Paper-aligned Dual-Guard predictor (safety-weighted loss, 120-min horizon):
```bash
PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/public/azt1d/processed/azt1d_merged.csv \
  --config research/configs/predictor_paper_dual_guard.yaml \
  --out models/paper_dual_guard
```

## Training
```bash
python research/train_predictor.py --data data/training.parquet --config research/configs/predictor.yaml --out models
```

## Evaluation
```bash
python research/evaluate_predictor.py --data data/validation.parquet --model models/predictor.pt
```

`evaluate_predictor.py` now enforces checkpoint-compatible feature dimensions and
reconstructs optional meal-announcement features when configured, so evaluation
cannot silently drift from the training pipeline.

MC Dropout calibration run (95% interval coverage + band metrics):
```bash
python research/evaluate_predictor.py \
  --data data/validation.parquet \
  --model models/predictor.pt \
  --mc-samples 50 \
  --out results/predictor_eval.json
```

Strict validation run with external datasets, uncertainty reliability plots,
subgroup reporting, hypo-detection sensitivity, and feature-drift checks:

```bash
python research/evaluate_predictor.py \
  --data data/validation.parquet \
  --model models/predictor.pt \
  --external-data azt1d=data/azt1d_validation.parquet \
  --external-data hupa=data/hupa_validation.parquet \
  --reference-data data/training.parquet \
  --subgroup-column cohort \
  --subgroup-column sex \
  --mc-samples 50 \
  --plots-dir results/predictor_plots \
  --out results/predictor_eval_strict.json
```

The JSON now includes:
- held-out external-dataset metrics
- hypo-detection sensitivity/specificity and missed-hypo rate
- MC-dropout reliability bins plus calibration plots
- subgroup metrics for each requested column
- raw-feature drift scores against the chosen reference dataset

Both training and evaluation outputs include lineage metadata:
- `schema_id` and `schema_version`
- dataframe fingerprint
- optional source file SHA-256

## Simulator realism benchmark
Use the empirical daily envelope derived from the real AZT1D and HUPA-UCM packs to
keep simulator tuning honest across multiple seeds:

```bash
PYTHONPATH=src python3 research/evaluate_simulator_realism.py \
  --presets realistic_reference_day,baseline_t1d,free_living_t1d \
  --seeds 1,2,3,42,99 \
  --reference free_living_t1d \
  --out results/simulator_realism_benchmark.json
```

This benchmark is deliberately separate from predictor training. It checks
whether simulated traces resemble real daily CGM envelopes; the predictor
training commands above measure forecast quality on real data.

## Simulator realism calibration
When a full-day preset needs improvement, calibrate the patient profile against
the same real-data envelope instead of hand-tuning one seed by eye:

```bash
PYTHONPATH=src python3 research/calibrate_simulator_realism.py \
  --preset realistic_reference_day \
  --reference free_living_t1d \
  --seeds 1,2,3,42,99 \
  --out results/simulator_calibration.json \
  --best-profile-out results/reference_free_living_t1d_calibrated.yaml
```

The calibrator searches plausible physiology settings, evaluates every candidate
across multiple deterministic seeds, and ranks them by robust realism first:

## Local controller-policy research

Controller learning is kept separate from predictor learning. Build controller
labels from safety-supervised simulation runs, not by pretending that a language
model should dose insulin directly:

```bash
iints research build-control-dataset \
  --run normal=results/jetson_research_day \
  --run stress=results/jetson_stress_day \
  --output data_packs/processed/controller_teacher_dataset.csv \
  --manifest data_packs/processed/controller_teacher_manifest.json

iints research train-controller \
  --data data_packs/processed/controller_teacher_dataset.csv \
  --output models/controller_imitation.json \
  --metrics-output models/controller_imitation_metrics.json
```

This first controller is intentionally an auditable imitation baseline. It is
useful for proving the research loop end to end before moving on to richer local
policy networks and held-out closed-loop scenario evaluation.

```bash
iints research train-neural-controller \
  --data data_packs/processed/controller_teacher_dataset.csv \
  --output models/controller_neural.pt \
  --metrics-output models/controller_neural_metrics.json

iints research evaluate-controller \
  --model models/controller_neural.pt \
  --model-kind neural \
  --output-dir results/controller_neural_eval
```

The evaluation report is intentionally closed-loop and held out: it compares
the learned controller against the clinical baseline over unseen scenarios and
seeds, with TIR, hypo burden, supervisor interventions, and early termination
counts all visible together.
how many runs are `likely_realistic`, then average realism score, then distance
from the empirical reference median. That makes the chosen preset reproducible
and less vulnerable to one lucky-looking trace.

To calibrate the packaged reference profiles for every supported real-data
envelope in one pass:

```bash
PYTHONPATH=src python3 research/calibrate_dataset_profiles.py \
  --out-dir results/dataset_profile_calibration \
  --profiles-dir src/iints/data/virtual_patients
```

That command refreshes:
- `reference_free_living_t1d`
- `reference_azt1d_t1d`
- `reference_hupa_ucm_t1d`

## Empirical physiology residuals
The mechanistic simulator should remain the source of meal-insulin physiology,
but real CGM days also contain small unmodeled fluctuations. Build the optional
empirical residual library from the local AZT1D and HUPA-UCM packs with:

```bash
PYTHONPATH=src python3 research/build_empirical_residual_profiles.py
```

Then enable the additive residual layer when you explicitly want that research
variant:

```python
outputs = run_simulation(
    algorithm=ClinicalBaselineAlgorithm(),
    scenario=get_preset("free_living_t1d_empirical")["scenario"],
    patient_config="reference_free_living_t1d",
    physiology_variation_profile="free_living_t1d",
    physiology_variation_scale=0.05,
)
```

The default `free_living_t1d` preset stays purely mechanistic because that path
currently scores better against the full realism validator; the empirical layer
is available for controlled experiments rather than silently changing every
benchmark.

## Real-vs-simulator gallery
Render side-by-side overlays for representative AZT1D and HUPA-UCM days:

```bash
MPLCONFIGDIR=.mplt PYTHONPATH=src python3 research/plot_simulator_vs_real.py \
  --output-dir results/realism_gallery
```

## Multi-reference scenario search
Use the scenario search loop when you want the free-living day itself to improve
against several real-data envelopes at once:

```bash
PYTHONPATH=src python3 research/search_realistic_scenarios.py \
  --references free_living_t1d,azt1d_daily,hupa_ucm_daily \
  --seeds 1,42,99 \
  --out results/scenario_search/report.json \
  --best-scenario-out results/scenario_search/best_multi_reference_scenario.json
```

The ranker prefers candidates that stay realistic across **all** references
before rewarding one-off high scores on a single cohort.

## Predictor retraining on improved data
Build a blended real+simulator dataset using the calibrated scenarios:

```bash
PYTHONPATH=src python3 research/build_augmented_training_set.py \
  --output data_packs/generated/realism_augmented_multimodal.csv \
  --manifest data_packs/generated/realism_augmented_multimodal_manifest.json
```

Then warm-start the existing multimodal predictor and fine-tune on the blended
dataset:

```bash
PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/generated/realism_augmented_multimodal.csv \
  --config research/configs/predictor_multimodal_realism_retrain.yaml \
  --warm-start models/hupa_finetuned_v2/predictor.pt \
  --out models/realism_augmented_v1
```

Evaluate the retrained checkpoint on both public real-data packs before using it
as a new default:

```bash
PYTHONPATH=src python3 research/evaluate_predictor.py \
  --data data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv \
  --model models/realism_augmented_v1/predictor.pt \
  --config research/configs/predictor_multimodal_realism_retrain.yaml \
  --out results/realism_augmented_v1_hupa_eval.json

PYTHONPATH=src python3 research/evaluate_predictor.py \
  --data data_packs/public/azt1d/processed/azt1d_merged.csv \
  --model models/realism_augmented_v1/predictor.pt \
  --config research/configs/predictor_multimodal_realism_retrain.yaml \
  --out results/realism_augmented_v1_azt1d_eval.json
```

## Export
```bash
python research/export_predictor.py --model models/predictor.pt --out models/predictor.onnx
```

## Integrate with Simulator (Option 1)
```python
import iints
from iints.research import load_predictor_service
from iints.core.algorithms.pid_controller import PIDController

predictor = load_predictor_service("models/predictor.pt")
sim = iints.Simulator(
    patient_model=iints.PatientModel(),
    algorithm=PIDController(),
    time_step=5,
    predictor=predictor,
)
results, safety = sim.run_batch(720)
```

See `model_card.md` and `datasheet.md` for documentation templates.
