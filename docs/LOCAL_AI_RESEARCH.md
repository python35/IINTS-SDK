# Local AI Research

Use this page when you want to train local models from IINTS-AF data instead of only running hand-written algorithms.

The SDK now separates three AI roles:

| Role | Best model type | Trained on | Purpose |
| --- | --- | --- | --- |
| explanation assistant | local LLM such as Ministral via Ollama | reports and certified payloads | explain, summarize, review |
| glucose predictor | time-series model | real multimodal T1D datasets | forecast future glucose |
| controller policy | compact numeric policy model | supervised safe-action labels from simulated runs | propose insulin actions in research simulations |

Do **not** confuse them:
- Ministral is useful for language and review.
- A controller needs physiological inputs and numeric validation.
- The deterministic safety supervisor still wraps experimental policy outputs.

## Recommended Data Strategy

| Dataset family | Best current role | Why |
| --- | --- | --- |
| AZT1D | primary multimodal predictor training | real AID-system data with detailed bolus variables |
| HUPA-UCM | multimodal predictor training and subgroup analysis | CGM, insulin, carbs, steps, calories, heart rate, sleep |
| OhioT1DM | external held-out benchmark | widely used T1D forecasting benchmark with CGM, insulin, meals, exercise, and life events |
| Jetson / simulator teacher runs | controller-policy imitation data | gives exact safe-action labels under known scenarios |

The scientific split is intentional:
- **real datasets** teach glucose dynamics
- **teacher-labeled simulator runs** teach experimental action policies

That is stronger than pretending one small dataset can safely solve both problems.

## 1. Build A Better Real-Data Predictor Blend

Prepare the source datasets first:

```bash
iints research prepare-azt1d
iints research prepare-hupa
iints research prepare-ohio
```

Then blend the real training sources while preserving source-aware subject IDs:

```bash
iints research blend-datasets \
  --source azt1d=data_packs/public/azt1d/processed/azt1d_merged.csv \
  --source hupa=data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv \
  --output data_packs/processed/predictor_blend.csv \
  --manifest data_packs/processed/predictor_blend_manifest.json
```

Use OhioT1DM separately as an external benchmark rather than silently mixing every dataset together:

```bash
PYTHONPATH=src python3 research/train_predictor.py \
  --data data_packs/processed/predictor_blend.csv \
  --config research/configs/predictor_multimodal_dual_guard.yaml \
  --out models/predictor_blend

PYTHONPATH=src python3 research/evaluate_predictor.py \
  --data data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv \
  --model models/predictor_blend/predictor.pt \
  --external-data ohio=data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv \
  --reference-data data_packs/processed/predictor_blend.csv \
  --out results/predictor_blend_external_eval.json
```

## 2. Build Controller Training Data

The Jetson research runner now writes:

```text
research/
  predictor_training.csv
  controller_teacher_dataset.csv
  training_manifest.json
```

For a true 24-hour research acquisition:

```bash
iints jetson endurance start \
  --algo algorithms/example_algorithm.py \
  --duration 1d \
  --profile normal \
  --wall-clock \
  --output-dir results/jetson_research_day
```

You can combine several safe supervised runs into one controller dataset:

```bash
iints research build-control-dataset \
  --run day1=results/jetson_research_day \
  --run stress=results/jetson_stress_day \
  --output data_packs/processed/controller_teacher_dataset.csv \
  --manifest data_packs/processed/controller_teacher_manifest.json
```

## 3. Train A Local Controller

The first controller learner is intentionally simple and auditable:

```bash
iints research train-controller \
  --data data_packs/processed/controller_teacher_dataset.csv \
  --output models/controller_imitation.json \
  --metrics-output models/controller_imitation_metrics.json
```

Use it in Python research code with:

```python
from iints.core.algorithms.imitation_controller import ExperimentalImitationController

algorithm = ExperimentalImitationController(
    settings={"model_path": "models/controller_imitation.json"}
)
```

This controller is **not** presented as clinically validated. It is a baseline that proves the full local-AI research loop:

1. collect or simulate data
2. build a supervised dataset
3. train a local policy
4. run it behind the supervisor
5. compare it against rule-based baselines

When you need a stronger local model after that auditable baseline, train the PyTorch policy:

```bash
iints research train-neural-controller \
  --data data_packs/processed/controller_teacher_dataset.csv \
  --output models/controller_neural.pt \
  --metrics-output models/controller_neural_metrics.json
```

Then require held-out closed-loop evidence before treating it as a serious research candidate:

```bash
iints research evaluate-controller \
  --model models/controller_neural.pt \
  --model-kind neural \
  --output-dir results/controller_neural_eval
```

That report compares the learned controller with `ClinicalBaselineAlgorithm` on unseen presets such as `hypo_prone_night`, `hyper_challenge`, `pizza_paradox`, and `midnight_crash`.

## 4. One-Step Jetson Research Finalization

After a completed endurance run you can close the whole post-run loop in one command:

```bash
iints jetson endurance finalize-research \
  --output-dir results/jetson_research_day
```

It will:

1. train the auditable linear imitation baseline
2. train the stronger PyTorch controller
3. train the glucose predictor when the exported dataset is large enough
4. run held-out closed-loop evaluation against the clinical baseline
5. write `research/RESEARCH_PIPELINE_REPORT.md`

If you want the same work to happen automatically at the end of the endurance command, add:

```bash
--finalize-research
```

to `iints jetson endurance start`.

## 5. Multi-Run Local AI Lab

For actual AI research, one run should not be the whole story. The SDK now has a higher-level lab command that combines multiple completed Jetson or simulator bundles into one training workspace:

```bash
iints research train-local-ai \
  --run day1=results/jetson_research_day \
  --run day2=results/jetson_research_day_2 \
  --output-dir results/local_ai_lab
```

It writes:

```text
results/local_ai_lab/
  datasets/
    predictor_training.csv
    predictor_dataset_manifest.json
    controller_teacher_dataset.csv
    controller_dataset_manifest.json
    LOCAL_AI_DATASET_CARD.json
  models/
    linear_controller.json
    neural_controller.pt
    predictor/
  evaluation/
    CONTROL_EVALUATION_REPORT.md
  LOCAL_AI_RESEARCH_SUMMARY.json
  LOCAL_AI_RESEARCH_REPORT.md
```

This is the clearest workflow when you want to use long Jetson runs as training data:

1. collect one or more 24h/7d research bundles
2. merge predictor rows and controller-teacher rows with source labels
3. train an auditable linear local controller
4. optionally train the PyTorch neural controller
5. optionally train the glucose predictor
6. evaluate learned controllers on held-out scenarios

For a first smoke test without heavy PyTorch/predictor work:

```bash
iints research train-local-ai \
  --run day1=results/jetson_research_day \
  --output-dir results/local_ai_lab_smoke \
  --skip-predictor \
  --skip-neural \
  --skip-evaluation
```

The important separation is:

- `predictor_training.csv` trains a glucose forecasting model
- `controller_teacher_dataset.csv` trains a research controller policy
- `LOCAL_AI_DATASET_CARD.json` records lineage, row counts, sources, and research-only limits

The older command name `iints research local-ai-lab` remains available, but `train-local-ai` is the clearest command for new users.

## 6. Publish The Research Evidence

After training, attach the run folders and local AI lab to one evidence bundle:

```bash
iints evidence build \
  --run day1=results/jetson_research_day \
  --run day2=results/jetson_research_day_2 \
  --local-ai-dir results/local_ai_lab \
  --output-dir results/local_ai_evidence
```

This writes:

- `README.md` with run-level glucose metrics
- `MODEL_CARD.md` with research-only use, non-use, and safety gate status
- `evidence_summary.json` for machine-readable review
- `run_index.csv` for quick comparison tables

## 7. What Good Research Looks Like

Before making any strong claim, require all of the following:

- subject/source-aware splits
- external-data evaluation for predictors
- calibration and hypo-detection analysis
- held-out scenario evaluation for controllers
- comparison against deterministic baselines
- safety-supervisor intervention counts
- exact run manifests and dataset manifests

The SDK now gives you both the model layer and the first evidence layer:

- auditable linear imitation baseline
- stronger PyTorch neural controller
- held-out closed-loop controller evaluation
- automatic post-run Jetson research finalization

The next scientific step after that is not "make the model bigger"; it is stricter promotion gates, more held-out patients, and external real-data validation for every predictor claim.
