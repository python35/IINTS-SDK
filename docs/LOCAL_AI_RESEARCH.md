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

## 4. What Good Research Looks Like

Before making any strong claim, require all of the following:

- subject/source-aware splits
- external-data evaluation for predictors
- calibration and hypo-detection analysis
- held-out scenario evaluation for controllers
- comparison against deterministic baselines
- safety-supervisor intervention counts
- exact run manifests and dataset manifests

## 5. Next Technical Step

The current policy learner is an **auditable imitation baseline**. The next stronger layer is:

- a PyTorch MLP or recurrent controller trained offline
- closed-loop evaluation over held-out simulator scenarios
- uncertainty-aware rejection or fallback to baseline control
- formal promotion gates before any model can become a validated research candidate

That is the path from "local AI exists" to "local AI has evidence."
