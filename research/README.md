# IINTS-AF Research Track (AI Predictor)

This folder contains the **AI research pipeline** for training a glucose predictor model
that plugs into the IINTS-AF safety workflow.

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

## Training
```bash
python research/train_predictor.py --data data/training.parquet --config research/configs/predictor.yaml --out models
```

## Evaluation
```bash
python research/evaluate_predictor.py --data data/validation.parquet --model models/predictor.pt
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
