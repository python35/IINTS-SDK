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
