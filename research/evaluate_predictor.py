from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "Torch is required for evaluation. Install with `pip install iints-sdk-python35[research]`."
    ) from exc

from iints.research.config import PredictorConfig
from iints.research.dataset import build_sequences, load_dataset
from iints.research.predictor import load_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Parquet dataset path")
    parser.add_argument("--model", required=True, type=Path, help="Model checkpoint (.pt)")
    parser.add_argument("--config", required=False, type=Path, help="Optional config YAML")
    parser.add_argument("--out", required=False, type=Path, help="Output metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, model_cfg = load_predictor(args.model)

    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        predictor_cfg = PredictorConfig(**cfg["predictor"])
    else:
        predictor_cfg = PredictorConfig(
            history_minutes=int(model_cfg["history_steps"]) * model_cfg.get("time_step_minutes", 5),
            horizon_minutes=int(model_cfg["horizon_steps"]) * model_cfg.get("time_step_minutes", 5),
            time_step_minutes=model_cfg.get("time_step_minutes", 5),
            feature_columns=model_cfg["feature_columns"],
            target_column=model_cfg["target_column"],
        )

    df = load_dataset(args.data)
    X, y = build_sequences(
        df,
        history_steps=predictor_cfg.history_steps,
        horizon_steps=predictor_cfg.horizon_steps,
        feature_columns=predictor_cfg.feature_columns,
        target_column=predictor_cfg.target_column,
    )

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X))
    preds_np = preds.numpy()

    mae = float(np.mean(np.abs(preds_np - y)))
    rmse = float(np.sqrt(np.mean((preds_np - y) ** 2)))

    metrics = {"mae": mae, "rmse": rmse}
    if args.out:
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics: {args.out}")
    else:
        print(metrics)


if __name__ == "__main__":
    main()
