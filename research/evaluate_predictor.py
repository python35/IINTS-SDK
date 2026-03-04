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
from iints.research.dataset import FeatureScaler, build_sequences, compute_dataset_lineage, load_dataset
from iints.research.metrics import (
    band_regression_metrics,
    interval_coverage_metrics,
    regression_metrics,
)
from iints.research.predictor import evaluate_baselines, load_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an LSTM glucose predictor and compare against baselines."
    )
    parser.add_argument("--data", required=True, type=Path, help="Dataset path (CSV or Parquet)")
    parser.add_argument("--model", required=True, type=Path, help="Model checkpoint (.pt)")
    parser.add_argument("--config", required=False, type=Path, help="Optional config YAML")
    parser.add_argument("--out", required=False, type=Path, help="Output metrics JSON")
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=0,
        help="If > 0, run MC Dropout inference with this many samples and report uncertainty.",
    )
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

    # P3-11: Evaluate baselines on raw (unscaled) X for interpretable comparison
    baselines = evaluate_baselines(
        X, y,
        horizon_steps=predictor_cfg.horizon_steps,
        time_step_minutes=predictor_cfg.time_step_minutes,
    )

    # Apply scaler from checkpoint if present
    scaler_data = model_cfg.get("scaler")
    if scaler_data:
        scaler = FeatureScaler.from_dict(scaler_data)
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_scaled))
    preds_np = preds.numpy()

    summary = regression_metrics(y, preds_np)
    band_metrics = band_regression_metrics(y, preds_np)
    lineage = compute_dataset_lineage(df, source_path=args.data)

    metrics: dict = {
        "model_config": {
            "history_steps": predictor_cfg.history_steps,
            "horizon_steps": predictor_cfg.horizon_steps,
            "time_step_minutes": predictor_cfg.time_step_minutes,
        },
        "lineage": lineage,
        "lstm": {
            "mae": summary["mae"],
            "rmse": summary["rmse"],
            "bias": summary["bias"],
            "bands": band_metrics,
        },
        "baselines": baselines,
    }

    # P3-12: Optional MC Dropout uncertainty
    if args.mc_samples > 0:
        tensor_x = torch.from_numpy(X_scaled)
        mean_t, std_t = model.predict_with_uncertainty(tensor_x, n_samples=args.mc_samples)
        mean_np = mean_t.numpy()
        std_np = std_t.numpy()
        mean_summary = regression_metrics(y, mean_np)
        mean_band_metrics = band_regression_metrics(y, mean_np)
        calibration = interval_coverage_metrics(y, mean_np, std_np, confidence=0.95)
        metrics["mc_dropout"] = {
            "n_samples": args.mc_samples,
            "mean_mae": mean_summary["mae"],
            "mean_rmse": mean_summary["rmse"],
            "mean_bias": mean_summary["bias"],
            "bands": mean_band_metrics,
            "mean_std": float(std_np.mean()),
            "max_std": float(std_np.max()),
            "calibration_95": calibration,
        }

    # Print comparison table
    print("\n=== Evaluation Results ===")
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8}")
    print("-" * 38)
    for bname, bm in baselines.items():
        print(f"{bname:<20} {bm['mae']:>8.3f} {bm['rmse']:>8.3f}")
    print(f"{'LSTM':<20} {summary['mae']:>8.3f} {summary['rmse']:>8.3f}")
    if "mc_dropout" in metrics:
        mcd = metrics["mc_dropout"]
        print(f"\nMC Dropout ({args.mc_samples} samples):")
        print(f"  Mean MAE : {mcd['mean_mae']:.3f}")
        print(f"  Mean RMSE: {mcd['mean_rmse']:.3f}")
        print(f"  Mean std : {mcd['mean_std']:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved metrics: {args.out}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
