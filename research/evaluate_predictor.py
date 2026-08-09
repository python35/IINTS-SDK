from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "Torch is required for evaluation. Install with `pip install iints-sdk-python35[research]`."
    ) from exc

from iints.research.config import PredictorConfig, TrainingConfig
from iints.research.dataset import FeatureScaler, build_sequences, compute_dataset_lineage, load_dataset
from iints.research.evaluation import (
    feature_drift_report,
    hypoglycemia_detection_report,
    subgroup_error_report,
    uncertainty_reliability_report,
)
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
        "--external-data",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Evaluate an additional held-out dataset. Repeat for multiple external datasets.",
    )
    parser.add_argument(
        "--reference-data",
        required=False,
        type=Path,
        help="Reference dataset used for raw feature drift checks against --data.",
    )
    parser.add_argument(
        "--subgroup-column",
        action="append",
        default=[],
        help="Column to use for subgroup performance. Repeat for multiple columns.",
    )
    parser.add_argument(
        "--plots-dir",
        required=False,
        type=Path,
        help="Optional directory for calibration/reliability plot PNGs.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=0,
        help="If > 0, run MC Dropout inference with this many samples and report uncertainty.",
    )
    return parser.parse_args()


def _apply_meal_announcement(
    df,
    predictor_cfg: PredictorConfig,
    training_cfg: TrainingConfig,
):
    """
    Rebuild the optional pre-announced meal feature used during training.
    """
    minutes = training_cfg.meal_announcement_minutes
    feature = training_cfg.meal_announcement_feature
    if feature not in predictor_cfg.feature_columns:
        return df

    # If the feature was part of training but no reconstruction settings are
    # available, fall back to zeros (common in datasets without meal announcements).
    if minutes is None:
        if feature not in df.columns:
            df[feature] = 0.0
        return df

    source = training_cfg.meal_announcement_column
    if source not in df.columns:
        df[feature] = 0.0
        return df

    shift_steps = int(round(minutes / predictor_cfg.time_step_minutes))
    if shift_steps <= 0:
        return df

    group_cols = []
    if "subject_id" in df.columns:
        group_cols.append("subject_id")
    if "segment" in df.columns:
        group_cols.append("segment")

    sort_cols = [c for c in (*group_cols, "time_minutes") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    if group_cols:
        df[feature] = (
            df.groupby(group_cols, observed=False)[source]
            .shift(-shift_steps)
            .fillna(0.0)
        )
    else:
        df[feature] = df[source].shift(-shift_steps).fillna(0.0)
    return df


def _parse_external_data(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("--external-data must use LABEL=PATH format.")
        label, raw_path = value.split("=", 1)
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("--external-data label cannot be empty.")
        parsed.append((clean_label, Path(raw_path).expanduser()))
    return parsed


def _sequence_target_labels(
    df,
    *,
    history_steps: int,
    horizon_steps: int,
    column: str,
    subject_column: str = "subject_id",
    segment_column: str | None = None,
) -> np.ndarray:
    if column not in df.columns:
        raise ValueError(f"Subgroup column '{column}' not found in dataset.")
    boundary = np.zeros(len(df), dtype=bool)
    if len(df) > 0:
        boundary[0] = True
    if subject_column in df.columns:
        boundary |= (df[subject_column] != df[subject_column].shift(1)).to_numpy(dtype=bool)
    if segment_column and segment_column in df.columns:
        boundary |= (df[segment_column] != df[segment_column].shift(1)).to_numpy(dtype=bool)

    labels: list[Any] = []
    end_index = len(df) - history_steps - horizon_steps + 1
    for idx in range(end_index):
        window_end = idx + history_steps
        horizon_end = window_end + horizon_steps
        if boundary[idx + 1 : horizon_end].any():
            continue
        labels.extend(df[column].iloc[window_end:horizon_end].tolist())
    return np.asarray(labels, dtype=object)


def _write_reliability_plot(report: dict[str, Any], path: Path, *, title: str) -> None:
    bins = report.get("bins", [])
    if not bins:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = [str(row["bin"]) for row in bins]
    coverage = [row["interval_95_coverage_pct"] for row in bins]
    target = [report["target_coverage_pct"]] * len(bins)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax.bar(x, coverage, color="#0F766E", alpha=0.88, label="Observed heuristic coverage")
    ax.plot(x, target, color="#DC2626", linestyle="--", linewidth=1.5, label="Gaussian reference")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Predicted-uncertainty bin")
    ax.set_ylabel("Coverage of prediction +/-1.96 model SD (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _evaluate_loaded_dataset(
    *,
    df,
    source_path: Path,
    label: str,
    model,
    model_cfg: dict[str, Any],
    predictor_cfg: PredictorConfig,
    training_cfg: TrainingConfig,
    subgroup_columns: list[str],
    mc_samples: int,
    plots_dir: Path | None,
) -> dict[str, Any]:
    prepared_df = _apply_meal_announcement(df.copy(), predictor_cfg, training_cfg)
    X, y = build_sequences(
        prepared_df,
        history_steps=predictor_cfg.history_steps,
        horizon_steps=predictor_cfg.horizon_steps,
        feature_columns=predictor_cfg.feature_columns,
        target_column=predictor_cfg.target_column,
    )
    baselines = evaluate_baselines(
        X,
        y,
        horizon_steps=predictor_cfg.horizon_steps,
        time_step_minutes=predictor_cfg.time_step_minutes,
        feature_columns=predictor_cfg.feature_columns,
    )

    scaler_data = model_cfg.get("scaler")
    if scaler_data:
        scaler = FeatureScaler.from_dict(scaler_data)
        expected_features = len(np.asarray(scaler_data.get("center", []), dtype=float))
        if expected_features and expected_features != X.shape[2]:
            raise ValueError(
                "Scaler/feature mismatch: checkpoint scaler expects "
                f"{expected_features} features but input has {X.shape[2]}. "
                "Re-evaluate with data prepared using the same feature pipeline as training."
            )
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_scaled))
    preds_np = preds.numpy()
    flat_y = y.reshape(-1)
    flat_pred = preds_np.reshape(-1)
    summary = regression_metrics(y, preds_np)
    payload: dict[str, Any] = {
        "label": label,
        "lineage": compute_dataset_lineage(prepared_df, source_path=source_path),
        "lstm": {
            "mae": summary["mae"],
            "rmse": summary["rmse"],
            "bias": summary["bias"],
            "bands": band_regression_metrics(y, preds_np),
            "hypoglycemia_detection": hypoglycemia_detection_report(flat_y, flat_pred),
        },
        "baselines": baselines,
    }

    subgroup_payload: dict[str, Any] = {}
    for column in subgroup_columns:
        labels = _sequence_target_labels(
            prepared_df,
            history_steps=predictor_cfg.history_steps,
            horizon_steps=predictor_cfg.horizon_steps,
            column=column,
        )
        subgroup_payload[column] = subgroup_error_report(flat_y, flat_pred, labels)
    if subgroup_payload:
        payload["subgroups"] = subgroup_payload

    if mc_samples > 0:
        tensor_x = torch.from_numpy(X_scaled)
        mean_t, std_t = model.predict_with_uncertainty(tensor_x, n_samples=mc_samples)
        mean_np = mean_t.numpy()
        std_np = std_t.numpy()
        flat_mean = mean_np.reshape(-1)
        flat_std = std_np.reshape(-1)
        mean_summary = regression_metrics(y, mean_np)
        reliability = uncertainty_reliability_report(flat_y, flat_mean, flat_std)
        payload["mc_dropout"] = {
            "n_samples": mc_samples,
            "mean_mae": mean_summary["mae"],
            "mean_rmse": mean_summary["rmse"],
            "mean_bias": mean_summary["bias"],
            "bands": band_regression_metrics(y, mean_np),
            "mean_std": float(std_np.mean()),
            "max_std": float(std_np.max()),
            "calibration_95": interval_coverage_metrics(y, mean_np, std_np, confidence=0.95),
            "uncertainty_reliability": reliability,
            "hypoglycemia_detection": hypoglycemia_detection_report(flat_y, flat_mean),
        }
        if subgroup_payload:
            payload["mc_dropout"]["subgroups"] = {
                column: subgroup_error_report(
                    flat_y,
                    flat_mean,
                    _sequence_target_labels(
                        prepared_df,
                        history_steps=predictor_cfg.history_steps,
                        horizon_steps=predictor_cfg.horizon_steps,
                        column=column,
                    ),
                    predicted_std=flat_std,
                )
                for column in subgroup_columns
            }
        if plots_dir is not None:
            _write_reliability_plot(
                reliability,
                plots_dir / f"{label}_uncertainty_reliability.png",
                title=f"{label}: uncertainty reliability",
            )

    return payload


def main() -> None:
    args = parse_args()
    model, model_cfg = load_predictor(args.model)
    predictor_cfg = PredictorConfig(
        history_minutes=int(model_cfg["history_steps"]) * model_cfg.get("time_step_minutes", 5),
        horizon_minutes=int(model_cfg["horizon_steps"]) * model_cfg.get("time_step_minutes", 5),
        time_step_minutes=model_cfg.get("time_step_minutes", 5),
        feature_columns=model_cfg["feature_columns"],
        target_column=model_cfg["target_column"],
    )
    training_cfg = TrainingConfig()

    if args.config:
        cfg = yaml.safe_load(args.config.read_text()) or {}
        if isinstance(cfg.get("training"), dict):
            training_cfg = TrainingConfig(**cfg["training"])
        if isinstance(cfg.get("predictor"), dict):
            requested_cfg = PredictorConfig(**cfg["predictor"])
            # Evaluation must match checkpoint architecture/features.
            if (
                requested_cfg.feature_columns != predictor_cfg.feature_columns
                or requested_cfg.history_steps != predictor_cfg.history_steps
                or requested_cfg.horizon_steps != predictor_cfg.horizon_steps
            ):
                print(
                    "WARNING: --config predictor settings differ from checkpoint; "
                    "using checkpoint feature/history/horizon to avoid invalid evaluation."
                )

    subgroup_columns = list(dict.fromkeys(args.subgroup_column))
    primary_df = load_dataset(args.data)
    primary = _evaluate_loaded_dataset(
        df=primary_df,
        source_path=args.data,
        label="primary",
        model=model,
        model_cfg=model_cfg,
        predictor_cfg=predictor_cfg,
        training_cfg=training_cfg,
        subgroup_columns=subgroup_columns,
        mc_samples=args.mc_samples,
        plots_dir=args.plots_dir,
    )
    metrics: dict[str, Any] = {
        "model_config": {
            "history_steps": predictor_cfg.history_steps,
            "horizon_steps": predictor_cfg.horizon_steps,
            "time_step_minutes": predictor_cfg.time_step_minutes,
        },
        "lineage": primary["lineage"],
        "lstm": primary["lstm"],
        "baselines": primary["baselines"],
    }
    if "mc_dropout" in primary:
        metrics["mc_dropout"] = primary["mc_dropout"]
    if "subgroups" in primary:
        metrics["subgroups"] = primary["subgroups"]

    external_datasets: dict[str, Any] = {}
    for label, path in _parse_external_data(args.external_data):
        external_datasets[label] = _evaluate_loaded_dataset(
            df=load_dataset(path),
            source_path=path,
            label=label,
            model=model,
            model_cfg=model_cfg,
            predictor_cfg=predictor_cfg,
            training_cfg=training_cfg,
            subgroup_columns=subgroup_columns,
            mc_samples=args.mc_samples,
            plots_dir=args.plots_dir,
        )
    if external_datasets:
        metrics["external_datasets"] = external_datasets

    if args.reference_data:
        reference_df = _apply_meal_announcement(
            load_dataset(args.reference_data),
            predictor_cfg,
            training_cfg,
        )
        candidate_df = _apply_meal_announcement(primary_df.copy(), predictor_cfg, training_cfg)
        metrics["feature_drift"] = feature_drift_report(
            reference_df[predictor_cfg.feature_columns].to_numpy(dtype=float),
            candidate_df[predictor_cfg.feature_columns].to_numpy(dtype=float),
            feature_names=predictor_cfg.feature_columns,
        )

    # Print comparison table
    print("\n=== Evaluation Results ===")
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8}")
    print("-" * 38)
    for bname, bm in primary["baselines"].items():
        print(f"{bname:<20} {bm['mae']:>8.3f} {bm['rmse']:>8.3f}")
    print(f"{'LSTM':<20} {primary['lstm']['mae']:>8.3f} {primary['lstm']['rmse']:>8.3f}")
    if "mc_dropout" in metrics:
        mcd = metrics["mc_dropout"]
        print(f"\nMC Dropout ({args.mc_samples} samples):")
        print(f"  Mean MAE : {mcd['mean_mae']:.3f}")
        print(f"  Mean RMSE: {mcd['mean_rmse']:.3f}")
        print(f"  Mean std : {mcd['mean_std']:.3f}")
    if external_datasets:
        print("\nExternal datasets:")
        for label, payload in external_datasets.items():
            print(
                f"  {label:<16} "
                f"MAE {payload['lstm']['mae']:.3f}  RMSE {payload['lstm']['rmse']:.3f}"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved metrics: {args.out}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
