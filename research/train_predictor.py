from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:
    raise SystemExit(
        "Torch is required for training. Install with `pip install iints-sdk-python35[research]`."
    ) from exc

import iints
from iints.research.config import PredictorConfig, TrainingConfig
from iints.research.dataset import (
    FeatureScaler,
    build_sequences,
    load_dataset,
    subject_split,
)
from iints.research.predictor import LSTMPredictor, evaluate_baselines
from iints.research.losses import QuantileLoss, SafetyWeightedMSE, BandWeightedMSE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM glucose predictor on the AZT1D / AIDE dataset."
    )
    parser.add_argument("--data", required=True, type=Path, help="Dataset path (CSV or Parquet)")
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Optional path to a pretrained predictor.pt to initialize weights",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to model registry JSON (default: <out>/../registry.json)",
    )
    return parser.parse_args()


def _apply_meal_announcement(
    df: pd.DataFrame,
    predictor_cfg: PredictorConfig,
    training_cfg: TrainingConfig,
) -> pd.DataFrame:
    """Optionally create a pre-announced meal feature (clinical practice)."""
    minutes = training_cfg.meal_announcement_minutes
    feature = training_cfg.meal_announcement_feature
    if minutes is None or feature not in predictor_cfg.feature_columns:
        return df
    source = training_cfg.meal_announcement_column
    if source not in df.columns:
        print(
            f"WARNING: Meal announcement enabled but source column '{source}' "
            "not found. Feature will be filled with zeros."
        )
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




# ---------------------------------------------------------------------------
# P1-5: Reproducibility helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# P1-6: Model registry
# ---------------------------------------------------------------------------

def _update_registry(registry_path: Path, entry: dict) -> None:
    """Append a training run entry to the model registry JSON."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        try:
            runs = json.loads(registry_path.read_text())
            if not isinstance(runs, list):
                runs = []
        except (json.JSONDecodeError, OSError):
            runs = []
    else:
        runs = []
    runs.append(entry)
    registry_path.write_text(json.dumps(runs, indent=2))


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(model: LSTMPredictor, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = model(batch_x)
            total += nn.functional.mse_loss(preds, batch_y).item()
    return total / max(1, len(loader))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config_text = args.config.read_text()
    config = yaml.safe_load(config_text)
    predictor_cfg = PredictorConfig(**config["predictor"])
    training_cfg = TrainingConfig(**config["training"])

    random.seed(training_cfg.seed)
    np.random.seed(training_cfg.seed)
    torch.manual_seed(training_cfg.seed)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df = load_dataset(args.data)
    df = _apply_meal_announcement(df, predictor_cfg, training_cfg)

    # P1-5: Compute dataset hash for reproducibility tracking
    data_sha256 = _sha256_file(args.data)
    config_sha256 = _sha256_str(config_text)

    # -----------------------------------------------------------------------
    # P0-2: Subject-level split
    # -----------------------------------------------------------------------
    if training_cfg.subject_level_split and "subject_id" in df.columns:
        min_rows = predictor_cfg.history_steps + predictor_cfg.horizon_steps + 1
        counts = df.groupby("subject_id").size()
        eligible = counts[counts >= min_rows].index
        dropped = sorted(set(counts.index) - set(eligible))
        if dropped:
            print(f"Dropping {len(dropped)} subjects with < {min_rows} rows: {dropped}")
        df = df[df["subject_id"].isin(eligible)].reset_index(drop=True)
        if df["subject_id"].nunique() < 3:
            print("WARNING: Too few eligible subjects for subject-level split; using row-level split.")
            training_cfg.subject_level_split = False

    if training_cfg.subject_level_split and "subject_id" in df.columns:
        train_df, val_df, test_df = subject_split(
            df,
            val_fraction=training_cfg.validation_split,
            test_fraction=training_cfg.test_split,
            subject_column="subject_id",
            seed=training_cfg.seed,
        )
        train_subjects = sorted(train_df["subject_id"].unique().tolist())
        val_subjects = sorted(val_df["subject_id"].unique().tolist())
        test_subjects = sorted(test_df["subject_id"].unique().tolist())
        print(
            f"Subject-level split — train: {len(train_subjects)} subjects, "
            f"val: {len(val_subjects)} subjects, "
            f"test: {len(test_subjects)} subjects"
        )
    else:
        # Fallback: row-level random split (no subject column or explicitly disabled)
        print("WARNING: Using row-level random split. "
              "Set subject_level_split=true and ensure 'subject_id' column is present "
              "to avoid data leakage.")
        rng = np.random.default_rng(training_cfg.seed)
        idx = rng.permutation(len(df))
        n_test = max(1, round(len(df) * training_cfg.test_split))
        n_val = max(1, round(len(df) * training_cfg.validation_split))
        test_df = df.iloc[idx[:n_test]].reset_index(drop=True)
        val_df = df.iloc[idx[n_test: n_test + n_val]].reset_index(drop=True)
        train_df = df.iloc[idx[n_test + n_val:]].reset_index(drop=True)
        train_subjects = val_subjects = test_subjects = []

    # -----------------------------------------------------------------------
    # Build sequences (boundary-safe)
    # -----------------------------------------------------------------------
    segment_column = "segment" if "segment" in df.columns else None
    try:
        X_train, y_train = build_sequences(
            train_df,
            history_steps=predictor_cfg.history_steps,
            horizon_steps=predictor_cfg.horizon_steps,
            feature_columns=predictor_cfg.feature_columns,
            target_column=predictor_cfg.target_column,
            segment_column=segment_column,
        )
        X_val, y_val = build_sequences(
            val_df,
            history_steps=predictor_cfg.history_steps,
            horizon_steps=predictor_cfg.horizon_steps,
            feature_columns=predictor_cfg.feature_columns,
            target_column=predictor_cfg.target_column,
            segment_column=segment_column,
        )
        X_test, y_test = build_sequences(
            test_df,
            history_steps=predictor_cfg.history_steps,
            horizon_steps=predictor_cfg.horizon_steps,
            feature_columns=predictor_cfg.feature_columns,
            target_column=predictor_cfg.target_column,
            segment_column=segment_column,
        )
    except ValueError as exc:
        if training_cfg.subject_level_split:
            print(f"WARNING: subject-level split yielded insufficient sequences ({exc}). Falling back to row-level split.")
            rng = np.random.default_rng(training_cfg.seed)
            idx = rng.permutation(len(df))
            n_test = max(1, round(len(df) * training_cfg.test_split))
            n_val = max(1, round(len(df) * training_cfg.validation_split))
            test_df = df.iloc[idx[:n_test]].reset_index(drop=True)
            val_df = df.iloc[idx[n_test: n_test + n_val]].reset_index(drop=True)
            train_df = df.iloc[idx[n_test + n_val:]].reset_index(drop=True)
            X_train, y_train = build_sequences(
                train_df,
                history_steps=predictor_cfg.history_steps,
                horizon_steps=predictor_cfg.horizon_steps,
                feature_columns=predictor_cfg.feature_columns,
                target_column=predictor_cfg.target_column,
                segment_column=segment_column,
            )
            X_val, y_val = build_sequences(
                val_df,
                history_steps=predictor_cfg.history_steps,
                horizon_steps=predictor_cfg.horizon_steps,
                feature_columns=predictor_cfg.feature_columns,
                target_column=predictor_cfg.target_column,
                segment_column=segment_column,
            )
            X_test, y_test = build_sequences(
                test_df,
                history_steps=predictor_cfg.history_steps,
                horizon_steps=predictor_cfg.horizon_steps,
                feature_columns=predictor_cfg.feature_columns,
                target_column=predictor_cfg.target_column,
                segment_column=segment_column,
            )
        else:
            raise

    # -----------------------------------------------------------------------
    # P3-10: Feature normalisation — fit on TRAINING data only
    # -----------------------------------------------------------------------
    scaler = FeatureScaler(strategy=training_cfg.normalization)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print(f"Normalisation: {training_cfg.normalization}")

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=training_cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=training_cfg.batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=training_cfg.batch_size,
    )

    # -----------------------------------------------------------------------
    # Model + loss
    # -----------------------------------------------------------------------
    model = LSTMPredictor(
        input_size=X_train.shape[-1],
        hidden_size=training_cfg.hidden_size,
        num_layers=training_cfg.num_layers,
        dropout=training_cfg.dropout,
        horizon_steps=predictor_cfg.horizon_steps,
    )

    # Optional warm-start from a pretrained checkpoint (same input size & horizon)
    if args.warm_start:
        ckpt = torch.load(args.warm_start, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
        if ckpt_cfg:
            if ckpt_cfg.get("input_size") != X_train.shape[-1]:
                raise ValueError(
                    f"Warm-start input size mismatch: checkpoint={ckpt_cfg.get('input_size')}, "
                    f"current={X_train.shape[-1]}"
                )
            if ckpt_cfg.get("horizon_steps") != predictor_cfg.horizon_steps:
                raise ValueError(
                    f"Warm-start horizon mismatch: checkpoint={ckpt_cfg.get('horizon_steps')}, "
                    f"current={predictor_cfg.horizon_steps}"
                )
        model.load_state_dict(state, strict=True)
        print(f"Warm-start: loaded weights from {args.warm_start}")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )

    # P3-12: Choose loss function
    if training_cfg.loss == "quantile":
        q = training_cfg.quantile
        if q is None:
            raise ValueError("training.quantile must be set when training.loss == 'quantile'")
        criterion: nn.Module = QuantileLoss(quantile=q)
        print(f"Loss: quantile (q={q})")
    elif training_cfg.loss == "safety_weighted":
        criterion = SafetyWeightedMSE(
            low_threshold=training_cfg.safety_weighted_low_threshold,
            alpha=training_cfg.safety_weighted_alpha,
            max_weight=training_cfg.safety_weighted_max_weight,
        )
        print(
            "Loss: safety_weighted "
            f"(low<{training_cfg.safety_weighted_low_threshold}, "
            f"alpha={training_cfg.safety_weighted_alpha}, "
            f"max_weight={training_cfg.safety_weighted_max_weight})"
        )
    elif training_cfg.loss == "band_weighted":
        criterion = BandWeightedMSE(
            low_threshold=training_cfg.band_weighted_low_threshold,
            high_threshold=training_cfg.band_weighted_high_threshold,
            low_weight=training_cfg.band_weighted_low_weight,
            high_weight=training_cfg.band_weighted_high_weight,
            max_weight=training_cfg.band_weighted_max_weight,
        )
        print(
            "Loss: band_weighted "
            f"(low<{training_cfg.band_weighted_low_threshold}, "
            f"high>{training_cfg.band_weighted_high_threshold}, "
            f"low_w={training_cfg.band_weighted_low_weight}, "
            f"high_w={training_cfg.band_weighted_high_weight}, "
            f"max_w={training_cfg.band_weighted_max_weight})"
        )
    else:
        criterion = nn.MSELoss()
        print("Loss: MSE")

    # -----------------------------------------------------------------------
    # Optional layer freezing (fine-tune)
    # -----------------------------------------------------------------------
    if training_cfg.freeze_lstm_layers > 0:
        freeze_n = min(training_cfg.freeze_lstm_layers, training_cfg.num_layers)
        for name, param in model.lstm.named_parameters():
            for layer_idx in range(freeze_n):
                if f"_l{layer_idx}" in name:
                    param.requires_grad = False
        print(f"Freeze: first {freeze_n} LSTM layer(s)")
        # Rebuild optimizer to respect requires_grad flags
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    metrics: dict = {"train_loss": [], "val_loss": []}
    start = time.time()
    best_val = float("inf")
    best_epoch = -1
    best_state: dict | None = None
    patience = 0

    for epoch in range(training_cfg.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        val_loss = _evaluate(model, val_loader)

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        print(f"Epoch {epoch + 1}/{training_cfg.epochs} — train={train_loss:.4f} val={val_loss:.4f}")

        # Early stopping
        if training_cfg.early_stopping_patience > 0:
            if val_loss < (best_val - training_cfg.early_stopping_min_delta):
                best_val = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= training_cfg.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best epoch {best_epoch}, best val {best_val:.4f})"
                    )
                    break

    # -----------------------------------------------------------------------
    # Held-out test evaluation
    # -----------------------------------------------------------------------
    # Restore best model if early stopping captured a better epoch
    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse = _evaluate(model, test_loader)
    test_mae = float(
        np.mean(
            [
                np.abs(
                    model(bx).detach().numpy() - by.numpy()
                ).mean()
                for bx, by in test_loader
            ]
        )
    )

    # -----------------------------------------------------------------------
    # P3-11: Baseline comparison on test set
    # -----------------------------------------------------------------------
    baselines = evaluate_baselines(
        X_test,
        y_test,
        horizon_steps=predictor_cfg.horizon_steps,
        time_step_minutes=predictor_cfg.time_step_minutes,
    )
    print("\nBaseline comparison (test set):")
    for bname, bmetrics in baselines.items():
        print(f"  {bname}: MAE={bmetrics['mae']:.2f}, RMSE={bmetrics['rmse']:.2f}")
    lstm_rmse = float(np.sqrt(test_mse))
    print(f"  LSTM   : MAE={test_mae:.2f}, RMSE={lstm_rmse:.2f}")

    # -----------------------------------------------------------------------
    # P1-6: Save model checkpoint with scaler embedded
    # -----------------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "predictor.pt"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "input_size": X_train.shape[-1],
            "hidden_size": training_cfg.hidden_size,
            "num_layers": training_cfg.num_layers,
            "dropout": training_cfg.dropout,
            "horizon_steps": predictor_cfg.horizon_steps,
            "history_steps": predictor_cfg.history_steps,
            "time_step_minutes": predictor_cfg.time_step_minutes,
            "feature_columns": predictor_cfg.feature_columns,
            "target_column": predictor_cfg.target_column,
            "scaler": scaler.to_dict(),   # P3-10: embed scaler for inference
        },
    }
    torch.save(payload, model_path)

    # -----------------------------------------------------------------------
    # P1-5: Training report with full reproducibility metadata
    # -----------------------------------------------------------------------
    duration = time.time() - start
    report = {
        # Run provenance
        "sdk_version": getattr(iints, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": training_cfg.seed,
        # Dataset
        "data_path": str(args.data),
        "data_sha256": data_sha256,
        "config_path": str(args.config),
        "config_sha256": config_sha256,
        "warm_start": str(args.warm_start) if args.warm_start else None,
        # Split
        "subject_level_split": training_cfg.subject_level_split,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val)),
        "test_sequences": int(len(X_test)),
        # Training
        "duration_sec": duration,
        "epochs": training_cfg.epochs,
        "normalization": training_cfg.normalization,
        "loss_fn": training_cfg.loss,
        "weight_decay": training_cfg.weight_decay,
        "freeze_lstm_layers": training_cfg.freeze_lstm_layers,
        "early_stopping_patience": training_cfg.early_stopping_patience,
        "early_stopping_min_delta": training_cfg.early_stopping_min_delta,
        "best_epoch": best_epoch if best_epoch > 0 else None,
        "train_loss_final": metrics["train_loss"][-1],
        "val_loss_final": metrics["val_loss"][-1],
        "metrics": metrics,
        # Evaluation
        "test_mse": float(test_mse),
        "test_rmse": lstm_rmse,
        "test_mae": test_mae,
        "baselines": baselines,
    }
    report_path = args.out / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # -----------------------------------------------------------------------
    # P1-6: Update model registry
    # -----------------------------------------------------------------------
    registry_path = args.registry or args.out.parent / "registry.json"
    registry_entry = {
        "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "model_path": str(model_path.resolve()),
        "sdk_version": report["sdk_version"],
        "data_sha256": data_sha256,
        "config_sha256": config_sha256,
        "seed": training_cfg.seed,
        "epochs": training_cfg.epochs,
        "val_loss_final": metrics["val_loss"][-1],
        "test_rmse": lstm_rmse,
        "test_mae": test_mae,
        "normalization": training_cfg.normalization,
        "timestamp_utc": report["timestamp_utc"],
    }
    _update_registry(registry_path, registry_entry)

    print(f"\nSaved model       : {model_path}")
    print(f"Saved report      : {report_path}")
    print(f"Updated registry  : {registry_path}")


if __name__ == "__main__":
    main()
