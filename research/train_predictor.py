from __future__ import annotations

import argparse
from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
except Exception as exc:
    raise SystemExit(
        "Torch is required for training. Install with `pip install iints-sdk-python35[research]`."
    ) from exc

from iints.research.config import PredictorConfig, TrainingConfig
from iints.research.dataset import build_sequences, load_dataset
from iints.research.predictor import LSTMPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Parquet dataset path")
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    predictor_cfg = PredictorConfig(**config["predictor"])
    training_cfg = TrainingConfig(**config["training"])

    random.seed(training_cfg.seed)
    np.random.seed(training_cfg.seed)
    torch.manual_seed(training_cfg.seed)

    df = load_dataset(args.data)
    X, y = build_sequences(
        df,
        history_steps=predictor_cfg.history_steps,
        horizon_steps=predictor_cfg.horizon_steps,
        feature_columns=predictor_cfg.feature_columns,
        target_column=predictor_cfg.target_column,
    )

    tensor_x = torch.from_numpy(X)
    tensor_y = torch.from_numpy(y)
    dataset = TensorDataset(tensor_x, tensor_y)

    val_size = int(len(dataset) * training_cfg.validation_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=training_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg.batch_size)

    model = LSTMPredictor(
        input_size=X.shape[-1],
        hidden_size=training_cfg.hidden_size,
        num_layers=training_cfg.num_layers,
        dropout=training_cfg.dropout,
        horizon_steps=predictor_cfg.horizon_steps,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    criterion = nn.MSELoss()

    metrics = {"train_loss": [], "val_loss": []}
    start = time.time()
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        print(f"Epoch {epoch + 1}/{training_cfg.epochs} - train={train_loss:.4f} val={val_loss:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "predictor.pt"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "input_size": X.shape[-1],
            "hidden_size": training_cfg.hidden_size,
            "num_layers": training_cfg.num_layers,
            "dropout": training_cfg.dropout,
            "horizon_steps": predictor_cfg.horizon_steps,
            "history_steps": predictor_cfg.history_steps,
            "feature_columns": predictor_cfg.feature_columns,
            "target_column": predictor_cfg.target_column,
        },
    }
    torch.save(payload, model_path)

    report = {
        "duration_sec": time.time() - start,
        "epochs": training_cfg.epochs,
        "train_loss_final": metrics["train_loss"][-1],
        "val_loss_final": metrics["val_loss"][-1],
        "metrics": metrics,
    }
    (args.out / "training_report.json").write_text(json.dumps(report, indent=2))

    print(f"Saved model: {model_path}")
    print(f"Saved report: {args.out / 'training_report.json'}")


if __name__ == "__main__":
    main()
