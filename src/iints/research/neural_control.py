from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from iints.research.control import CONTROL_FEATURE_COLUMNS, CONTROL_TARGET_COLUMN, evaluate_controller_predictions

try:  # pragma: no cover - exercised through the explicit import guard below
    import torch as torch_mod
    from torch import nn as torch_nn
    from torch.utils.data import DataLoader as torch_data_loader_cls
    from torch.utils.data import TensorDataset as torch_tensor_dataset_cls
except Exception:  # pragma: no cover - keeps the core SDK importable without research extras
    torch_mod = None  # type: ignore[assignment]
    torch_nn = None  # type: ignore[assignment]
    torch_data_loader_cls = None  # type: ignore[assignment,misc]
    torch_tensor_dataset_cls = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class NeuralControllerConfig:
    hidden_sizes: Tuple[int, ...] = (32, 16)
    learning_rate: float = 1e-3
    epochs: int = 120
    batch_size: int = 128
    validation_fraction: float = 0.2
    seed: int = 42
    max_output_units: float = 5.0
    hypo_loss_weight: float = 2.0


def _require_torch() -> Any:
    if (
        torch_mod is None
        or torch_nn is None
        or torch_data_loader_cls is None
        or torch_tensor_dataset_cls is None
    ):
        raise RuntimeError(
            "Torch is required for neural controller training. "
            "Install with `pip install iints-sdk-python35[research]`."
        )
    return torch_mod


def _controller_net(input_size: int, hidden_sizes: Sequence[int]) -> Any:
    torch_mod = _require_torch()
    layers: list[Any] = []
    previous = input_size
    for hidden_size in hidden_sizes:
        layers.extend([torch_nn.Linear(previous, int(hidden_size)), torch_nn.ReLU()])
        previous = int(hidden_size)
    layers.extend([torch_nn.Linear(previous, 1), torch_nn.Softplus()])
    return torch_nn.Sequential(*layers).to(dtype=torch_mod.float32)


def _prepare_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[CONTROL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df[CONTROL_TARGET_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    glucose = pd.to_numeric(df["glucose_actual_mgdl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return X, y, glucose


def instantiate_neural_controller_model(payload: Dict[str, Any]) -> Any:
    model = _controller_net(
        len(payload["feature_columns"]),
        tuple(int(size) for size in payload["hidden_sizes"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _predict_from_matrix(payload: Dict[str, Any], model: Any, X: np.ndarray) -> np.ndarray:
    torch_mod = _require_torch()
    mean = np.asarray(payload["feature_mean"], dtype=float)
    std = np.asarray(payload["feature_std"], dtype=float)
    scaled = (X - mean) / std
    with torch_mod.no_grad():
        tensor = torch_mod.tensor(scaled, dtype=torch_mod.float32)
        predictions = model(tensor).reshape(-1).cpu().numpy()
    return np.clip(predictions, 0.0, float(payload["max_output_units"]))


def train_neural_imitation_controller(
    df: pd.DataFrame,
    *,
    config: NeuralControllerConfig | None = None,
) -> Dict[str, Any]:
    torch_mod = _require_torch()
    cfg = config or NeuralControllerConfig()
    X, y, glucose = _prepare_arrays(df)
    if len(X) < 8:
        raise ValueError("Need at least eight rows to train a neural controller.")
    if not 0.0 <= cfg.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0).")

    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(len(X))
    validation_rows = max(1, int(round(len(X) * cfg.validation_fraction))) if cfg.validation_fraction else 0
    validation_rows = min(validation_rows, max(0, len(X) - 2))
    val_idx = indices[:validation_rows]
    train_idx = indices[validation_rows:]

    feature_mean = X[train_idx].mean(axis=0)
    feature_std = X[train_idx].std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    X_train = (X[train_idx] - feature_mean) / feature_std
    X_val = (X[val_idx] - feature_mean) / feature_std if len(val_idx) else np.empty((0, X.shape[1]))

    torch_mod.manual_seed(cfg.seed)
    model = _controller_net(X.shape[1], cfg.hidden_sizes)
    optimizer = torch_mod.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    train_weights = np.where(glucose[train_idx] < 70.0, cfg.hypo_loss_weight, 1.0)
    dataset = torch_tensor_dataset_cls(
        torch_mod.tensor(X_train, dtype=torch_mod.float32),
        torch_mod.tensor(y[train_idx], dtype=torch_mod.float32),
        torch_mod.tensor(train_weights, dtype=torch_mod.float32),
    )
    loader = torch_data_loader_cls(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        generator=torch_mod.Generator().manual_seed(cfg.seed),
    )

    loss_curve: list[float] = []
    for _ in range(cfg.epochs):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y, batch_weight in loader:
            optimizer.zero_grad()
            predictions = model(batch_x).reshape(-1)
            loss = torch_mod.mean(batch_weight * (predictions - batch_y) ** 2)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        loss_curve.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

    payload: Dict[str, Any] = {
        "model_type": "neural_imitation_controller",
        "feature_columns": list(CONTROL_FEATURE_COLUMNS),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "hidden_sizes": list(cfg.hidden_sizes),
        "max_output_units": cfg.max_output_units,
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "loss_curve": loss_curve,
    }
    train_predictions = _predict_from_matrix(payload, model, X[train_idx])
    payload["train_metrics"] = evaluate_controller_predictions(df.iloc[train_idx].reset_index(drop=True), train_predictions)
    if len(val_idx):
        val_predictions = _predict_from_matrix(payload, model, X[val_idx])
        payload["validation_metrics"] = evaluate_controller_predictions(
            df.iloc[val_idx].reset_index(drop=True),
            val_predictions,
        )
    else:
        payload["validation_metrics"] = None
    return payload


def predict_neural_controller(payload: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    model = instantiate_neural_controller_model(payload)
    X, _, _ = _prepare_arrays(df)
    return _predict_from_matrix(payload, model, X)


def save_neural_controller(payload: Dict[str, Any], path: Path) -> None:
    torch_mod = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_mod.save(payload, path)


def load_neural_controller(path: Path) -> Dict[str, Any]:
    torch_mod = _require_torch()
    payload = torch_mod.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("model_type") != "neural_imitation_controller":
        raise ValueError(f"{path} is not a neural imitation controller checkpoint.")
    return payload
