from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Protocol, Sequence, TYPE_CHECKING

import numpy as np

_IMPORT_ERROR: Optional[BaseException]
try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# LSTM predictor
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    import torch  # pragma: no cover
    from torch import nn  # pragma: no cover

    class LSTMPredictor(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
            horizon_steps: int = 12,
        ) -> None: ...

        def forward(self, x: "torch.Tensor") -> "torch.Tensor": ...

        def predict_with_uncertainty(
            self,
            x: "torch.Tensor",
            n_samples: int = 50,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]: ...
else:
    if nn is None:  # pragma: no cover
        class LSTMPredictor:  # type: ignore[no-redef]
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise ImportError(
                    "Torch is required for LSTMPredictor. Install with `pip install iints-sdk-python35[research]`."
                ) from _IMPORT_ERROR
    else:
        class LSTMPredictor(nn.Module):  # type: ignore[misc,no-redef]
            def __init__(
                self,
                input_size: int,
                hidden_size: int = 64,
                num_layers: int = 2,
                dropout: float = 0.1,
                horizon_steps: int = 12,
            ) -> None:
                super().__init__()
                self.horizon_steps = horizon_steps
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),   # P3-12: dropout in head for MC Dropout inference
                    nn.Linear(hidden_size, horizon_steps),
                )

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                _, (hidden, _) = self.lstm(x)
                last_hidden = hidden[-1]
                return self.head(last_hidden)

            # P3-12: Monte Carlo Dropout inference
            def predict_with_uncertainty(
                self,
                x: "torch.Tensor",
                n_samples: int = 50,
            ) -> Tuple["torch.Tensor", "torch.Tensor"]:
                """
                Run MC Dropout inference to estimate predictive uncertainty.

                Activates dropout at inference time and runs ``n_samples`` forward
                passes.  Returns the mean prediction and standard deviation across
                samples as a proxy for aleatoric + epistemic uncertainty.

                Parameters
                ----------
                x : torch.Tensor of shape [B, T, F]
                    Input batch.
                n_samples : int
                    Number of stochastic forward passes.

                Returns
                -------
                mean : torch.Tensor of shape [B, horizon_steps]
                std  : torch.Tensor of shape [B, horizon_steps]
                """
                # Keep dropout active during inference
                self.train()
                with torch.no_grad():
                    preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
                self.eval()
                return preds.mean(dim=0), preds.std(dim=0)


# ---------------------------------------------------------------------------
# P3-11: Baseline predictors
# ---------------------------------------------------------------------------

class BaselinePredictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def name(self) -> str:
        ...


class LastValueBaseline:
    """
    Naïve last-value (persistence) baseline for glucose forecasting.

    Predicts the same glucose value for all future time steps
    (i.e. ``y_hat[t+k] = y[t]`` for k = 1..horizon).

    This is the minimum bar any LSTM model must beat.
    """

    def __init__(self, horizon_steps: int) -> None:
        self.horizon_steps = horizon_steps

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray of shape [N, T, F]
            Feature sequences.  Assumes the first feature column is glucose
            (index 0 along the last axis), which is standard for the AZT1D pipeline.

        Returns
        -------
        np.ndarray of shape [N, horizon_steps]
        """
        # Last glucose reading in each sequence
        last_glucose = X[:, -1, 0]  # shape [N]
        return np.tile(last_glucose[:, None], (1, self.horizon_steps)).astype(np.float32)

    def name(self) -> str:
        return "LastValue"


class LinearTrendBaseline:
    """
    Linear-trend extrapolation baseline.

    Fits a least-squares line to the glucose values in the history window and
    extrapolates it ``horizon_steps`` steps into the future.

    Captures short-term trends (e.g. a rising glucose after a meal) without
    any knowledge of insulin or carbs, providing a stronger baseline than
    simple last-value persistence.
    """

    def __init__(self, horizon_steps: int, time_step_minutes: float = 5.0) -> None:
        self.horizon_steps = horizon_steps
        self.time_step_minutes = time_step_minutes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray of shape [N, T, F]
            Feature sequences.  First feature (index 0) must be glucose.

        Returns
        -------
        np.ndarray of shape [N, horizon_steps]
        """
        N, T, _ = X.shape
        glucose = X[:, :, 0]  # [N, T]
        t_hist = np.arange(T, dtype=np.float32)  # relative time indices
        t_future = np.arange(T, T + self.horizon_steps, dtype=np.float32)

        preds = np.empty((N, self.horizon_steps), dtype=np.float32)
        # Vectorised least-squares over the batch
        t_mean = t_hist.mean()
        t_var = ((t_hist - t_mean) ** 2).sum()

        if t_var < 1e-8:
            # Degenerate case: all time points identical → last-value fallback
            preds[:] = glucose[:, -1, None]
            return preds

        slopes = ((glucose * (t_hist - t_mean)).sum(axis=1)) / t_var  # [N]
        intercepts = glucose.mean(axis=1) - slopes * t_mean             # [N]
        preds = intercepts[:, None] + slopes[:, None] * t_future[None, :]
        return preds.astype(np.float32)

    def name(self) -> str:
        return "LinearTrend"


def evaluate_baselines(
    X: np.ndarray,
    y: np.ndarray,
    horizon_steps: int,
    time_step_minutes: float = 5.0,
) -> dict:
    """
    Compute MAE and RMSE for both baseline predictors.

    Parameters
    ----------
    X : np.ndarray [N, T, F]
    y : np.ndarray [N, horizon_steps]
    horizon_steps : int
    time_step_minutes : float

    Returns
    -------
    dict with keys "last_value" and "linear_trend", each containing
    {"mae": float, "rmse": float}.
    """
    results = {}
    baselines: Sequence[BaselinePredictor] = [
        LastValueBaseline(horizon_steps),
        LinearTrendBaseline(horizon_steps, time_step_minutes),
    ]
    for baseline in baselines:
        preds = baseline.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        results[baseline.name()] = {"mae": mae, "rmse": rmse}
    return results


# ---------------------------------------------------------------------------
# Service / loading helpers
# ---------------------------------------------------------------------------

class PredictorService:
    def __init__(self, model: "LSTMPredictor", config: dict) -> None:
        self.model = model
        self.config = config
        self.feature_columns = list(config.get("feature_columns", []))
        self.history_steps = int(config.get("history_steps", 1))
        self.horizon_steps = int(config.get("horizon_steps", 1))

        # Restore scaler if present in checkpoint
        from iints.research.dataset import FeatureScaler
        scaler_data = config.get("scaler")
        self.scaler: Optional[FeatureScaler] = (
            FeatureScaler.from_dict(scaler_data) if scaler_data else None
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        if torch is None:  # pragma: no cover
            raise ImportError(
                "Torch is required for predictor inference. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR
        if self.scaler is not None:
            x = self.scaler.transform(x)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x.astype(np.float32))
            outputs = self.model(tensor).cpu().numpy()
        return outputs

    def predict_with_uncertainty(
        self, x: np.ndarray, n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout inference — returns (mean, std) arrays."""
        if torch is None:  # pragma: no cover
            raise ImportError("Torch required.") from _IMPORT_ERROR
        if self.scaler is not None:
            x = self.scaler.transform(x)
        tensor = torch.from_numpy(x.astype(np.float32))
        mean_t, std_t = self.model.predict_with_uncertainty(tensor, n_samples=n_samples)
        return mean_t.detach().cpu().numpy(), std_t.detach().cpu().numpy()


def load_predictor(model_path: Path) -> Tuple["LSTMPredictor", dict]:
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError(
            "Torch is required for predictor loading. Install with `pip install iints-sdk-python35[research]`."
        ) from _IMPORT_ERROR
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = LSTMPredictor(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        horizon_steps=config["horizon_steps"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


def load_predictor_service(model_path: Path) -> PredictorService:
    model, config = load_predictor(model_path)
    return PredictorService(model, config)


def predict_batch(model: "LSTMPredictor", x: np.ndarray) -> np.ndarray:
    if torch is None:  # pragma: no cover
        raise ImportError(
            "Torch is required for predictor inference. Install with `pip install iints-sdk-python35[research]`."
        ) from _IMPORT_ERROR
    with torch.no_grad():
        tensor = torch.from_numpy(x.astype(np.float32))
        outputs = model(tensor).cpu().numpy()
    return outputs
