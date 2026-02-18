from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class LSTMPredictor(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon_steps: int = 12,
    ) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError(
                "Torch is required for LSTMPredictor. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR
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
            nn.Linear(hidden_size, horizon_steps),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden)


def load_predictor(model_path: Path) -> Tuple["LSTMPredictor", dict]:
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError(
            "Torch is required for predictor loading. Install with `pip install iints-sdk-python35[research]`."
        ) from _IMPORT_ERROR
    payload = torch.load(model_path, map_location="cpu")
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


def predict_batch(model: "LSTMPredictor", x: np.ndarray) -> np.ndarray:
    if torch is None:  # pragma: no cover
        raise ImportError(
            "Torch is required for predictor inference. Install with `pip install iints-sdk-python35[research]`."
        ) from _IMPORT_ERROR
    with torch.no_grad():
        tensor = torch.from_numpy(x.astype(np.float32))
        outputs = model(tensor).cpu().numpy()
    return outputs
