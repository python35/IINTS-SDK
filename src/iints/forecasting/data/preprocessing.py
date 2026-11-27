"""Step 2a — Leakage-Proof Train-Only Preprocessing (`TrainOnlyStandardScaler`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = ["TrainOnlyStandardScaler"]


@dataclass
class TrainOnlyStandardScaler:
    """Standardizes CGM windows using mean & SD estimated strictly on `train_fit`."""

    mean_mgdl: float = 0.0
    std_mgdl: float = 1.0
    is_fitted: bool = False

    def fit(self, x_train_fit: np.ndarray) -> "TrainOnlyStandardScaler":
        arr = np.asarray(x_train_fit, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("TrainOnlyStandardScaler.fit requires non-empty finite training windows.")
        self.mean_mgdl = float(np.mean(arr))
        std = float(np.std(arr))
        self.std_mgdl = std if std > 1e-8 else 1.0
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TrainOnlyStandardScaler must be fitted on train_fit before transform().")
        return (np.asarray(x, dtype=float) - self.mean_mgdl) / self.std_mgdl

    def to_dict(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted.")
        return {"mean_mgdl": self.mean_mgdl, "std_mgdl": self.std_mgdl}

    @classmethod
    def from_dict(cls, payload: Dict[str, float]) -> "TrainOnlyStandardScaler":
        return cls(
            mean_mgdl=float(payload["mean_mgdl"]),
            std_mgdl=float(payload["std_mgdl"]),
            is_fitted=True,
        )
