from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _IMPORT_ERROR: Optional[BaseException] = exc
else:
    _IMPORT_ERROR = None


if nn is None:  # pragma: no cover
    class QuantileLoss:  # type: ignore[no-redef]
        """Pinball / quantile loss for probabilistic forecasting."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Torch is required for QuantileLoss. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

    class SafetyWeightedMSE:  # type: ignore[no-redef]
        """MSE with extra weight on low-glucose targets (safety-critical)."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Torch is required for SafetyWeightedMSE. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

    class BandWeightedMSE:  # type: ignore[no-redef]
        """MSE with extra weight on low/high glucose bands."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Torch is required for BandWeightedMSE. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR
else:
    class QuantileLoss(nn.Module):  # type: ignore[misc,no-redef]
        """Pinball / quantile loss for probabilistic forecasting."""

        def __init__(self, quantile: float = 0.9) -> None:
            super().__init__()
            if not 0.0 < quantile < 1.0:
                raise ValueError(f"quantile must be in (0, 1), got {quantile}")
            self.quantile = quantile

        def forward(self, preds: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
            errors = targets - preds
            loss = torch.where(
                errors >= 0,
                self.quantile * errors,
                (self.quantile - 1.0) * errors,
            )
            return loss.mean()

    class SafetyWeightedMSE(nn.Module):  # type: ignore[misc,no-redef]
        """MSE with extra weight on low-glucose targets (safety-critical)."""

        def __init__(self, low_threshold: float = 80.0, alpha: float = 2.0, max_weight: float = 4.0) -> None:
            super().__init__()
            self.low_threshold = float(low_threshold)
            self.alpha = float(alpha)
            self.max_weight = float(max_weight)

        def forward(self, preds: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
            # Emphasize errors below the low threshold
            delta = torch.clamp(self.low_threshold - targets, min=0.0)
            weights = 1.0 + self.alpha * (delta / max(self.low_threshold, 1.0))
            weights = torch.clamp(weights, max=self.max_weight)
            return ((preds - targets) ** 2 * weights).mean()

    class BandWeightedMSE(nn.Module):  # type: ignore[misc,no-redef]
        """MSE with extra weight on low/high glucose targets (band-weighted)."""

        def __init__(
            self,
            low_threshold: float = 70.0,
            high_threshold: float = 180.0,
            low_weight: float = 2.0,
            high_weight: float = 1.5,
            max_weight: float = 5.0,
        ) -> None:
            super().__init__()
            self.low_threshold = float(low_threshold)
            self.high_threshold = float(high_threshold)
            self.low_weight = float(low_weight)
            self.high_weight = float(high_weight)
            self.max_weight = float(max_weight)

        def forward(self, preds: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
            weights = torch.ones_like(targets)
            weights = weights + self.low_weight * (targets < self.low_threshold).float()
            weights = weights + self.high_weight * (targets > self.high_threshold).float()
            weights = torch.clamp(weights, max=self.max_weight)
            return ((preds - targets) ** 2 * weights).mean()
