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


class QuantileLoss(nn.Module):
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


class SafetyWeightedMSE(nn.Module):
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
