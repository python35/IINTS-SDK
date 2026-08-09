from __future__ import annotations

from typing import Any, Optional

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

        def __call__(self, *args: object, **kwargs: object) -> Any:
            raise ImportError(
                "Torch is required for QuantileLoss. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

    class SafetyWeightedMSE:  # type: ignore[no-redef]
        """MSE with extra weight on low-glucose targets (safety-critical)."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Torch is required for SafetyWeightedMSE. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

        def __call__(self, *args: object, **kwargs: object) -> Any:
            raise ImportError(
                "Torch is required for SafetyWeightedMSE. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

    class BandWeightedMSE:  # type: ignore[no-redef]
        """MSE with extra weight on low/high glucose bands."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Torch is required for BandWeightedMSE. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

        def __call__(self, *args: object, **kwargs: object) -> Any:
            raise ImportError(
                "Torch is required for BandWeightedMSE. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

    class PhysiologicalPINNLoss:  # type: ignore[no-redef]
        """Physiology-informed constraint loss (legacy PINN class name)."""

        pinn_lambda: float

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Torch is required for PhysiologicalPINNLoss. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

        def __call__(self, *args: object, **kwargs: object) -> Any:
            raise ImportError(
                "Torch is required for PhysiologicalPINNLoss. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

        def physiology_penalty(self, *args: object, **kwargs: object) -> Any:
            raise ImportError(
                "Torch is required for PhysiologicalPINNLoss. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

    class BandWeightedPINNLoss:  # type: ignore[no-redef]
        """Band-weighted MSE plus physiology-informed penalties."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Torch is required for BandWeightedPINNLoss. Install with `pip install iints-sdk-python35[research]`."
            ) from _IMPORT_ERROR

        def __call__(self, *args: object, **kwargs: object) -> Any:
            raise ImportError(
                "Torch is required for BandWeightedPINNLoss. Install with `pip install iints-sdk-python35[research]`."
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

    class PhysiologicalPINNLoss(nn.Module):  # type: ignore[misc,no-redef]
        """
        Physiology-informed constraint loss for glucose forecasting.

        The historical class name is retained for checkpoint/configuration
        compatibility. Inputs supplied to this loss must remain in physical
        units; model-normalized inputs are not scientifically interpretable.
        """

        def __init__(
            self,
            feature_columns: list[str],
            pinn_lambda: float = 0.5,
            pinn_max_roc: float = 3.0,
            time_step_minutes: int = 5,
        ) -> None:
            super().__init__()
            self.pinn_lambda = pinn_lambda
            self.pinn_max_roc = pinn_max_roc
            self.time_step_minutes = time_step_minutes

            self.idx_glucose = feature_columns.index("glucose_actual_mgdl") if "glucose_actual_mgdl" in feature_columns else -1

            self.mse = nn.MSELoss()

        def physiology_penalty(self, preds: "torch.Tensor", inputs: "torch.Tensor") -> "torch.Tensor":
            """
            inputs shape: (batch, history_steps, features)
            preds shape: (batch, horizon_steps) OR (batch, 1)
            """
            if self.idx_glucose == -1:
                return torch.tensor(0.0, device=preds.device)

            # Extract the last known state from inputs (time step = -1)
            # inputs is (batch, history, features)
            last_glucose = inputs[:, -1, self.idx_glucose]  # shape: (batch,)

            pinn_penalty = torch.tensor(0.0, device=preds.device)

            # 1. Absolute Bounds Penalty
            # Explicit simulator support envelope, not a diagnostic boundary.
            p_low = torch.relu(20.0 - preds)  # >0 if pred < 20
            p_high = torch.relu(preds - 600.0) # >0 if pred > 600
            pinn_penalty += (p_low ** 2).mean() * 10.0
            pinn_penalty += (p_high ** 2).mean() * 10.0

            # For derivatives, compare first prediction step with last_glucose
            # preds[:, 0] is the prediction t+1
            if len(preds.shape) > 1 and preds.shape[1] > 0:
                first_pred = preds[:, 0]
            else:
                first_pred = preds.view(-1)

            # Rate of change (mg/dL per minute)
            roc = (first_pred - last_glucose) / max(1.0, float(self.time_step_minutes))

            # 2. Configured rate-of-change support-envelope penalty.
            p_roc_up = torch.relu(roc - self.pinn_max_roc)
            p_roc_down = torch.relu((-roc) - self.pinn_max_roc)
            pinn_penalty += (p_roc_up ** 2).mean() * 5.0
            pinn_penalty += (p_roc_down ** 2).mean() * 5.0

            # Apply the same rate constraint throughout the forecast horizon.
            # IOB/COB direction rules are intentionally excluded: stress,
            # exercise, illness and unannounced meals make them non-universal.
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                future_roc = torch.diff(preds, dim=1) / max(
                    1.0, float(self.time_step_minutes)
                )
                future_excess = torch.relu(torch.abs(future_roc) - self.pinn_max_roc)
                pinn_penalty += (future_excess ** 2).mean() * 5.0
            return pinn_penalty

        def forward(self, preds: "torch.Tensor", targets: "torch.Tensor", inputs: "torch.Tensor") -> "torch.Tensor":
            """
            inputs shape: (batch, history_steps, features)
            preds shape: (batch, horizon_steps) OR (batch, 1)
            targets shape: (batch, horizon_steps) OR (batch, 1)
            """
            mse_loss = self.mse(preds, targets)
            pinn_penalty = self.physiology_penalty(preds, inputs)

            total_loss = mse_loss + self.pinn_lambda * pinn_penalty
            return total_loss

    class BandWeightedPINNLoss(PhysiologicalPINNLoss):  # type: ignore[misc,no-redef]
        """
        Band-weighted loss with physiology-informed trajectory constraints.

        This is the recommended long-run training objective for the public
        IINTS glucose forecaster: band weighting focuses the optimizer on
        hypo/hyper ranges, while the constraint term discourages trajectories
        outside the configured research support envelope.
        """

        def __init__(
            self,
            feature_columns: list[str],
            pinn_lambda: float = 0.5,
            pinn_max_roc: float = 3.0,
            time_step_minutes: int = 5,
            low_threshold: float = 70.0,
            high_threshold: float = 180.0,
            low_weight: float = 2.0,
            high_weight: float = 1.5,
            max_weight: float = 5.0,
        ) -> None:
            super().__init__(
                feature_columns=feature_columns,
                pinn_lambda=pinn_lambda,
                pinn_max_roc=pinn_max_roc,
                time_step_minutes=time_step_minutes,
            )
            self.band_loss = BandWeightedMSE(
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                low_weight=low_weight,
                high_weight=high_weight,
                max_weight=max_weight,
            )

        def forward(self, preds: "torch.Tensor", targets: "torch.Tensor", inputs: "torch.Tensor") -> "torch.Tensor":
            band_loss = self.band_loss(preds, targets)
            pinn_penalty = self.physiology_penalty(preds, inputs)
            return band_loss + self.pinn_lambda * pinn_penalty
