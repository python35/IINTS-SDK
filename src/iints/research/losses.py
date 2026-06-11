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

    class PhysiologicalPINNLoss:  # type: ignore[no-redef]
        """PINN loss for physiological boundaries."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Torch is required for PhysiologicalPINNLoss. Install with `pip install iints-sdk-python35[research]`."
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
        Physics-Informed Neural Network (PINN) loss for physiological glucose forecasting.
        Combines standard MSE with heavy penalties for biologically impossible states.
        """

        def __init__(
            self,
            feature_columns: list[str],
            pinn_lambda: float = 0.5,
            pinn_max_roc: float = 10.0,
            time_step_minutes: int = 5,
        ) -> None:
            super().__init__()
            self.pinn_lambda = pinn_lambda
            self.pinn_max_roc = pinn_max_roc
            self.time_step_minutes = time_step_minutes

            self.idx_iob = feature_columns.index("patient_iob_units") if "patient_iob_units" in feature_columns else -1
            self.idx_cob = feature_columns.index("patient_cob_grams") if "patient_cob_grams" in feature_columns else -1
            self.idx_glucose = feature_columns.index("glucose_actual_mgdl") if "glucose_actual_mgdl" in feature_columns else -1

            self.mse = nn.MSELoss()

        def forward(self, preds: "torch.Tensor", targets: "torch.Tensor", inputs: "torch.Tensor") -> "torch.Tensor":
            """
            inputs shape: (batch, history_steps, features)
            preds shape: (batch, horizon_steps) OR (batch, 1)
            targets shape: (batch, horizon_steps) OR (batch, 1)
            """
            mse_loss = self.mse(preds, targets)

            if self.idx_glucose == -1:
                return mse_loss  # Cannot apply PINN without glucose history

            # Extract the last known state from inputs (time step = -1)
            # inputs is (batch, history, features)
            last_glucose = inputs[:, -1, self.idx_glucose]  # shape: (batch,)

            iob = inputs[:, -1, self.idx_iob] if self.idx_iob != -1 else torch.zeros_like(last_glucose)
            cob = inputs[:, -1, self.idx_cob] if self.idx_cob != -1 else torch.zeros_like(last_glucose)

            pinn_penalty = torch.tensor(0.0, device=preds.device)

            # 1. Absolute Bounds Penalty
            # Glucose physiologically cannot be < 20 or > 600
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

            # 2. Maximum Rate of Change Penalty (e.g. > 10 mg/dL/min is biologically unrealistic)
            p_roc_up = torch.relu(roc - self.pinn_max_roc)
            p_roc_down = torch.relu((-roc) - self.pinn_max_roc)
            pinn_penalty += (p_roc_up ** 2).mean() * 5.0
            pinn_penalty += (p_roc_down ** 2).mean() * 5.0

            # 3. IOB without COB should not result in a sharp rise
            # If IOB > 1.0 U and COB < 5.0 g, a sharp rise (roc > 2.0) is very suspicious
            iob_mask = (iob > 1.0).float()
            no_cob_mask = (cob < 5.0).float()
            suspicious_rise = torch.relu(roc - 2.0)
            pinn_penalty += (suspicious_rise * iob_mask * no_cob_mask).mean() * 2.0

            # 4. COB without IOB should not result in a sharp drop
            # If COB > 10.0 g and IOB < 0.5 U, a sharp drop (roc < -2.0) is very suspicious
            cob_mask = (cob > 10.0).float()
            no_iob_mask = (iob < 0.5).float()
            suspicious_drop = torch.relu((-roc) - 2.0)
            pinn_penalty += (suspicious_drop * cob_mask * no_iob_mask).mean() * 2.0

            total_loss = mse_loss + self.pinn_lambda * pinn_penalty
            return total_loss
