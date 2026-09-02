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

    def _weighting_levels(
        targets: "torch.Tensor",
        inputs: "Optional[torch.Tensor]",
        anchor_index: Optional[int],
        loss_name: str,
    ) -> "torch.Tensor":
        """Return the glucose levels, in mg/dL, that a level threshold applies to.

        With absolute targets the target *is* the level, so this is the identity.

        With delta targets (``predictor.predict_delta``) the target is a change,
        and a level threshold applied to a change is meaningless: a drop of
        5 mg/dL is not hypoglycemia, and a change is almost never above 180 or
        below 70. Weighting on the raw delta therefore makes every sample fall in
        the "low" band and none in the "high" band -- a constant weight, which is
        plain MSE with a rescaled learning rate. Nothing raises; the run simply
        stops doing what its config says.

        The level is recovered from the anchor: the last observed glucose in the
        input window, which is exactly what ``build_sequences`` subtracted to
        form the delta. ``inputs`` must be the *unscaled* window, since the
        thresholds are in mg/dL.
        """
        if anchor_index is None:
            return targets
        if inputs is None:
            raise ValueError(
                f"{loss_name} is configured for delta targets but received no input "
                "window, so it cannot recover the glucose level its thresholds refer "
                "to. Pass the unscaled input batch as the third argument."
            )
        anchor = inputs[:, -1, anchor_index].unsqueeze(-1).to(targets.dtype)
        return anchor + targets

    class SafetyWeightedMSE(nn.Module):  # type: ignore[misc,no-redef]
        """MSE with extra weight on low-glucose targets (safety-critical).

        ``anchor_index`` is the position of the glucose column in the feature
        window. Set it when the targets are deltas; leave it ``None`` for
        absolute targets. See :func:`_weighting_levels`.
        """

        def __init__(
            self,
            low_threshold: float = 80.0,
            alpha: float = 2.0,
            max_weight: float = 4.0,
            anchor_index: Optional[int] = None,
        ) -> None:
            super().__init__()
            self.low_threshold = float(low_threshold)
            self.alpha = float(alpha)
            self.max_weight = float(max_weight)
            self.anchor_index = anchor_index

        def forward(
            self,
            preds: "torch.Tensor",
            targets: "torch.Tensor",
            inputs: "Optional[torch.Tensor]" = None,
        ) -> "torch.Tensor":
            levels = _weighting_levels(targets, inputs, self.anchor_index, "safety_weighted")
            # Emphasize errors below the low threshold
            delta = torch.clamp(self.low_threshold - levels, min=0.0)
            weights = 1.0 + self.alpha * (delta / max(self.low_threshold, 1.0))
            weights = torch.clamp(weights, max=self.max_weight)
            return ((preds - targets) ** 2 * weights).mean()

    class BandWeightedMSE(nn.Module):  # type: ignore[misc,no-redef]
        """MSE with extra weight on low/high glucose targets (band-weighted).

        ``anchor_index`` is the position of the glucose column in the feature
        window. Set it when the targets are deltas; leave it ``None`` for
        absolute targets. See :func:`_weighting_levels`.
        """

        def __init__(
            self,
            low_threshold: float = 70.0,
            high_threshold: float = 180.0,
            low_weight: float = 2.0,
            high_weight: float = 1.5,
            max_weight: float = 5.0,
            anchor_index: Optional[int] = None,
        ) -> None:
            super().__init__()
            self.low_threshold = float(low_threshold)
            self.high_threshold = float(high_threshold)
            self.low_weight = float(low_weight)
            self.high_weight = float(high_weight)
            self.max_weight = float(max_weight)
            self.anchor_index = anchor_index

        def forward(
            self,
            preds: "torch.Tensor",
            targets: "torch.Tensor",
            inputs: "Optional[torch.Tensor]" = None,
        ) -> "torch.Tensor":
            levels = _weighting_levels(targets, inputs, self.anchor_index, "band_weighted")
            weights = torch.ones_like(levels)
            weights = weights + self.low_weight * (levels < self.low_threshold).float()
            weights = weights + self.high_weight * (levels > self.high_threshold).float()
            weights = torch.clamp(weights, max=self.max_weight)
            return ((preds - targets) ** 2 * weights).mean()

    class PhysiologicalPINNLoss(nn.Module):  # type: ignore[misc,no-redef]
        """
        Physiology-informed constraint loss for glucose forecasting.

        The historical class name is retained for checkpoint/configuration
        compatibility. Inputs supplied to this loss must remain in physical
        units; model-normalized inputs are not scientifically interpretable.

        Set ``predict_delta`` when the model emits changes rather than levels.
        Every constraint here is a statement about a glucose *level* or about a
        rate between levels, so the penalty is applied to the reconstructed
        trajectory. Applied to raw deltas the bounds penalty would fire on every
        sample (a delta near zero looks like 0 mg/dL of blood glucose) and the
        first-step rate would be the distance between a change and a level.
        """

        def __init__(
            self,
            feature_columns: list[str],
            pinn_lambda: float = 0.5,
            pinn_max_roc: float = 3.0,
            time_step_minutes: int = 5,
            predict_delta: bool = False,
        ) -> None:
            super().__init__()
            self.pinn_lambda = pinn_lambda
            self.pinn_max_roc = pinn_max_roc
            self.time_step_minutes = time_step_minutes
            self.predict_delta = bool(predict_delta)

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

            # Constraints are statements about levels; recover them if the model
            # emits changes. The anchor is the last observed glucose, which is
            # what build_sequences subtracted to form the delta target.
            if self.predict_delta:
                anchor = last_glucose.to(preds.dtype)
                preds = (anchor.unsqueeze(-1) if preds.dim() > 1 else anchor) + preds

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
            anchor_index: Optional[int] = None,
        ) -> None:
            super().__init__(
                feature_columns=feature_columns,
                pinn_lambda=pinn_lambda,
                pinn_max_roc=pinn_max_roc,
                time_step_minutes=time_step_minutes,
                predict_delta=anchor_index is not None,
            )
            self.band_loss = BandWeightedMSE(
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                low_weight=low_weight,
                high_weight=high_weight,
                max_weight=max_weight,
                anchor_index=anchor_index,
            )

        def forward(self, preds: "torch.Tensor", targets: "torch.Tensor", inputs: "torch.Tensor") -> "torch.Tensor":
            band_loss = self.band_loss(preds, targets, inputs)
            pinn_penalty = self.physiology_penalty(preds, inputs)
            return band_loss + self.pinn_lambda * pinn_penalty
