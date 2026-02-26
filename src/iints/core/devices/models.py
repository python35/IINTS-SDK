from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np


@dataclass
class SensorReading:
    value: float
    status: str


class SensorModel:
    """
    Sensor error model for CGM readings.

    Supports noise, bias, lag (minutes), and dropout (hold last value).
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        bias: float = 0.0,
        lag_minutes: int = 0,
        dropout_prob: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.noise_std = noise_std
        self.bias = bias
        self.lag_minutes = lag_minutes
        self.dropout_prob = dropout_prob
        self._rng = np.random.default_rng(seed)
        self._history: list[tuple[float, float]] = []
        self._last_reading: Optional[float] = None

    def reset(self) -> None:
        self._history = []
        self._last_reading = None

    def read(self, true_glucose: float, current_time: float) -> SensorReading:
        self._history.append((current_time, true_glucose))
        # Keep history window bounded
        if self.lag_minutes > 0:
            cutoff = current_time - (self.lag_minutes * 2)
            self._history = [(t, v) for (t, v) in self._history if t >= cutoff]

        if self.lag_minutes > 0:
            target_time = current_time - self.lag_minutes
            candidates = [v for (t, v) in self._history if t <= target_time]
            base = candidates[-1] if candidates else true_glucose
        else:
            base = true_glucose

        reading = base + self.bias
        if self.noise_std > 0:
            reading += float(self._rng.normal(0, self.noise_std))

        status = "ok"
        if self.dropout_prob > 0 and float(self._rng.random()) < self.dropout_prob:
            status = "dropout_hold"
            if self._last_reading is not None:
                reading = self._last_reading

        self._last_reading = reading
        return SensorReading(value=reading, status=status)

    def get_state(self) -> Dict[str, Any]:
        return {
            "noise_std": self.noise_std,
            "bias": self.bias,
            "lag_minutes": self.lag_minutes,
            "dropout_prob": self.dropout_prob,
            "last_reading": self._last_reading,
            "history": self._history,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.noise_std = state.get("noise_std", self.noise_std)
        self.bias = state.get("bias", self.bias)
        self.lag_minutes = state.get("lag_minutes", self.lag_minutes)
        self.dropout_prob = state.get("dropout_prob", self.dropout_prob)
        self._last_reading = state.get("last_reading")
        self._history = state.get("history", [])


@dataclass
class PumpDelivery:
    delivered_units: float
    status: str
    reason: str


class PumpModel:
    """
    Pump error model for insulin delivery.

    Supports max delivery per step, quantization, and occlusion/dropout.
    """

    def __init__(
        self,
        max_units_per_step: Optional[float] = None,
        quantization_units: Optional[float] = None,
        dropout_prob: float = 0.0,
        delivery_noise_std: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.max_units_per_step = max_units_per_step
        self.quantization_units = quantization_units
        self.dropout_prob = dropout_prob
        self.delivery_noise_std = delivery_noise_std
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def deliver(self, requested_units: float, time_step_minutes: float) -> PumpDelivery:
        delivered = requested_units
        status = "ok"
        reason = "approved"

        if delivered < 0.0:
            delivered = 0.0
            status = "clamped"
            reason = "negative_request"

        if self.max_units_per_step is not None and delivered > self.max_units_per_step:
            delivered = self.max_units_per_step
            status = "capped"
            reason = f"max_units_per_step {self.max_units_per_step:.2f}"

        if self.quantization_units:
            delivered = round(delivered / self.quantization_units) * self.quantization_units

        if self.delivery_noise_std > 0:
            delivered += float(self._rng.normal(0, self.delivery_noise_std))
            delivered = max(0.0, delivered)

        if self.dropout_prob > 0 and float(self._rng.random()) < self.dropout_prob:
            delivered = 0.0
            status = "occlusion"
            reason = "pump_dropout"

        return PumpDelivery(delivered_units=delivered, status=status, reason=reason)

    def get_state(self) -> Dict[str, Any]:
        return {
            "max_units_per_step": self.max_units_per_step,
            "quantization_units": self.quantization_units,
            "dropout_prob": self.dropout_prob,
            "delivery_noise_std": self.delivery_noise_std,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.max_units_per_step = state.get("max_units_per_step", self.max_units_per_step)
        self.quantization_units = state.get("quantization_units", self.quantization_units)
        self.dropout_prob = state.get("dropout_prob", self.dropout_prob)
        self.delivery_noise_std = state.get("delivery_noise_std", self.delivery_noise_std)
