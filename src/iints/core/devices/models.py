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
        drift_std_per_hour: float = 0.0,
        drift_max_abs_mgdl: float = 0.0,
        dropout_duration_steps: int | Tuple[int, int] = 1,
        compression_low_prob: float = 0.0,
        compression_low_max_glucose: float = 140.0,
        compression_low_mgdl_range: Tuple[float, float] = (12.0, 28.0),
        compression_low_duration_steps: int | Tuple[int, int] = (3, 8),
    ) -> None:
        self.noise_std = noise_std
        self.bias = bias
        self.lag_minutes = lag_minutes
        self.dropout_prob = dropout_prob
        self.drift_std_per_hour = drift_std_per_hour
        self.drift_max_abs_mgdl = drift_max_abs_mgdl
        self.dropout_duration_steps = _normalize_step_window(dropout_duration_steps)
        self.compression_low_prob = compression_low_prob
        self.compression_low_max_glucose = compression_low_max_glucose
        self.compression_low_mgdl_range = compression_low_mgdl_range
        self.compression_low_duration_steps = _normalize_step_window(compression_low_duration_steps)
        self._rng = np.random.default_rng(seed)
        self._history: list[tuple[float, float]] = []
        self._last_reading: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._drift_offset = 0.0
        self._dropout_remaining_steps = 0
        self._compression_remaining_steps = 0
        self._compression_offset = 0.0

    def reset(self) -> None:
        self._history = []
        self._last_reading = None
        self._last_timestamp = None
        self._drift_offset = 0.0
        self._dropout_remaining_steps = 0
        self._compression_remaining_steps = 0
        self._compression_offset = 0.0

    def _sample_duration_steps(self, window: Tuple[int, int]) -> int:
        start, end = window
        if start == end:
            return start
        return int(self._rng.integers(start, end + 1))

    def _update_drift(self, current_time: float) -> None:
        if self.drift_std_per_hour <= 0:
            self._last_timestamp = current_time
            return
        if self._last_timestamp is None or current_time <= self._last_timestamp:
            self._last_timestamp = current_time
            return
        dt_hours = max((current_time - self._last_timestamp) / 60.0, 0.0)
        drift_step = float(self._rng.normal(0.0, self.drift_std_per_hour * max(dt_hours, 1e-6) ** 0.5))
        self._drift_offset += drift_step
        if self.drift_max_abs_mgdl > 0:
            self._drift_offset = float(
                np.clip(self._drift_offset, -self.drift_max_abs_mgdl, self.drift_max_abs_mgdl)
            )
        self._last_timestamp = current_time

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

        self._update_drift(current_time)
        reading = base + self.bias + self._drift_offset
        if self.noise_std > 0:
            reading += float(self._rng.normal(0, self.noise_std))

        status_parts = ["ok"]
        if (
            self._compression_remaining_steps <= 0
            and self.compression_low_prob > 0
            and base <= self.compression_low_max_glucose
            and float(self._rng.random()) < self.compression_low_prob
        ):
            self._compression_remaining_steps = self._sample_duration_steps(self.compression_low_duration_steps)
            self._compression_offset = float(
                self._rng.uniform(self.compression_low_mgdl_range[0], self.compression_low_mgdl_range[1])
            )
        if self._compression_remaining_steps > 0:
            reading -= self._compression_offset
            status_parts = ["compression_low"]
            self._compression_remaining_steps -= 1

        if self._dropout_remaining_steps <= 0 and self.dropout_prob > 0 and float(self._rng.random()) < self.dropout_prob:
            self._dropout_remaining_steps = self._sample_duration_steps(self.dropout_duration_steps)
        if self._dropout_remaining_steps > 0:
            if self._last_reading is not None:
                reading = self._last_reading
            status_parts = ["dropout_hold"]
            self._dropout_remaining_steps -= 1

        self._last_reading = reading
        return SensorReading(value=reading, status="+".join(status_parts))

    def get_state(self) -> Dict[str, Any]:
        return {
            "noise_std": self.noise_std,
            "bias": self.bias,
            "lag_minutes": self.lag_minutes,
            "dropout_prob": self.dropout_prob,
            "drift_std_per_hour": self.drift_std_per_hour,
            "drift_max_abs_mgdl": self.drift_max_abs_mgdl,
            "dropout_duration_steps": self.dropout_duration_steps,
            "compression_low_prob": self.compression_low_prob,
            "compression_low_max_glucose": self.compression_low_max_glucose,
            "compression_low_mgdl_range": self.compression_low_mgdl_range,
            "compression_low_duration_steps": self.compression_low_duration_steps,
            "last_reading": self._last_reading,
            "last_timestamp": self._last_timestamp,
            "drift_offset": self._drift_offset,
            "dropout_remaining_steps": self._dropout_remaining_steps,
            "compression_remaining_steps": self._compression_remaining_steps,
            "compression_offset": self._compression_offset,
            "history": self._history,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.noise_std = state.get("noise_std", self.noise_std)
        self.bias = state.get("bias", self.bias)
        self.lag_minutes = state.get("lag_minutes", self.lag_minutes)
        self.dropout_prob = state.get("dropout_prob", self.dropout_prob)
        self.drift_std_per_hour = state.get("drift_std_per_hour", self.drift_std_per_hour)
        self.drift_max_abs_mgdl = state.get("drift_max_abs_mgdl", self.drift_max_abs_mgdl)
        self.dropout_duration_steps = _normalize_step_window(state.get("dropout_duration_steps", self.dropout_duration_steps))
        self.compression_low_prob = state.get("compression_low_prob", self.compression_low_prob)
        self.compression_low_max_glucose = state.get("compression_low_max_glucose", self.compression_low_max_glucose)
        self.compression_low_mgdl_range = tuple(state.get("compression_low_mgdl_range", self.compression_low_mgdl_range))
        self.compression_low_duration_steps = _normalize_step_window(
            state.get("compression_low_duration_steps", self.compression_low_duration_steps)
        )
        self._last_reading = state.get("last_reading")
        self._last_timestamp = state.get("last_timestamp")
        self._drift_offset = state.get("drift_offset", self._drift_offset)
        self._dropout_remaining_steps = int(state.get("dropout_remaining_steps", self._dropout_remaining_steps))
        self._compression_remaining_steps = int(state.get("compression_remaining_steps", self._compression_remaining_steps))
        self._compression_offset = state.get("compression_offset", self._compression_offset)
        self._history = state.get("history", [])


def _normalize_step_window(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        start, end = int(value[0]), int(value[1])
    else:
        start = end = int(value)
    start = max(1, start)
    end = max(start, end)
    return (start, end)


SENSOR_PROFILES: Dict[str, Dict[str, Any]] = {
    "ideal": {
        "noise_std": 0.0,
        "bias": 0.0,
        "lag_minutes": 0,
        "dropout_prob": 0.0,
        "drift_std_per_hour": 0.0,
        "drift_max_abs_mgdl": 0.0,
        "dropout_duration_steps": (1, 1),
        "compression_low_prob": 0.0,
        "compression_low_max_glucose": 140.0,
        "compression_low_mgdl_range": (12.0, 28.0),
        "compression_low_duration_steps": (3, 8),
    },
    "clinical_cgm": {
        "noise_std": 7.0,
        "bias": 0.0,
        "lag_minutes": 10,
        "dropout_prob": 0.0,
        "drift_std_per_hour": 0.0,
        "drift_max_abs_mgdl": 0.0,
        "dropout_duration_steps": (1, 1),
        "compression_low_prob": 0.0,
        "compression_low_max_glucose": 140.0,
        "compression_low_mgdl_range": (12.0, 28.0),
        "compression_low_duration_steps": (3, 8),
    },
    "free_living_cgm": {
        "noise_std": 8.0,
        "bias": 0.0,
        "lag_minutes": 10,
        "dropout_prob": 0.004,
        "drift_std_per_hour": 0.8,
        "drift_max_abs_mgdl": 18.0,
        "dropout_duration_steps": (2, 6),
        "compression_low_prob": 0.006,
        "compression_low_max_glucose": 145.0,
        "compression_low_mgdl_range": (10.0, 26.0),
        "compression_low_duration_steps": (3, 10),
    },
    "compression_prone": {
        "noise_std": 8.5,
        "bias": 0.0,
        "lag_minutes": 12,
        "dropout_prob": 0.003,
        "drift_std_per_hour": 0.9,
        "drift_max_abs_mgdl": 20.0,
        "dropout_duration_steps": (2, 6),
        "compression_low_prob": 0.015,
        "compression_low_max_glucose": 155.0,
        "compression_low_mgdl_range": (18.0, 42.0),
        "compression_low_duration_steps": (4, 12),
    },
}


def create_sensor_model(
    *,
    profile: str = "clinical_cgm",
    seed: Optional[int] = None,
    **overrides: Any,
) -> SensorModel:
    if profile not in SENSOR_PROFILES:
        available = ", ".join(sorted(SENSOR_PROFILES))
        raise ValueError(f"Unknown sensor profile '{profile}'. Available profiles: {available}")
    config = dict(SENSOR_PROFILES[profile])
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return SensorModel(seed=seed, **config)


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
