from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, cast

import numpy as np
from numpy.typing import NDArray


@dataclass
class SensorReading:
    value: float
    status: str


class SensorModel:
    """
    Sensor error model for CGM readings.

    Supports a first-order blood-to-interstitial approximation, optional
    transport dead time, measurement noise, bias, drift, and data gaps.

    ``lag_minutes`` is retained for API compatibility and represents explicit
    transport dead time. ``isf_tau_minutes`` is the time constant of the
    first-order interstitial compartment. They are separate assumptions and
    should not both be fitted to the same observed total CGM lag.
    """

    def __init__(
        self,
        noise_std: float = 8.5,
        bias: float = 0.0,
        lag_minutes: int = 5,
        isf_tau_minutes: float = 5.0,
        noise_ar1_phi: float = 0.85,
        noise_fbm_hurst: Optional[float] = 0.78,
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
        numeric = {
            "noise_std": float(noise_std),
            "bias": float(bias),
            "lag_minutes": float(lag_minutes),
            "isf_tau_minutes": float(isf_tau_minutes),
            "noise_ar1_phi": float(noise_ar1_phi),
            "dropout_prob": float(dropout_prob),
            "drift_std_per_hour": float(drift_std_per_hour),
            "drift_max_abs_mgdl": float(drift_max_abs_mgdl),
            "compression_low_prob": float(compression_low_prob),
            "compression_low_max_glucose": float(compression_low_max_glucose),
        }
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("sensor parameters must all be finite")
        if int(lag_minutes) != float(lag_minutes) or int(lag_minutes) < 0:
            raise ValueError("lag_minutes must be a non-negative whole number")
        if numeric["noise_std"] < 0.0:
            raise ValueError("noise_std must be non-negative")
        if numeric["isf_tau_minutes"] <= 0.0:
            raise ValueError("isf_tau_minutes must be positive")
        if not 0.0 <= numeric["noise_ar1_phi"] < 1.0:
            raise ValueError("noise_ar1_phi must satisfy 0 <= phi < 1")
        for name in ("dropout_prob", "compression_low_prob"):
            if not 0.0 <= numeric[name] <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "drift_std_per_hour",
            "drift_max_abs_mgdl",
            "compression_low_max_glucose",
        ):
            if numeric[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if noise_fbm_hurst is not None:
            hurst = float(noise_fbm_hurst)
            if not np.isfinite(hurst) or not 0.5 <= hurst <= 0.95:
                raise ValueError("noise_fbm_hurst must be between 0.5 and 0.95")
        compression_range = tuple(float(value) for value in compression_low_mgdl_range)
        if (
            len(compression_range) != 2
            or not all(np.isfinite(value) for value in compression_range)
            or compression_range[0] < 0.0
            or compression_range[1] < compression_range[0]
        ):
            raise ValueError(
                "compression_low_mgdl_range must be an ordered non-negative pair"
            )

        self.noise_std = numeric["noise_std"]
        self.bias = numeric["bias"]
        self.lag_minutes = int(lag_minutes)
        self.dropout_prob = numeric["dropout_prob"]
        self.drift_std_per_hour = numeric["drift_std_per_hour"]
        self.drift_max_abs_mgdl = numeric["drift_max_abs_mgdl"]
        self.isf_tau_minutes = numeric["isf_tau_minutes"]
        self.noise_ar1_phi = numeric["noise_ar1_phi"]
        self.noise_fbm_hurst = None if noise_fbm_hurst is None else float(noise_fbm_hurst)
        self.dropout_duration_steps = _normalize_step_window(dropout_duration_steps)
        self.compression_low_prob = numeric["compression_low_prob"]
        self.compression_low_max_glucose = numeric["compression_low_max_glucose"]
        self.compression_low_mgdl_range = compression_range
        self.compression_low_duration_steps = _normalize_step_window(compression_low_duration_steps)
        self._rng = np.random.default_rng(seed)
        self._initial_rng_state = copy.deepcopy(self._rng.bit_generator.state)
        self._history: list[tuple[float, float]] = []
        self._last_reading: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._current_isf: Optional[float] = None
        self._current_colored_noise: float = 0.0
        self._fbm_components: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._drift_offset = 0.0
        self._dropout_remaining_steps = 0
        self._compression_remaining_steps = 0
        self._compression_offset = 0.0

    def reset(self) -> None:
        self._rng.bit_generator.state = copy.deepcopy(self._initial_rng_state)
        self._history = []
        self._last_reading = None
        self._last_timestamp = None
        self._current_isf = None
        self._current_colored_noise = 0.0
        self._fbm_components = np.zeros(3, dtype=np.float64)
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

    def _lagged_glucose(self, current_time: float) -> float:
        """Return a delayed blood-glucose sample for the ISF compartment."""
        if self.lag_minutes <= 0 or not self._history:
            return self._history[-1][1]
        target_time = current_time - float(self.lag_minutes)
        if target_time <= self._history[0][0]:
            return self._history[0][1]
        previous_time, previous_value = self._history[0]
        for sample_time, sample_value in self._history[1:]:
            if sample_time >= target_time:
                span = max(sample_time - previous_time, 1e-9)
                fraction = (target_time - previous_time) / span
                return float(previous_value + fraction * (sample_value - previous_value))
            previous_time, previous_value = sample_time, sample_value
        return self._history[-1][1]

    def read(self, true_glucose: float, current_time: float) -> SensorReading:
        if not np.isfinite(true_glucose) or true_glucose < 0.0:
            raise ValueError("true_glucose must be a finite non-negative value")
        if not np.isfinite(current_time):
            raise ValueError("current_time must be finite")
        if self._last_timestamp is not None and current_time <= self._last_timestamp:
            raise ValueError("Sensor timestamps must be strictly increasing")
        self._history.append((current_time, true_glucose))

        # Calculate time step
        dt = 0.0
        if self._last_timestamp is not None and current_time > self._last_timestamp:
            dt = current_time - self._last_timestamp

        # First-order blood-to-interstitial approximation. This is a compact
        # sensor research model, not a device-specific CGM transfer function.
        lagged_glucose = self._lagged_glucose(current_time)
        if self._current_isf is None:
            self._current_isf = lagged_glucose
        elif dt > 0:
            # Exact update for constant input over the step:
            # ISF(t + dt) = BG + (ISF(t) - BG) * exp(-dt / tau).
            alpha = 1.0 - float(np.exp(-dt / self.isf_tau_minutes))
            self._current_isf = self._current_isf + alpha * (lagged_glucose - self._current_isf)

        base = self._current_isf

        self._update_drift(current_time)
        reading = base + self.bias + self._drift_offset

        # AR(1) colored noise or a long-memory, fractional-noise-style approximation.
        if self.noise_std > 0:
            if self.noise_fbm_hurst is not None:
                # Fractional-Brownian-style memory via multi-scale AR(1) superposition.
                # This is a compact CGM-noise approximation, not an exact fBM sampler.
                phis = np.array([0.5, 0.85, 0.98])
                scales = np.array([1.0, 6.0, 24.0])
                hurst = float(self.noise_fbm_hurst)
                weights = scales ** max(0.0, hurst - 0.5)
                weights = weights / np.linalg.norm(weights) * self.noise_std
                for i in range(3):
                    white_noise = float(self._rng.normal(0, weights[i] * np.sqrt(1 - phis[i]**2)))
                    self._fbm_components[i] = phis[i] * self._fbm_components[i] + white_noise
                reading += float(np.sum(self._fbm_components))
            else:
                white_noise_component = float(self._rng.normal(0, self.noise_std * np.sqrt(1 - self.noise_ar1_phi**2)))
                self._current_colored_noise = (self.noise_ar1_phi * self._current_colored_noise) + white_noise_component
                reading += self._current_colored_noise

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
            "isf_tau_minutes": self.isf_tau_minutes,
            "noise_ar1_phi": self.noise_ar1_phi,
            "noise_fbm_hurst": self.noise_fbm_hurst,
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
            "current_isf": self._current_isf,
            "current_colored_noise": self._current_colored_noise,
            "fbm_components": self._fbm_components.tolist(),
            "drift_offset": self._drift_offset,
            "dropout_remaining_steps": self._dropout_remaining_steps,
            "compression_remaining_steps": self._compression_remaining_steps,
            "compression_offset": self._compression_offset,
            "history": self._history,
            "rng_state": self._rng.bit_generator.state,
            "initial_rng_state": self._initial_rng_state,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        validated = SensorModel(
            noise_std=state.get("noise_std", self.noise_std),
            bias=state.get("bias", self.bias),
            lag_minutes=state.get("lag_minutes", self.lag_minutes),
            isf_tau_minutes=state.get("isf_tau_minutes", self.isf_tau_minutes),
            noise_ar1_phi=state.get("noise_ar1_phi", self.noise_ar1_phi),
            noise_fbm_hurst=state.get("noise_fbm_hurst", self.noise_fbm_hurst),
            dropout_prob=state.get("dropout_prob", self.dropout_prob),
            drift_std_per_hour=state.get(
                "drift_std_per_hour", self.drift_std_per_hour
            ),
            drift_max_abs_mgdl=state.get(
                "drift_max_abs_mgdl", self.drift_max_abs_mgdl
            ),
            dropout_duration_steps=state.get(
                "dropout_duration_steps", self.dropout_duration_steps
            ),
            compression_low_prob=state.get(
                "compression_low_prob", self.compression_low_prob
            ),
            compression_low_max_glucose=state.get(
                "compression_low_max_glucose", self.compression_low_max_glucose
            ),
            compression_low_mgdl_range=state.get(
                "compression_low_mgdl_range", self.compression_low_mgdl_range
            ),
            compression_low_duration_steps=state.get(
                "compression_low_duration_steps",
                self.compression_low_duration_steps,
            ),
        )
        for name in (
            "noise_std",
            "bias",
            "lag_minutes",
            "isf_tau_minutes",
            "noise_ar1_phi",
            "noise_fbm_hurst",
            "dropout_prob",
            "drift_std_per_hour",
            "drift_max_abs_mgdl",
            "dropout_duration_steps",
            "compression_low_prob",
            "compression_low_max_glucose",
            "compression_low_mgdl_range",
            "compression_low_duration_steps",
        ):
            setattr(self, name, getattr(validated, name))

        optional_dynamic = {
            "last_reading": state.get("last_reading"),
            "last_timestamp": state.get("last_timestamp"),
            "current_isf": state.get("current_isf"),
        }
        for name, value in optional_dynamic.items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"sensor snapshot {name} must be finite")
        self._last_reading = optional_dynamic["last_reading"]
        self._last_timestamp = optional_dynamic["last_timestamp"]
        self._current_isf = optional_dynamic["current_isf"]
        self._current_colored_noise = float(state.get("current_colored_noise", self._current_colored_noise))
        self._fbm_components = cast(
            NDArray[np.float64],
            np.asarray(state.get("fbm_components", self._fbm_components), dtype=np.float64),
        )
        if self._fbm_components.shape != (3,) or not np.all(
            np.isfinite(self._fbm_components)
        ):
            raise ValueError("sensor snapshot fbm_components must contain 3 finite values")
        self._drift_offset = float(state.get("drift_offset", self._drift_offset))
        self._dropout_remaining_steps = int(state.get("dropout_remaining_steps", self._dropout_remaining_steps))
        self._compression_remaining_steps = int(state.get("compression_remaining_steps", self._compression_remaining_steps))
        self._compression_offset = float(
            state.get("compression_offset", self._compression_offset)
        )
        if not all(
            np.isfinite(value)
            for value in (
                self._current_colored_noise,
                self._drift_offset,
                self._compression_offset,
            )
        ):
            raise ValueError("sensor snapshot dynamic values must be finite")
        if self._dropout_remaining_steps < 0 or self._compression_remaining_steps < 0:
            raise ValueError("sensor snapshot remaining-step counts must be non-negative")
        if self._compression_offset < 0.0:
            raise ValueError("sensor snapshot compression_offset must be non-negative")
        history = state.get("history", [])
        if not isinstance(history, list):
            raise ValueError("sensor snapshot history must be a list")
        normalized_history: list[tuple[float, float]] = []
        for item in history:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("sensor snapshot history entries must be time/value pairs")
            timestamp, glucose = float(item[0]), float(item[1])
            if not np.isfinite(timestamp) or not np.isfinite(glucose) or glucose < 0.0:
                raise ValueError("sensor snapshot history contains an invalid value")
            if normalized_history and timestamp <= normalized_history[-1][0]:
                raise ValueError("sensor snapshot history timestamps must increase")
            normalized_history.append((timestamp, glucose))
        self._history = normalized_history
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]
        if "initial_rng_state" in state:
            self._initial_rng_state = copy.deepcopy(state["initial_rng_state"])


def _normalize_step_window(value: int | Tuple[int, int] | list[int]) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("duration step window must contain exactly two values")
        start, end = int(value[0]), int(value[1])
    else:
        start = end = int(value)
    if start < 1 or end < start:
        raise ValueError("duration step window must satisfy 1 <= start <= end")
    return (start, end)


SENSOR_PROFILES: Dict[str, Dict[str, Any]] = {
    "ideal": {
        "noise_std": 0.0,
        "bias": 0.0,
        "lag_minutes": 0,
        "isf_tau_minutes": 1.0,
        "noise_ar1_phi": 0.85,
        "noise_fbm_hurst": None,
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
        "noise_std": 5.0,
        "bias": 0.0,
        "lag_minutes": 0,
        "isf_tau_minutes": 8.0,
        "noise_ar1_phi": 0.85,
        "noise_fbm_hurst": None,
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
        "lag_minutes": 0,
        "isf_tau_minutes": 10.0,
        "noise_ar1_phi": 0.85,
        "noise_fbm_hurst": 0.75,
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
        "lag_minutes": 0,
        "isf_tau_minutes": 12.0,
        "noise_ar1_phi": 0.88,
        "noise_fbm_hurst": 0.78,
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

    Supports a per-step delivery cap, dose quantization, stochastic delivery
    error, and a binary delivery-interruption scenario. These are transparent
    bench-test abstractions; they are not a validated motor or fluid model.
    """

    def __init__(
        self,
        max_units_per_step: Optional[float] = None,
        quantization_units: Optional[float] = None,
        dropout_prob: float = 0.0,
        delivery_noise_std: float = 0.0,
        step_error_probability: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        optional_positive = {
            "max_units_per_step": max_units_per_step,
            "quantization_units": quantization_units,
        }
        for name, value in optional_positive.items():
            if value is not None and (
                not np.isfinite(value) or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive when supplied")
        if not np.isfinite(dropout_prob) or not 0.0 <= float(dropout_prob) <= 1.0:
            raise ValueError("dropout_prob must be between 0 and 1")
        if not np.isfinite(delivery_noise_std) or float(delivery_noise_std) < 0.0:
            raise ValueError("delivery_noise_std must be finite and non-negative")
        if step_error_probability is not None and (
            not np.isfinite(step_error_probability)
            or not 0.0 <= float(step_error_probability) <= 1.0
        ):
            raise ValueError("step_error_probability must be between 0 and 1")

        self.max_units_per_step = (
            None if max_units_per_step is None else float(max_units_per_step)
        )
        self.quantization_units = (
            None if quantization_units is None else float(quantization_units)
        )
        self.dropout_prob = float(dropout_prob)
        self.delivery_noise_std = float(delivery_noise_std)
        self.step_error_probability = (
            None
            if step_error_probability is None
            else float(step_error_probability)
        )
        self._rng = np.random.default_rng(seed)
        self._initial_rng_state = copy.deepcopy(self._rng.bit_generator.state)

    def reset(self) -> None:
        self._rng.bit_generator.state = copy.deepcopy(self._initial_rng_state)

    def deliver(self, requested_units: float, time_step_minutes: float) -> PumpDelivery:
        if not np.isfinite(requested_units):
            raise ValueError("requested_units must be finite")
        if not np.isfinite(time_step_minutes) or time_step_minutes <= 0.0:
            raise ValueError("time_step_minutes must be finite and positive")

        delivered = float(requested_units)
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

        if self.delivery_noise_std > 0.0:
            # Dose-domain uncertainty remains in units. It is not reinterpreted
            # as a probability merely because quantization is enabled.
            delivered += float(self._rng.normal(0.0, self.delivery_noise_std))
            delivered = max(0.0, delivered)
            status = "delivery_error"
            reason = "continuous_delivery_noise"

        if self.quantization_units is not None:
            micro_steps = round(delivered / self.quantization_units)
            error_probability = self.step_error_probability

            step_error_count = 0
            if error_probability is not None and error_probability > 0.0:
                actual_steps = micro_steps
                for _ in range(int(micro_steps)):
                    if float(self._rng.random()) < error_probability:
                        # Generic missed/extra actuation event. The asymmetry is
                        # a configurable research assumption, not rotor physics.
                        if float(self._rng.random()) < 0.7:
                            actual_steps -= 1
                        else:
                            actual_steps += 1
                        step_error_count += 1
                micro_steps = max(0, actual_steps)

            delivered = micro_steps * self.quantization_units
            delivered = max(0.0, delivered)
            if step_error_count:
                status = "delivery_error"
                reason = f"quantized_step_errors={step_error_count}"
        if self.dropout_prob > 0 and float(self._rng.random()) < self.dropout_prob:
            delivered = 0.0
            status = "delivery_interruption"
            reason = "configured_dropout"

        if self.max_units_per_step is not None:
            delivered = min(delivered, self.max_units_per_step)

        return PumpDelivery(delivered_units=delivered, status=status, reason=reason)

    def get_state(self) -> Dict[str, Any]:
        return {
            "max_units_per_step": self.max_units_per_step,
            "quantization_units": self.quantization_units,
            "dropout_prob": self.dropout_prob,
            "delivery_noise_std": self.delivery_noise_std,
            "step_error_probability": self.step_error_probability,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "initial_rng_state": copy.deepcopy(self._initial_rng_state),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        validated = PumpModel(
            max_units_per_step=state.get(
                "max_units_per_step", self.max_units_per_step
            ),
            quantization_units=state.get(
                "quantization_units", self.quantization_units
            ),
            dropout_prob=state.get("dropout_prob", self.dropout_prob),
            delivery_noise_std=state.get(
                "delivery_noise_std", self.delivery_noise_std
            ),
            step_error_probability=state.get(
                "step_error_probability", self.step_error_probability
            ),
        )
        self.max_units_per_step = validated.max_units_per_step
        self.quantization_units = validated.quantization_units
        self.dropout_prob = validated.dropout_prob
        self.delivery_noise_std = validated.delivery_noise_std
        self.step_error_probability = validated.step_error_probability
        if "initial_rng_state" in state:
            self._initial_rng_state = copy.deepcopy(state["initial_rng_state"])
        if "rng_state" in state:
            self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
