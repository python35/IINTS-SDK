import logging
import pandas as pd
import json
import time
from typing import Callable
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
from iints.core.patient.models import PatientModel
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput
from iints.core.supervisor import IndependentSupervisor, SafetyLevel
from iints.core.safety import InputValidator, SafetyConfig
from iints.core.safety.config import (
    SIMULATION_GLUCOSE_CEILING_MGDL,
    SIMULATION_GLUCOSE_FLOOR_MGDL,
    SENSOR_MAX_GLUCOSE_RATE_PER_MIN_MGDL,
)
from iints.core.devices.models import SensorModel, PumpModel
from iints.core.physiology_variation import EmpiricalResidualModel
import numpy as np

logger = logging.getLogger("iints")

class SimulationLimitError(RuntimeError):
    """Raised when a simulation violates critical safety limits."""

    def __init__(self, message: str, current_time: float, glucose_value: float, duration_minutes: float):
        super().__init__(message)
        self.current_time = current_time
        self.glucose_value = glucose_value
        self.duration_minutes = duration_minutes

class StressEvent:
    """Represents a discrete event that can occur during a simulation for stress testing."""
    def __init__(
        self,
        start_time: int,
        event_type: str,
        value: Any = None,
        reported_value: Any = None,
        absorption_delay_minutes: int = 0,
        duration: int = 0,
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        basal_rate: Optional[float] = None,
        dia_minutes: Optional[float] = None,
    ) -> None:
        """
        Args:
            start_time (int): The simulation time (in minutes) when the event should occur.
            event_type (str): Type of event (e.g., 'meal', 'missed_meal', 'sensor_error', 'exercise').
            value (Any): Value associated with the event (e.g., carb amount, error value).
            reported_value (Any): Value reported to the algorithm (if different from actual value, e.g., for "prul" scenarios).
            absorption_delay_minutes (int): Delay in minutes before carbohydrates from this event start affecting glucose.
            duration (int): The duration of the event in minutes (e.g., for exercise).
        """
        self.start_time = start_time
        self.event_type = event_type
        self.value = value
        self.reported_value = reported_value # New attribute
        self.absorption_delay_minutes = absorption_delay_minutes # New attribute
        self.duration = duration
        self.isf = isf
        self.icr = icr
        self.basal_rate = basal_rate
        self.dia_minutes = dia_minutes

    def __str__(self) -> str:
        reported_str = f", Reported: {self.reported_value}" if self.reported_value is not None else ""
        ratio_str = ""
        if self.event_type == "ratio_change":
            parts = []
            if self.isf is not None:
                parts.append(f"ISF={self.isf}")
            if self.icr is not None:
                parts.append(f"ICR={self.icr}")
            if self.basal_rate is not None:
                parts.append(f"Basal={self.basal_rate}")
            if self.dia_minutes is not None:
                parts.append(f"DIA={self.dia_minutes}")
            if parts:
                ratio_str = ", " + ", ".join(parts)
        delay_str = f", Delay: {self.absorption_delay_minutes}m" if self.absorption_delay_minutes > 0 else ""
        duration_str = f", Duration: {self.duration}m" if self.duration > 0 else ""
        return f"Event(Time: {self.start_time}m, Type: {self.event_type}, Value: {self.value}{reported_str}{ratio_str}{delay_str}{duration_str})"

class Simulator:
    """
    Orchestrates the interaction between a patient model and an insulin algorithm
    over a simulated period, including stress-test scenarios.
    """
    def __init__(
        self,
        patient_model: PatientModel,
        algorithm: InsulinAlgorithm,
        time_step: int = 5,
        seed: Optional[int] = None,
        audit_log_path: Optional[str] = None,
        enable_profiling: bool = False,
        sensor_model: Optional[SensorModel] = None,
        pump_model: Optional[PumpModel] = None,
        on_step: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        on_safety_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        critical_glucose_threshold: float = 40.0,
        critical_glucose_duration_minutes: int = 30,
        safety_config: Optional[SafetyConfig] = None,
        predictor: Optional[object] = None,
        physiology_variation_model: Optional[EmpiricalResidualModel] = None,
    ) -> None:
        """
        Initializes the simulator.

        Args:
            patient_model (PatientModel): The patient model to simulate.
            algorithm (InsulinAlgorithm): The insulin delivery algorithm to test.
            time_step (int): The duration of each simulation step in minutes.
            seed (Optional[int]): Random seed for reproducible simulations.
            audit_log_path (Optional[str]): If provided, path to write a detailed JSON audit log.
        """
        self.patient_model = patient_model
        self.algorithm = algorithm
        self.time_step = time_step
        self.simulation_data: List[Any] = [] # To store results
        self.explainable_events_log: List[str] = [] # Store XAI strings
        self.stress_events: List[StressEvent] = []
        self.seed = seed
        self.safety_config = safety_config or SafetyConfig()
        self.supervisor = IndependentSupervisor(safety_config=self.safety_config)
        # Keep the raw validator for older snapshots/audits, but do not apply
        # CGM rate limits to hidden patient truth; only algorithm-facing sensor
        # inputs are plausibility-filtered.
        self.raw_input_validator = InputValidator(safety_config=self.safety_config)
        self.input_validator = InputValidator(safety_config=self.safety_config)
        self.sensor_model = sensor_model or SensorModel(seed=seed)
        self.pump_model = pump_model or PumpModel(seed=seed)
        self.physiology_variation_model = physiology_variation_model
        self.predictor = predictor
        self._predictor_history: List[Dict[str, float]] = []
        self._predictor_feature_columns: List[str] = [
            "glucose_actual_mgdl",
            "patient_iob_units",
            "patient_cob_grams",
            "effective_isf",
            "effective_icr",
            "effective_basal_rate_u_per_hr",
            "glucose_trend_mgdl_min",
        ]
        self._predictor_history_steps = max(int(240 / float(self.time_step)), 1)
        self._predictor_horizon_steps = max(
            int(self.safety_config.predicted_hypoglycemia_horizon_minutes / float(self.time_step)),
            1,
        )
        self._init_predictor_settings()
        self.on_step = on_step
        self.on_safety_event = on_safety_event
        self.meal_queue: List[Dict[str, Any]] = [] # Initialize meal queue for delayed absorption
        self.audit_log_path = audit_log_path
        self.enable_profiling = enable_profiling
        if self.safety_config is not None:
            critical_glucose_threshold = self.safety_config.critical_glucose_threshold
            critical_glucose_duration_minutes = self.safety_config.critical_glucose_duration_minutes
        self.critical_glucose_threshold = critical_glucose_threshold
        self.critical_glucose_duration_minutes = critical_glucose_duration_minutes
        self._critical_low_minutes = 0
        self._current_time = 0
        self._resume_state = False
        self._termination_info: Optional[Dict[str, Any]] = None
        self._ratio_overrides: List[Dict[str, Any]] = []
        self._base_ratio_state: Optional[Dict[str, float]] = None
        self._previous_glucose_for_trend: Optional[float] = None
        self._glucose_trend_history: List[Tuple[float, float]] = []
        self._xai_previous_glucose_trend: Optional[float] = None
        self._xai_last_event_time: Dict[str, float] = {}
        self._profiling_samples: Dict[str, List[float]] = {
            "algorithm_latency_ms": [],
            "supervisor_latency_ms": [],
            "step_latency_ms": [],
        }
        # Layer-1 telemetry (InputValidator)
        self._input_validator_fail_soft_count = 0
        self._input_validator_negative_insulin_clamp_count = 0
        self._glucagon_safety_interventions_count = 0
        self._glucagon_dose_history: List[Tuple[float, float]] = []
        self._input_validator_last_error: Optional[str] = None
        self._input_validator_last_step_fail_soft = False
        if self.audit_log_path:
            # Clear the log file at the beginning of a simulation run
            try:
                with open(self.audit_log_path, 'w') as f:
                    f.write("") # Overwrite the file
            except IOError as e:
                logger.warning("Could not clear audit log file at %s. Error: %s", self.audit_log_path, e)

    def _init_predictor_settings(self) -> None:
        if self.predictor is None:
            return
        try:
            config = getattr(self.predictor, "config", None)
            if isinstance(config, dict):
                feature_columns = config.get("feature_columns")
                history_steps = config.get("history_steps")
                horizon_steps = config.get("horizon_steps")
            else:
                feature_columns = getattr(self.predictor, "feature_columns", None)
                history_steps = getattr(self.predictor, "history_steps", None)
                horizon_steps = getattr(self.predictor, "horizon_steps", None)

            if isinstance(feature_columns, list) and feature_columns:
                self._predictor_feature_columns = [str(col) for col in feature_columns]
            if isinstance(history_steps, int) and history_steps > 0:
                self._predictor_history_steps = history_steps
            if isinstance(horizon_steps, int) and horizon_steps > 0:
                self._predictor_horizon_steps = horizon_steps
        except Exception:
            return

    def _predictor_ood_status(self, current_vector: np.ndarray) -> Dict[str, Any]:
        if self.predictor is None:
            return {
                "in_distribution": True,
                "max_zscore": 0.0,
                "feature_fraction_over_threshold": 0.0,
            }
        scaler = getattr(self.predictor, "scaler", None)
        center = getattr(scaler, "_center", None)
        scale = getattr(scaler, "_scale", None)
        if center is None or scale is None:
            return {
                "in_distribution": True,
                "max_zscore": 0.0,
                "feature_fraction_over_threshold": 0.0,
            }

        safe_scale = np.where(np.abs(scale) < 1e-8, 1.0, np.abs(scale))
        z_scores = np.abs((current_vector - center) / safe_scale)
        max_zscore = float(np.max(z_scores))
        threshold = float(self.safety_config.predictor_ood_zscore_threshold)
        fraction = float(np.mean(z_scores > threshold))
        in_distribution = not (
            self.safety_config.predictor_ood_gate_enabled
            and (
                max_zscore > threshold
                or fraction > float(self.safety_config.predictor_ood_max_feature_fraction)
            )
        )
        return {
            "in_distribution": in_distribution,
            "max_zscore": max_zscore,
            "feature_fraction_over_threshold": fraction,
        }

    def _predict_with_model(self, feature_row: Dict[str, float], fallback: float) -> Tuple[float, float, Dict[str, Any]]:
        """
        Returns (ai_prediction, heuristic_prediction, meta). If model fails or lacks history,
        ai_prediction falls back to heuristic_prediction.
        """
        meta: Dict[str, Any] = {
            "predictor_used": False,
            "predictor_gate_reason": "none",
            "predictor_uncertainty_std_mgdl": None,
            "predictor_in_distribution": True,
            "predictor_ood_max_zscore": 0.0,
            "predictor_ood_feature_fraction": 0.0,
        }
        self._predictor_history.append(feature_row)
        if len(self._predictor_history) < self._predictor_history_steps:
            meta["predictor_gate_reason"] = "warmup"
            return fallback, fallback, meta

        history_slice = self._predictor_history[-self._predictor_history_steps :]
        try:
            import numpy as np

            X = np.zeros((1, self._predictor_history_steps, len(self._predictor_feature_columns)), dtype=float)
            for i, row in enumerate(history_slice):
                for j, col in enumerate(self._predictor_feature_columns):
                    X[0, i, j] = float(row.get(col, 0.0))

            ood_status = self._predictor_ood_status(X[0, -1, :])
            meta["predictor_in_distribution"] = bool(ood_status["in_distribution"])
            meta["predictor_ood_max_zscore"] = float(ood_status["max_zscore"])
            meta["predictor_ood_feature_fraction"] = float(ood_status["feature_fraction_over_threshold"])
            if not bool(ood_status["in_distribution"]):
                meta["predictor_gate_reason"] = "ood_gate"
                return fallback, fallback, meta

            uncertainty_std: Optional[float] = None
            predict_uncertainty_fn = getattr(self.predictor, "predict_with_uncertainty", None)
            if callable(predict_uncertainty_fn):
                mean_pred, std_pred = predict_uncertainty_fn(
                    X,
                    n_samples=int(self.safety_config.predictor_mc_dropout_samples),
                )
                mean_arr = np.array(mean_pred, dtype=float)
                std_arr = np.array(std_pred, dtype=float)
                if mean_arr.ndim == 2:
                    candidate = float(mean_arr[0, -1])
                elif mean_arr.ndim == 1:
                    candidate = float(mean_arr[-1])
                else:
                    candidate = float(fallback)
                if std_arr.ndim == 2:
                    uncertainty_std = float(std_arr[0, -1])
                elif std_arr.ndim == 1:
                    uncertainty_std = float(std_arr[-1])
                else:
                    uncertainty_std = None
                meta["predictor_uncertainty_std_mgdl"] = uncertainty_std
                if (
                    self.safety_config.predictor_uncertainty_gate_enabled
                    and uncertainty_std is not None
                    and uncertainty_std > float(self.safety_config.predictor_uncertainty_max_std_mgdl)
                ):
                    meta["predictor_gate_reason"] = "uncertainty_gate"
                    return fallback, fallback, meta
                meta["predictor_used"] = True
                return candidate, fallback, meta

            predict_fn = getattr(self.predictor, "predict", None)
            if not callable(predict_fn):
                meta["predictor_gate_reason"] = "missing_predict_function"
                return fallback, fallback, meta
            output = predict_fn(X)
            if hasattr(output, "shape"):
                output_arr = np.array(output, dtype=float)
                if output_arr.ndim == 2:
                    meta["predictor_used"] = True
                    return float(output_arr[0, -1]), fallback, meta
                if output_arr.ndim == 1:
                    meta["predictor_used"] = True
                    return float(output_arr[-1]), fallback, meta
            if isinstance(output, (list, tuple)) and output:
                meta["predictor_used"] = True
                return float(output[-1]), fallback, meta
            if isinstance(output, (float, int)):
                meta["predictor_used"] = True
                return float(output), fallback, meta
        except Exception:
            meta["predictor_gate_reason"] = "predictor_exception"
            return fallback, fallback, meta
        return fallback, fallback, meta

    def _write_audit_log(self, data: Dict[str, Any]) -> None:
        """Writes a single step's audit data to the JSON log file."""
        if self.audit_log_path:
            try:
                with open(self.audit_log_path, 'a') as f:
                    f.write(json.dumps(data, default=str) + '\n')
            except IOError as e:
                logger.warning("Could not write to audit log file at %s. Error: %s", self.audit_log_path, e)

    def _emit_safety_event(self, payload: Dict[str, Any]) -> None:
        if not self.on_safety_event:
            return
        try:
            self.on_safety_event(payload)
        except Exception as exc:
            logger.warning("on_safety_event callback raised: %s", exc)

    def _evaluate_glucagon_safety(
        self,
        *,
        requested_glucagon_mg: float,
        current_glucose: float,
        predicted_glucose_30min: float,
        current_time: float,
    ) -> Dict[str, Any]:
        """Apply research-only safety rails to simulated glucagon outputs."""
        actions: List[str] = []
        raw_requested = requested_glucagon_mg
        try:
            requested = float(requested_glucagon_mg)
        except (TypeError, ValueError):
            requested = 0.0
            actions.append("INVALID_GLUCAGON_REQUEST_CLAMPED")

        if not np.isfinite(requested):
            requested = 0.0
            actions.append("NONFINITE_GLUCAGON_REQUEST_CLAMPED")
        requested = max(0.0, requested)
        approved = requested

        if approved <= 0.0:
            return {
                "requested_glucagon_mg": max(0.0, float(raw_requested)) if isinstance(raw_requested, (int, float)) and np.isfinite(raw_requested) else 0.0,
                "approved_glucagon_mg": 0.0,
                "glucagon_reduction_mg": 0.0,
                "glucagon_safety_triggered": bool(actions),
                "glucagon_safety_reason": "; ".join(actions),
                "glucagon_safety_actions": actions,
            }

        # Glucagon is only a rescue actuator in this simulator. If both the
        # current and predicted glucose are safely above low thresholds, block it.
        if (
            current_glucose > float(self.safety_config.glucagon_allowed_above_glucose_mgdl)
            and predicted_glucose_30min > float(self.safety_config.predicted_hypoglycemia_threshold)
        ):
            approved = 0.0
            actions.append(
                "GLUCAGON_BLOCKED_NOT_LOW_RISK: "
                f"glucose {current_glucose:.1f} mg/dL, predicted {predicted_glucose_30min:.1f} mg/dL"
            )

        max_step = float(self.safety_config.max_glucagon_per_step_mg)
        if approved > max_step:
            approved = max_step
            actions.append(f"GLUCAGON_STEP_CAP: capped at {max_step:.2f} mg")

        self._glucagon_dose_history = [
            (t, d) for (t, d) in self._glucagon_dose_history if current_time - t < 60.0
        ]
        total_last_60 = sum(d for _, d in self._glucagon_dose_history)
        max_hour = float(self.safety_config.max_glucagon_per_hour_mg)
        if total_last_60 + approved > max_hour:
            approved = max(0.0, max_hour - total_last_60)
            actions.append(f"GLUCAGON_60MIN_CAP: capped by {max_hour:.2f} mg/hour limit")

        if approved > 0.0:
            self._glucagon_dose_history.append((current_time, approved))

        triggered = bool(actions) or approved < requested
        if triggered:
            self._glucagon_safety_interventions_count += 1

        return {
            "requested_glucagon_mg": requested,
            "approved_glucagon_mg": approved,
            "glucagon_reduction_mg": max(0.0, requested - approved),
            "glucagon_safety_triggered": triggered,
            "glucagon_safety_reason": "; ".join(actions),
            "glucagon_safety_actions": actions,
        }

    def _validate_glucose_fail_soft(
        self,
        glucose_value: float,
        current_time: float,
        source: str,
        *,
        validator: Optional[InputValidator] = None,
    ) -> float:
        active_validator = validator or self.input_validator
        try:
            return active_validator.validate_glucose(glucose_value, current_time)
        except ValueError as exc:
            self._input_validator_fail_soft_count += 1
            self._input_validator_last_error = str(exc)
            self._input_validator_last_step_fail_soft = True
            fallback = active_validator.last_valid_glucose
            incoming_value = float(glucose_value)
            if not np.isfinite(incoming_value):
                incoming_value = float(fallback) if fallback is not None else active_validator.min_glucose
            if fallback is not None and active_validator.last_validation_time is not None:
                time_delta = max(float(current_time) - float(active_validator.last_validation_time), 0.0)
                if time_delta > 0.0:
                    allowed_delta = active_validator.max_glucose_delta_per_5_min * (time_delta / 5.0)
                    requested_delta = incoming_value - float(fallback)
                    fallback = float(fallback) + float(
                        max(-allowed_delta, min(requested_delta, allowed_delta))
                    )
            if fallback is None:
                fallback = incoming_value
            fallback = float(max(active_validator.min_glucose, min(fallback, active_validator.max_glucose)))
            self._write_audit_log(
                {
                    "timestamp": current_time,
                    "event": "glucose_validation_fail_soft",
                    "source": source,
                    "input_value": glucose_value,
                    "fallback_value": fallback,
                    "error": str(exc),
                }
            )
            active_validator.last_valid_glucose = fallback
            active_validator.last_validation_time = current_time
            return fallback

    def _bound_simulation_glucose(self, glucose_value: float, current_time: float) -> float:
        """
        Keep the hidden physiological state numerically bounded without applying
        CGM input rate limits to the patient truth trace.

        The input validator is intentionally strict for sensor values that reach
        an algorithm. The simulated patient state can move faster during stress
        tests, so rate-limiting it would hide the very physiology we need to
        evaluate.
        """
        if not np.isfinite(glucose_value):
            fallback = float(self.patient_model.get_current_glucose())
            self._write_audit_log(
                {
                    "timestamp": current_time,
                    "event": "simulation_glucose_non_finite",
                    "input_value": glucose_value,
                    "fallback_value": fallback,
                }
            )
            return fallback
        clipped = float(
            np.clip(
                glucose_value,
                SIMULATION_GLUCOSE_FLOOR_MGDL,
                SIMULATION_GLUCOSE_CEILING_MGDL,
            )
        )
        if clipped != float(glucose_value):
            self._write_audit_log(
                {
                    "timestamp": current_time,
                    "event": "simulation_glucose_clipped",
                    "input_value": glucose_value,
                    "clipped_value": clipped,
                }
            )
        return clipped

    def _update_glucose_trend(self, current_time: float, glucose_value: float) -> float:
        """
        Estimate CGM trend with a short rolling slope instead of one noisy delta.

        A single noisy 5-minute CGM drop can otherwise look like an impossible
        30-minute hypoglycemia trajectory. A rolling 20-minute regression keeps
        the signal responsive while avoiding startup/noise spikes.
        """
        self._glucose_trend_history.append((float(current_time), float(glucose_value)))
        window_minutes = max(20.0, float(self.time_step) * 3.0)
        cutoff = float(current_time) - window_minutes
        self._glucose_trend_history = [
            (t, g) for (t, g) in self._glucose_trend_history if t >= cutoff
        ]
        self._previous_glucose_for_trend = float(glucose_value)

        if len(self._glucose_trend_history) < 3:
            return 0.0

        times = np.array([t for t, _ in self._glucose_trend_history], dtype=float)
        values = np.array([g for _, g in self._glucose_trend_history], dtype=float)
        if float(times[-1] - times[0]) < max(float(self.time_step) * 2.0, 10.0):
            return 0.0

        centered_time = times - float(times.mean())
        denom = float(np.dot(centered_time, centered_time))
        if denom <= 0.0:
            return 0.0

        slope = float(np.dot(centered_time, values - float(values.mean())) / denom)
        max_rate = min(
            float(getattr(self.safety_config, "glucose_rate_alarm", SENSOR_MAX_GLUCOSE_RATE_PER_MIN_MGDL)),
            SENSOR_MAX_GLUCOSE_RATE_PER_MIN_MGDL,
        )
        return float(np.clip(slope, -max_rate, max_rate))

    def _should_emit_xai_event(self, event_key: str, current_time: float, cooldown_minutes: float) -> bool:
        """Return True when an explanation is new enough to avoid event spam."""
        current = float(current_time)
        last = self._xai_last_event_time.get(event_key)
        if last is not None and current - last < float(cooldown_minutes):
            return False
        self._xai_last_event_time[event_key] = current
        return True

    def _build_explainable_events(
        self,
        *,
        current_time: float,
        actual_glucose_reading: float,
        glucose_trend: float,
        predicted_glucose_30: float,
        safety_triggered: bool,
        safety_result: Dict[str, Any],
    ) -> List[str]:
        """Build human-readable research explanations for notable simulation events."""
        step_events: List[str] = []
        time_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
        previous_trend = self._xai_previous_glucose_trend

        if (
            self.patient_model.carbs_on_board > 5.0
            and glucose_trend > 0.5
            and previous_trend is not None
            and previous_trend <= 0.5
            and self._should_emit_xai_event("meal_response", current_time, 45.0)
        ):
            step_events.append(f"At {time_str} glucose started rising after meal/breakfast.")

        if (
            self.patient_model.carbs_on_board > 1.0
            and len(self._predictor_history) >= self._predictor_horizon_steps
        ):
            past_prediction = self._predictor_history[-self._predictor_horizon_steps].get(
                "predicted_glucose_heuristic_30min"
            )
            if (
                past_prediction is not None
                and (actual_glucose_reading - past_prediction) > 15.0
                and self._should_emit_xai_event("absorption_anomaly", current_time, 30.0)
            ):
                step_events.append(f"At {time_str} the model detected faster-than-expected absorption.")

        if (
            glucose_trend < -1.0
            and self.patient_model.insulin_on_board > 1.0
            and actual_glucose_reading < 120.0
            and (previous_trend is None or previous_trend >= -1.0)
            and self._should_emit_xai_event("hypo_risk", current_time, 30.0)
        ):
            step_events.append(
                f"At {time_str} hypo risk increased because downward trend combined with active insulin."
            )

        insulin_reduction = float(safety_result.get("insulin_reduction", 0.0) or 0.0)
        if (
            safety_triggered
            and insulin_reduction > 0.0
            and self._should_emit_xai_event("supervisor_intervention", current_time, 15.0)
        ):
            safety_reason_text = str(safety_result.get("safety_reason", "")).lower()
            hypo_threshold = float(self.safety_config.hypoglycemia_threshold)
            predicted_low = predicted_glucose_30 < hypo_threshold
            low_reason = any(token in safety_reason_text for token in ("low", "hypo", "predicted"))
            if predicted_low or low_reason:
                step_events.append(
                    f"At {time_str} supervisor intervention prevented predicted glucose below "
                    f"{hypo_threshold:.0f} mg/dL."
                )
            else:
                step_events.append(
                    f"At {time_str} supervisor intervention reduced insulin after an independent safety check."
                )

        return step_events

    def _apply_ratio_overrides(self, current_time: float) -> Dict[str, float]:
        if self._base_ratio_state is None:
            self._base_ratio_state = self.patient_model.get_ratio_state()
        effective = dict(self._base_ratio_state)
        active = [
            override
            for override in self._ratio_overrides
            if override["start_time"] <= current_time <= override["end_time"]
        ]
        if active:
            latest = active[-1]
            if latest.get("isf") is not None:
                effective["isf"] = latest["isf"]
            if latest.get("icr") is not None:
                effective["icr"] = latest["icr"]
            if latest.get("basal_rate_u_per_hr") is not None:
                effective["basal_rate_u_per_hr"] = latest["basal_rate_u_per_hr"]
            if latest.get("dia_minutes") is not None:
                effective["dia_minutes"] = latest["dia_minutes"]

        self.patient_model.set_ratio_state(
            isf=effective.get("isf"),
            icr=effective.get("icr"),
            basal_rate=effective.get("basal_rate_u_per_hr"),
            dia_minutes=effective.get("dia_minutes"),
        )
        try:
            if effective.get("isf") is not None:
                self.algorithm.set_isf(float(effective["isf"]))
            if effective.get("icr") is not None:
                self.algorithm.set_icr(float(effective["icr"]))
        except Exception:
            # Algorithm may ignore dynamic ratio updates; no hard failure.
            pass
        return effective

    def _predict_glucose(
        self,
        current_glucose: float,
        trend_mgdl_min: float,
        iob_units: float,
        cob_grams: float,
        isf: float,
        icr: float,
        dia_minutes: float,
        horizon_minutes: int,
        carb_absorption_minutes: float,
    ) -> float:
        trend_component = trend_mgdl_min * horizon_minutes

        insulin_component = 0.0
        if dia_minutes > 0:
            insulin_component = -iob_units * isf * min(horizon_minutes / dia_minutes, 1.0)

        carb_component = 0.0
        if icr > 0:
            carb_effect_per_gram = isf / icr
            absorption_fraction = 1.0 if carb_absorption_minutes <= 0 else min(horizon_minutes / carb_absorption_minutes, 1.0)
            carb_component = cob_grams * carb_effect_per_gram * absorption_fraction

        return current_glucose + trend_component + insulin_component + carb_component

    def add_stress_event(self, event: StressEvent) -> None:
        """Adds a stress event to be triggered during the simulation."""
        self.stress_events.append(event)
        self.stress_events.sort(key=lambda e: e.start_time) # Keep events sorted by time

    def run(self, duration_minutes: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Alias for run_batch to ensure backward compatibility."""
        return self.run_batch(duration_minutes)

    def run_batch(self, duration_minutes: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs the entire simulation and returns the results as a single DataFrame.

        Args:
            duration_minutes (int): Total simulation duration in minutes.

        Returns:
            pd.DataFrame: A DataFrame containing the complete simulation results.
            Dict[str, Any]: A dictionary containing the safety report from the supervisor.
        """
        logger.info("Starting batch simulation for %d minutes...", duration_minutes)
        all_records: List[Dict[str, Any]] = []
        try:
            for record in self.run_live(duration_minutes):
                all_records.append(record)
        except SimulationLimitError as err:
            logger.error("Simulation terminated early: %s", err)
            self._termination_info = {
                "reason": str(err),
                "current_time_minutes": err.current_time,
                "glucose_value": err.glucose_value,
                "duration_minutes": err.duration_minutes,
            }
        simulation_results_df = pd.DataFrame(all_records)
        safety_report = self.supervisor.get_safety_report()
        if self._termination_info:
            safety_report["terminated_early"] = True
            safety_report["termination_reason"] = self._termination_info
        safety_report["input_validator_fail_soft_count"] = int(self._input_validator_fail_soft_count)
        safety_report["input_validator_negative_insulin_clamp_count"] = int(
            self._input_validator_negative_insulin_clamp_count
        )
        safety_report["glucagon_safety_interventions_count"] = int(
            self._glucagon_safety_interventions_count
        )
        safety_report["input_validator_last_error"] = self._input_validator_last_error
        if self.enable_profiling:
            safety_report["performance_report"] = self._build_performance_report()
        logger.info("Batch simulation completed. %d records generated.", len(simulation_results_df))
        return simulation_results_df, safety_report

    def export_audit_trail(self, simulation_results_df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """
        Export audit trail as JSONL, CSV, and summary JSON.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        audit_columns = [
            "time_minutes",
            "glucose_actual_mgdl",
            "glucose_to_algo_mgdl",
            "algo_recommended_insulin_units",
            "delivered_insulin_units",
            "algo_recommended_glucagon_mg",
            "delivered_glucagon_mg",
            "glucagon_safety_reason",
            "glucagon_safety_triggered",
            "safety_reason",
            "safety_triggered",
            "supervisor_latency_ms",
            "input_validator_fail_soft",
            "sensor_status",
            "pump_status",
            "pump_reason",
            "human_intervention",
        ]
        available_columns = [c for c in audit_columns if c in simulation_results_df.columns]
        audit_df = simulation_results_df[available_columns].copy()

        jsonl_path = output_path / "audit_trail.jsonl"
        csv_path = output_path / "audit_trail.csv"
        summary_path = output_path / "audit_summary.json"

        audit_df.to_json(jsonl_path, orient="records", lines=True)
        audit_df.to_csv(csv_path, index=False)

        overrides = audit_df[audit_df.get("safety_triggered", False) == True] if "safety_triggered" in audit_df.columns else audit_df.iloc[0:0]
        reasons = overrides["safety_reason"].value_counts().to_dict() if "safety_reason" in overrides.columns else {}

        summary = {
            "total_steps": int(len(audit_df)),
            "total_overrides": int(len(overrides)),
            "top_reasons": reasons,
            "input_validator_fail_soft_count": int(
                audit_df["input_validator_fail_soft"].fillna(False).astype(bool).sum()
            ) if "input_validator_fail_soft" in audit_df.columns else 0,
        }
        if self._termination_info:
            summary["terminated_early"] = True
            summary["termination_reason"] = self._termination_info
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "summary": str(summary_path),
        }

    def run_live(self, duration_minutes: int) -> Generator[Dict[str, Any], None, None]:
        """
        Runs the simulation as a generator, yielding the record of each time step.

        Args:
            duration_minutes (int): Total simulation duration in minutes.

        Yields:
            Dict[str, Any]: The data record for each simulation time step.
        """
        if not self._resume_state:
            self.patient_model.reset()
            self.algorithm.reset()
            self.supervisor.reset()
            self.raw_input_validator.reset()
            self.input_validator.reset() # Reset input validator for new run
            self.sensor_model.reset()
            self.pump_model.reset()
            if self.physiology_variation_model is not None:
                self.physiology_variation_model.reset()
            self.simulation_data = []
            self.explainable_events_log = []
            self.meal_queue = [] # Reset meal queue for new run
            self._predictor_history = []
            self._critical_low_minutes = 0
            self._current_time = 0
            self._termination_info = None
            self._ratio_overrides = []
            self._base_ratio_state = self.patient_model.get_ratio_state()
            self._previous_glucose_for_trend = None
            self._glucose_trend_history = []
            self._xai_previous_glucose_trend = None
            self._xai_last_event_time = {}
            self._input_validator_fail_soft_count = 0
            self._input_validator_negative_insulin_clamp_count = 0
            self._glucagon_safety_interventions_count = 0
            self._glucagon_dose_history = []
            self._input_validator_last_error = None
            self._input_validator_last_step_fail_soft = False
        else:
            self._resume_state = False
        if self.enable_profiling:
            self._profiling_samples = {
                "algorithm_latency_ms": [],
                "supervisor_latency_ms": [],
                "step_latency_ms": [],
            }
        current_time = self._current_time

        logger.debug("Starting live simulation loop.")

        while current_time <= duration_minutes:
            self._current_time = current_time
            if self.enable_profiling:
                step_start_time = time.perf_counter()
            patient_carb_absorbed_this_step = 0.0
            patient_carb_event_this_step = 0.0
            algo_carb_intake_this_step = 0.0
            self._input_validator_last_step_fail_soft = False
            mechanistic_glucose_reading = self.patient_model.get_current_glucose()
            physiology_residual = 0.0
            if self.physiology_variation_model is not None:
                physiology_residual = self.physiology_variation_model.offset_at(float(current_time))
            actual_glucose_reading = mechanistic_glucose_reading + physiology_residual
            actual_glucose_reading = self._bound_simulation_glucose(
                actual_glucose_reading,
                float(current_time),
            )
            sensor_reading = self.sensor_model.read(actual_glucose_reading, float(current_time))
            glucose_to_algorithm = sensor_reading.value

            # Process newly triggered stress events
            events_to_process_now = [] # Events that affect algorithm input immediately (e.g., reported carbs, sensor error)
            events_to_queue_for_patient = [] # Meal events that affect patient after delay

            for event in self.stress_events:
                if current_time == event.start_time:
                    logger.info("[%d min] Triggering stress event: %s", current_time, event)
                    if event.event_type == 'meal':
                        events_to_queue_for_patient.append(event)
                        patient_carb_event_this_step += float(event.value or 0.0)
                        # Algorithm gets carb info based on reported_value if available, otherwise actual value
                        algo_carb_intake_this_step += event.reported_value if event.reported_value is not None else event.value
                    elif event.event_type == 'missed_meal':
                        # Patient consumes carbs, algorithm not aware.
                        events_to_queue_for_patient.append(event)
                        patient_carb_event_this_step += float(event.value or 0.0)
                        algo_carb_intake_this_step = 0.0 # Algorithm gets 0 carbs
                    elif event.event_type == 'sensor_error':
                        glucose_to_algorithm = event.value
                    elif event.event_type == 'exercise':
                        self.patient_model.start_exercise(event.value)
                        # Schedule the end of the exercise
                        end_event = StressEvent(start_time=current_time + event.duration, event_type='exercise_end')
                        self.add_stress_event(end_event)
                    elif event.event_type == 'exercise_end':
                        self.patient_model.stop_exercise()
                    elif event.event_type in {'stress', 'illness'}:
                        start_stress = getattr(self.patient_model, 'start_stress', None)
                        if callable(start_stress):
                            start_stress(float(event.value or 0.0))
                            end_type = f"{event.event_type}_end"
                            end_event = StressEvent(
                                start_time=current_time + event.duration,
                                event_type=end_type,
                            )
                            self.add_stress_event(end_event)
                    elif event.event_type in {'stress_end', 'illness_end'}:
                        stop_stress = getattr(self.patient_model, 'stop_stress', None)
                        if callable(stop_stress):
                            stop_stress()
                    elif event.event_type == 'ratio_change':
                        duration = event.duration if event.duration > 0 else float("inf")
                        self._ratio_overrides.append(
                            {
                                "start_time": current_time,
                                "end_time": current_time + duration,
                                "isf": event.isf,
                                "icr": event.icr,
                                "basal_rate_u_per_hr": event.basal_rate,
                                "dia_minutes": event.dia_minutes,
                            }
                        )

                    events_to_process_now.append(event) # Mark for removal from stress_events list

            # Remove processed stress events that don't need to be queued for patient absorption
            self.stress_events = [e for e in self.stress_events if e not in events_to_process_now]

            # Add newly triggered meal events to the meal queue for delayed absorption
            for event in events_to_queue_for_patient:
                total_carbs = float(event.value or 0.0)
                self.meal_queue.append({
                    'event': event,
                    'absorption_start_time': current_time + event.absorption_delay_minutes,
                    'remaining_carbs': total_carbs,
                    'total_carbs': total_carbs,
                })

            # Process meals from the queue that are now ready for absorption
            updated_meal_queue = []
            for meal_entry in self.meal_queue:
                if current_time >= meal_entry['absorption_start_time']:
                    event = meal_entry['event']
                    remaining_carbs = float(meal_entry.get('remaining_carbs', event.value or 0.0))
                    # The patient models already contain the physiological
                    # digestion curve. The simulator queue models only the
                    # gastric delay before that curve starts; spreading carbs
                    # here as well double-damped meals and produced flat,
                    # late, non-physiological traces.
                    patient_carb_absorbed_this_step += remaining_carbs
                    meal_entry['remaining_carbs'] = 0.0
                if float(meal_entry.get('remaining_carbs', 0.0)) > 1e-6:
                    updated_meal_queue.append(meal_entry)

            self.meal_queue = updated_meal_queue

            # Validate glucose passed to algorithm (could be modified by stress events)
            # This ensures even sensor error stress events are validated
            glucose_to_algorithm = self._validate_glucose_fail_soft(
                glucose_to_algorithm, float(current_time), "sensor_to_algo"
            )

            # Apply dynamic ratio overrides (ISF/ICR/DIA/Basal) if any
            ratio_state = self._apply_ratio_overrides(float(current_time))
            effective_isf = float(ratio_state.get("isf", self.patient_model.insulin_sensitivity))
            effective_icr = float(ratio_state.get("icr", self.patient_model.carb_factor))
            effective_dia = float(ratio_state.get("dia_minutes", self.patient_model.insulin_action_duration))
            effective_basal = float(ratio_state.get("basal_rate_u_per_hr", self.patient_model.basal_insulin_rate))

            # Glucose trend (mg/dL per minute) based on a short smoothed CGM window.
            glucose_trend = self._update_glucose_trend(float(current_time), glucose_to_algorithm)

            predicted_glucose_heuristic = self._predict_glucose(
                current_glucose=glucose_to_algorithm,
                trend_mgdl_min=glucose_trend,
                iob_units=self.patient_model.insulin_on_board,
                cob_grams=self.patient_model.carbs_on_board,
                isf=effective_isf,
                icr=effective_icr,
                dia_minutes=effective_dia,
                horizon_minutes=self.safety_config.predicted_hypoglycemia_horizon_minutes,
                carb_absorption_minutes=self.patient_model.carb_absorption_duration_minutes,
            )
            predicted_glucose_30 = predicted_glucose_heuristic
            predicted_glucose_ai = None
            predictor_meta: Dict[str, Any] = {
                "predictor_used": False,
                "predictor_gate_reason": "disabled",
                "predictor_uncertainty_std_mgdl": None,
                "predictor_in_distribution": True,
                "predictor_ood_max_zscore": 0.0,
                "predictor_ood_feature_fraction": 0.0,
            }
            if self.predictor is not None:
                feature_row = {
                    "glucose_actual_mgdl": float(glucose_to_algorithm),
                    "patient_iob_units": float(self.patient_model.insulin_on_board),
                    "patient_cob_grams": float(self.patient_model.carbs_on_board),
                    "effective_isf": float(effective_isf),
                    "effective_icr": float(effective_icr),
                    "effective_basal_rate_u_per_hr": float(effective_basal),
                    "glucose_trend_mgdl_min": float(glucose_trend),
                }
                predicted_glucose_ai, _, predictor_meta = self._predict_with_model(
                    feature_row,
                    predicted_glucose_heuristic,
                )
                predicted_glucose_30 = float(predicted_glucose_ai)

            # --- Algorithm Input ---
            algo_input = AlgorithmInput(
                current_glucose=glucose_to_algorithm,
                time_step=self.time_step,
                insulin_on_board=self.patient_model.insulin_on_board,
                carb_intake=algo_carb_intake_this_step, # Use algo's perspective of carbs
                carbs_on_board=self.patient_model.carbs_on_board,
                isf=effective_isf,
                icr=effective_icr,
                dia_minutes=effective_dia,
                basal_rate_u_per_hr=effective_basal,
                glucose_trend_mgdl_min=glucose_trend,
                predicted_glucose_30min=predicted_glucose_30,
                predicted_glucose_30min_std=predictor_meta.get("predictor_uncertainty_std_mgdl"),
                predictor_in_distribution=bool(predictor_meta.get("predictor_in_distribution", True)),
                predictor_gate_reason=str(predictor_meta.get("predictor_gate_reason", "none")),
                patient_state=self.patient_model.get_patient_state(),
                current_time=float(current_time) # Pass current_time
            )

            # --- Algorithm Calculation ---
            if self.enable_profiling:
                algo_start_time = time.perf_counter()
                insulin_output = self.algorithm.predict_insulin(algo_input)
                algorithm_latency_ms = (time.perf_counter() - algo_start_time) * 1000
                self._profiling_samples["algorithm_latency_ms"].append(algorithm_latency_ms)
            else:
                insulin_output = self.algorithm.predict_insulin(algo_input)
            algo_recommended_insulin = insulin_output.get("total_insulin_delivered", 0.0)
            # Validate the algorithm's output to prevent negative insulin requests
            raw_algo_recommended_insulin = float(algo_recommended_insulin)
            algo_recommended_insulin = self.input_validator.validate_insulin(algo_recommended_insulin)
            if raw_algo_recommended_insulin < 0.0 and algo_recommended_insulin == 0.0:
                self._input_validator_negative_insulin_clamp_count += 1
            
            # Get the why_log from the algorithm
            algorithm_why_log = self.algorithm.get_why_log()

            # --- Safety Supervision ---
            start_perf_time = time.perf_counter()
            proposed_basal_units = float(insulin_output.get("basal_insulin", 0.0))
            basal_limit_u_per_hr = effective_basal * self.safety_config.max_basal_multiplier
            basal_limit_units = (basal_limit_u_per_hr / 60.0) * float(self.time_step)
            safety_result = self.supervisor.evaluate_safety(
                current_glucose=glucose_to_algorithm,
                proposed_insulin=algo_recommended_insulin,
                current_time=float(current_time),
                current_iob=self.patient_model.insulin_on_board,
                predicted_glucose_30min=predicted_glucose_30,
                basal_insulin_units=proposed_basal_units,
                basal_limit_units=basal_limit_units,
            )
            supervisor_latency_ms = (time.perf_counter() - start_perf_time) * 1000
            if self.enable_profiling:
                self._profiling_samples["supervisor_latency_ms"].append(supervisor_latency_ms)

            delivered_insulin = safety_result["approved_insulin"]
            overridden = safety_result["insulin_reduction"] > 0
            safety_level = safety_result["safety_level"]
            safety_actions = "; ".join(safety_result["actions_taken"])
            safety_reason = safety_result.get("safety_reason", "")
            safety_triggered = safety_result.get("safety_triggered", False)

            pump_delivery = self.pump_model.deliver(delivered_insulin, self.time_step)
            delivered_insulin = pump_delivery.delivered_units

            if safety_triggered:
                self._emit_safety_event(
                    {
                        "source": "supervisor",
                        "time_minutes": current_time,
                        "glucose_mgdl": glucose_to_algorithm,
                        "ai_requested_units": algo_recommended_insulin,
                        "supervisor_approved_units": safety_result["approved_insulin"],
                        "pump_delivered_units": delivered_insulin,
                        "safety_level": safety_level.value,
                        "safety_reason": safety_reason,
                        "safety_actions": safety_actions,
                        "predicted_glucose_30min": predicted_glucose_30,
                        "predictor_gate_reason": predictor_meta.get("predictor_gate_reason"),
                        "predictor_uncertainty_std_mgdl": predictor_meta.get("predictor_uncertainty_std_mgdl"),
                        "predictor_in_distribution": predictor_meta.get("predictor_in_distribution", True),
                        "iob_units": self.patient_model.insulin_on_board,
                        "cob_grams": self.patient_model.carbs_on_board,
                    }
                )

            # --- Human-in-the-loop callback ---
            human_intervention = None
            if self.on_step:
                context = {
                    "time_minutes": current_time,
                    "glucose_actual_mgdl": actual_glucose_reading,
                    "glucose_to_algo_mgdl": glucose_to_algorithm,
                    "algo_recommended_insulin_units": algo_recommended_insulin,
                    "delivered_insulin_units": delivered_insulin,
                    "patient_iob_units": self.patient_model.insulin_on_board,
                    "patient_cob_grams": self.patient_model.carbs_on_board,
                    "safety_reason": safety_reason,
                    "safety_triggered": safety_triggered,
                    "sensor_status": sensor_reading.status,
                    "pump_status": pump_delivery.status,
                }
                human_intervention = self.on_step(context) or None
                if isinstance(human_intervention, dict):
                    if "additional_carbs" in human_intervention:
                        additional_carbs = float(human_intervention["additional_carbs"])
                        patient_carb_event_this_step += additional_carbs
                        patient_carb_absorbed_this_step += additional_carbs
                    if "override_delivered_insulin" in human_intervention:
                        requested_override = float(human_intervention["override_delivered_insulin"])
                        if requested_override < 0.0:
                            requested_override = 0.0
                        # Remove any same-timestamp dose to avoid double-counting in supervisor history
                        self.supervisor.dose_history = [
                            (t, d) for (t, d) in self.supervisor.dose_history if t != float(current_time)
                        ]
                        override_result = self.supervisor.evaluate_safety(
                            current_glucose=glucose_to_algorithm,
                            proposed_insulin=requested_override,
                            current_time=float(current_time),
                            current_iob=self.patient_model.insulin_on_board,
                            predicted_glucose_30min=predicted_glucose_30,
                            basal_insulin_units=proposed_basal_units,
                            basal_limit_units=basal_limit_units,
                        )
                        delivered_override = override_result["approved_insulin"]
                        pump_delivery = self.pump_model.deliver(delivered_override, self.time_step)
                        delivered_insulin = pump_delivery.delivered_units
                        safety_level = override_result["safety_level"]
                        safety_actions = "; ".join(override_result["actions_taken"])
                        safety_reason = override_result.get("safety_reason", safety_reason)
                        safety_triggered = override_result.get("safety_triggered", safety_triggered)
                        if override_result.get("safety_triggered", False):
                            self._emit_safety_event(
                                {
                                    "source": "human_override",
                                    "time_minutes": current_time,
                                    "glucose_mgdl": glucose_to_algorithm,
                                    "ai_requested_units": algo_recommended_insulin,
                                    "human_requested_units": requested_override,
                                    "supervisor_approved_units": override_result["approved_insulin"],
                                    "pump_delivered_units": delivered_insulin,
                                    "safety_level": safety_level.value,
                                    "safety_reason": safety_reason,
                                    "safety_actions": safety_actions,
                                    "predicted_glucose_30min": predicted_glucose_30,
                                    "predictor_gate_reason": predictor_meta.get("predictor_gate_reason"),
                                    "predictor_uncertainty_std_mgdl": predictor_meta.get("predictor_uncertainty_std_mgdl"),
                                    "predictor_in_distribution": predictor_meta.get("predictor_in_distribution", True),
                                    "iob_units": self.patient_model.insulin_on_board,
                                    "cob_grams": self.patient_model.carbs_on_board,
                                }
                            )
                        self._write_audit_log(
                            {
                                "timestamp": current_time,
                                "event": "human_override_revalidated",
                                "requested_override": requested_override,
                                "approved_override": delivered_override,
                                "final_dose": delivered_insulin,
                                "safety_reason": safety_reason,
                            }
                        )
                    if human_intervention.get("stop_simulation"):
                        logger.warning("Simulation stopped by human-in-the-loop callback.")

            # --- Audit Logging ---
            self._write_audit_log({
                "timestamp": current_time,
                "cgm": actual_glucose_reading,
                "ai_suggestion": algo_recommended_insulin,
                "supervisor_override": overridden,
                "final_dose": delivered_insulin,
                "safety_reason": safety_reason,
                "ai_glucagon_suggestion_mg": float(insulin_output.get("total_glucagon_delivered_mg", 0.0)),
                "hidden_state_summary": self.algorithm.get_state()
            })

            # --- Patient Model Update ---
            algo_recommended_glucagon_mg = float(insulin_output.get("total_glucagon_delivered_mg", 0.0))
            glucagon_safety_result = self._evaluate_glucagon_safety(
                requested_glucagon_mg=algo_recommended_glucagon_mg,
                current_glucose=glucose_to_algorithm,
                predicted_glucose_30min=predicted_glucose_30,
                current_time=float(current_time),
            )
            delivered_glucagon_mg = float(glucagon_safety_result["approved_glucagon_mg"])
            glucagon_safety_triggered = bool(glucagon_safety_result["glucagon_safety_triggered"])
            glucagon_safety_reason = str(glucagon_safety_result["glucagon_safety_reason"])
            glucagon_safety_actions = "; ".join(glucagon_safety_result["glucagon_safety_actions"])

            if glucagon_safety_triggered:
                self._emit_safety_event(
                    {
                        "source": "glucagon_supervisor",
                        "time_minutes": current_time,
                        "glucose_mgdl": glucose_to_algorithm,
                        "predicted_glucose_30min": predicted_glucose_30,
                        "ai_requested_glucagon_mg": algo_recommended_glucagon_mg,
                        "approved_glucagon_mg": delivered_glucagon_mg,
                        "glucagon_safety_reason": glucagon_safety_reason,
                        "glucagon_safety_actions": glucagon_safety_actions,
                        "iob_units": self.patient_model.insulin_on_board,
                        "cob_grams": self.patient_model.carbs_on_board,
                    }
                )
                self._write_audit_log(
                    {
                        "timestamp": current_time,
                        "event": "glucagon_safety_revalidated",
                        "requested_glucagon_mg": algo_recommended_glucagon_mg,
                        "approved_glucagon_mg": delivered_glucagon_mg,
                        "glucagon_safety_reason": glucagon_safety_reason,
                    }
                )

            self.patient_model.update(
                time_step=self.time_step,
                delivered_insulin=delivered_insulin,
                carb_intake=patient_carb_absorbed_this_step,
                delivered_glucagon_mg=delivered_glucagon_mg,
                current_time=float(current_time),
            )

            bolus_insulin_reported = insulin_output.get("bolus_insulin")
            if bolus_insulin_reported is None:
                bolus_insulin_reported = (
                    float(insulin_output.get("meal_bolus", 0.0))
                    + float(insulin_output.get("correction_bolus", 0.0))
                )

            # --- XAI Engine (Explainable Events) ---
            step_events = self._build_explainable_events(
                current_time=float(current_time),
                actual_glucose_reading=float(actual_glucose_reading),
                glucose_trend=float(glucose_trend),
                predicted_glucose_30=float(predicted_glucose_30),
                safety_triggered=bool(safety_triggered),
                safety_result=safety_result,
            )
            self._xai_previous_glucose_trend = float(glucose_trend)
            self.explainable_events_log.extend(step_events)

            # --- Record Data ---
            record = {
                "time_minutes": current_time,
                "glucose_actual_mgdl": actual_glucose_reading,
                "glucose_mechanistic_mgdl": mechanistic_glucose_reading,
                "physiology_residual_mgdl": physiology_residual,
                "glucose_to_algo_mgdl": glucose_to_algorithm,
                "glucose_trend_mgdl_min": glucose_trend,
                "predicted_glucose_30min": predicted_glucose_30,
                "predicted_glucose_heuristic_30min": predicted_glucose_heuristic,
                "predicted_glucose_ai_30min": predicted_glucose_ai,
                "predictor_used": predictor_meta.get("predictor_used", False),
                "predictor_gate_reason": predictor_meta.get("predictor_gate_reason", "none"),
                "predictor_uncertainty_std_mgdl": predictor_meta.get("predictor_uncertainty_std_mgdl"),
                "predictor_in_distribution": predictor_meta.get("predictor_in_distribution", True),
                "predictor_ood_max_zscore": predictor_meta.get("predictor_ood_max_zscore", 0.0),
                "predictor_ood_feature_fraction": predictor_meta.get("predictor_ood_feature_fraction", 0.0),
                "delivered_insulin_units": delivered_insulin,
                "algo_recommended_insulin_units": algo_recommended_insulin,
                "algo_recommended_glucagon_mg": algo_recommended_glucagon_mg,
                "delivered_glucagon_mg": delivered_glucagon_mg,
                "glucagon_safety_triggered": glucagon_safety_triggered,
                "glucagon_safety_reason": glucagon_safety_reason,
                "glucagon_safety_actions": glucagon_safety_actions,
                "sensor_status": sensor_reading.status,
                "input_validator_fail_soft": self._input_validator_last_step_fail_soft,
                "pump_status": pump_delivery.status,
                "pump_reason": pump_delivery.reason,
                "basal_insulin_units": insulin_output.get("basal_insulin", 0.0),
                "bolus_insulin_units": bolus_insulin_reported,
                "meal_bolus_units": insulin_output.get("meal_bolus", 0.0),
                "correction_bolus_units": insulin_output.get("correction_bolus", 0.0),
                "carb_intake_grams": patient_carb_event_this_step,
                "carb_absorbed_grams": patient_carb_absorbed_this_step,
                "patient_iob_units": self.patient_model.insulin_on_board,
                "patient_cob_grams": self.patient_model.carbs_on_board,
                "effective_isf": effective_isf,
                "effective_icr": effective_icr,
                "effective_basal_rate_u_per_hr": effective_basal,
                "effective_dia_minutes": effective_dia,
                "uncertainty": insulin_output.get("uncertainty", 0.0),
                "fallback_triggered": insulin_output.get("fallback_triggered", False),
                "safety_level": safety_level.value,
                "safety_actions": safety_actions,
                "safety_reason": safety_reason,
                "safety_triggered": safety_triggered,
                "supervisor_latency_ms": supervisor_latency_ms,
                "human_intervention": bool(human_intervention),
                "human_intervention_note": human_intervention.get("note") if isinstance(human_intervention, dict) else "",
                "algorithm_why_log": [entry.to_dict() for entry in algorithm_why_log], # Convert WhyLogEntry to dict for serialization
                "explainable_events": "; ".join(step_events) if step_events else "",
                **{f"algo_state_{k}": v for k, v in self.algorithm.get_state().items()} # Include algorithm internal state
            }

            if self.enable_profiling:
                step_latency_ms = (time.perf_counter() - step_start_time) * 1000
                self._profiling_samples["step_latency_ms"].append(step_latency_ms)
                record["algorithm_latency_ms"] = algorithm_latency_ms
                record["step_latency_ms"] = step_latency_ms
            
            yield record

            if isinstance(human_intervention, dict) and human_intervention.get("stop_simulation"):
                break

            # Critical failure stop: sustained severe hypoglycemia
            if actual_glucose_reading < self.critical_glucose_threshold:
                self._critical_low_minutes += self.time_step
                if self._critical_low_minutes >= self.critical_glucose_duration_minutes:
                    message = (
                        f"Critical failure: glucose < {self.critical_glucose_threshold:.1f} mg/dL "
                        f"for {self._critical_low_minutes} minutes."
                    )
                    raise SimulationLimitError(
                        message=message,
                        current_time=current_time,
                        glucose_value=actual_glucose_reading,
                        duration_minutes=self._critical_low_minutes,
                    )
            else:
                self._critical_low_minutes = 0

            current_time += self.time_step

    def save_state(self) -> Dict[str, Any]:
        """Serialize simulator state for time-travel debugging."""
        return {
            "current_time": self._current_time,
            "patient_state": self.patient_model.get_state(),
            "algorithm_state": self.algorithm.get_state(),
            "supervisor_state": self.supervisor.get_state(),
            "input_validator_state": self.input_validator.get_state(),
            "sensor_state": self.sensor_model.get_state(),
            "pump_state": self.pump_model.get_state(),
            "physiology_variation_state": (
                self.physiology_variation_model.get_state()
                if self.physiology_variation_model is not None
                else None
            ),
            "meal_queue": self.meal_queue,
            "stress_events": [event.__dict__ for event in self.stress_events],
            "critical_low_minutes": self._critical_low_minutes,
            "glucose_trend_history": self._glucose_trend_history,
            "xai_previous_glucose_trend": self._xai_previous_glucose_trend,
            "xai_last_event_time": self._xai_last_event_time,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore simulator state from a previous save."""
        self._current_time = state.get("current_time", 0)
        self.patient_model.set_state(state.get("patient_state", {}))
        self.algorithm.set_state(state.get("algorithm_state", {}))
        self.supervisor.set_state(state.get("supervisor_state", {}))
        self.input_validator.set_state(state.get("input_validator_state", {}))
        self.sensor_model.set_state(state.get("sensor_state", {}))
        self.pump_model.set_state(state.get("pump_state", {}))
        physiology_variation_state = state.get("physiology_variation_state")
        if self.physiology_variation_model is not None and isinstance(physiology_variation_state, dict):
            self.physiology_variation_model.set_state(physiology_variation_state)
        self.meal_queue = state.get("meal_queue", [])
        self.stress_events = [StressEvent(**payload) for payload in state.get("stress_events", [])]
        self._critical_low_minutes = state.get("critical_low_minutes", 0)
        self._glucose_trend_history = [
            (float(item[0]), float(item[1]))
            for item in state.get("glucose_trend_history", [])
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]
        previous_trend = state.get("xai_previous_glucose_trend")
        self._xai_previous_glucose_trend = None if previous_trend is None else float(previous_trend)
        xai_last_event_time = state.get("xai_last_event_time", {})
        self._xai_last_event_time = (
            {str(key): float(value) for key, value in xai_last_event_time.items()}
            if isinstance(xai_last_event_time, dict)
            else {}
        )
        self._resume_state = True

    def _build_performance_report(self) -> Dict[str, Any]:
        def summarize(samples: List[float]) -> Dict[str, float]:
            if not samples:
                return {}
            values = np.array(samples, dtype=float)
            return {
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "min_ms": float(np.min(values)),
                "max_ms": float(np.max(values)),
                "std_ms": float(np.std(values)),
            }

        return {
            "algorithm_latency_ms": summarize(self._profiling_samples["algorithm_latency_ms"]),
            "supervisor_latency_ms": summarize(self._profiling_samples["supervisor_latency_ms"]),
            "step_latency_ms": summarize(self._profiling_samples["step_latency_ms"]),
            "sample_counts": {
                "algorithm_latency_ms": len(self._profiling_samples["algorithm_latency_ms"]),
                "supervisor_latency_ms": len(self._profiling_samples["supervisor_latency_ms"]),
                "step_latency_ms": len(self._profiling_samples["step_latency_ms"]),
            },
        }
