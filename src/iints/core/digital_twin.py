from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional
import numpy as np
from scipy.optimize import minimize

from iints.core.patient.hovorka_model import HovorkaPatientModel, HovorkaParameters

logger = logging.getLogger(__name__)

class DigitalTwinCalibrator:
    """
    Research parameter-calibration engine for a Hovorka simulation profile.

    It estimates a deliberately small parameter subset with bounded numerical
    optimization. The output is not a unique physiological identification and
    must not be described as a clinically validated patient digital twin.
    """

    def __init__(
        self,
        base_weight_kg: float = 70.0,
        *,
        basal_insulin_rate_u_per_hour: float = 0.8,
        insulin_semantics: Literal["bolus_only", "total_interval"] = "bolus_only",
        observation_semantics: Literal["cgm_observation", "latent_glucose"] = "cgm_observation",
    ):
        if not np.isfinite(base_weight_kg) or base_weight_kg <= 0.0:
            raise ValueError("base_weight_kg must be finite and positive")
        if not np.isfinite(basal_insulin_rate_u_per_hour) or basal_insulin_rate_u_per_hour < 0.0:
            raise ValueError("basal_insulin_rate_u_per_hour must be finite and non-negative")
        if insulin_semantics not in {"bolus_only", "total_interval"}:
            raise ValueError("insulin_semantics must be 'bolus_only' or 'total_interval'")
        if observation_semantics not in {"cgm_observation", "latent_glucose"}:
            raise ValueError(
                "observation_semantics must be 'cgm_observation' or 'latent_glucose'"
            )
        self.base_weight_kg = base_weight_kg
        self.basal_insulin_rate_u_per_hour = float(basal_insulin_rate_u_per_hour)
        self.insulin_semantics = insulin_semantics
        self.observation_semantics = observation_semantics
        self.calibrated_params: Optional[HovorkaParameters] = None
        self.last_report: Dict[str, Any] = {}

    def fit(self, history: List[Dict[str, Any]]) -> HovorkaParameters:
        """
        Fit the Hovorka model to historical data.
        history format: [{"time_min": 0, "cgm": 110, "carbs": 0, "insulin": 0.0}, ...]
        Assumes uniform time steps for simplicity of this implementation.
        """
        if len(history) < 24:
            raise ValueError("At least 24 historical samples are required to calibrate.")

        times = np.asarray([row["time_min"] for row in history], dtype=float)
        true_cgm = np.asarray([row["cgm"] for row in history], dtype=float)
        carbs = np.asarray([row.get("carbs", 0.0) for row in history], dtype=float)
        insulin = np.asarray([row.get("insulin", 0.0) for row in history], dtype=float)

        if not np.all(np.isfinite(times)):
            raise ValueError("time_min must contain only finite values")
        if np.any(np.isinf(true_cgm)):
            raise ValueError("cgm may contain NaN for missingness, but not infinity")
        if not np.all(np.isfinite(carbs)) or np.any(carbs < 0.0):
            raise ValueError("carbs must contain finite non-negative interval events")
        if not np.all(np.isfinite(insulin)) or np.any(insulin < 0.0):
            raise ValueError("insulin must contain finite non-negative interval doses")

        deltas = np.diff(times.astype(float))
        if np.any(deltas <= 0.0):
            raise ValueError("time_min must be strictly increasing")
        dt = float(np.median(deltas))
        if not np.allclose(deltas, dt, rtol=0.0, atol=1e-6):
            raise ValueError("Calibration currently requires uniformly sampled data")

        # We only consider valid true_cgm for MSE calculation (ignore NaNs)
        valid_mask = ~np.isnan(true_cgm)
        if not np.any(valid_mask):
            raise ValueError("No valid CGM data found in history.")

        # CGM plus meal/bolus events cannot uniquely identify the full Hovorka
        # parameter vector. Fit only three aggregate profile scales and leave
        # insulin distribution/clearance fixed at their nominal values.
        parameter_names = ("egp_scale", "insulin_sensitivity_scale", "meal_tmax_scale")
        x0 = np.ones(len(parameter_names), dtype=float)

        # Base nominal values from Hovorka
        nominal = HovorkaParameters(body_weight_kg=self.base_weight_kg)

        # Conservative research bounds prevent a numerically good fit from
        # silently becoming a physiologically implausible parameter profile.
        bounds = [
            (0.6, 1.6),  # EGP scaling
            (0.35, 2.5),  # global insulin-sensitivity scaling
            (0.5, 2.0),  # published meal-compartment time constant scaling
        ]

        calibration_end = max(16, int(len(times) * 0.8))
        calibration_mask = valid_mask & (np.arange(len(times)) < calibration_end)
        validation_mask = valid_mask & (np.arange(len(times)) >= calibration_end)
        if int(np.sum(calibration_mask)) < 12:
            raise ValueError("At least 12 valid calibration samples are required")
        if int(np.sum(validation_mask)) < 4:
            raise ValueError("At least 4 held-out validation samples are required")

        def params_from_scale(x: np.ndarray) -> HovorkaParameters:
            return HovorkaParameters(
                body_weight_kg=self.base_weight_kg,
                EGP_0_per_kg=x[0] * nominal.EGP_0_per_kg,
                S_IT=x[1] * nominal.S_IT,
                S_ID=x[1] * nominal.S_ID,
                S_IE=x[1] * nominal.S_IE,
                t_max_G=x[2] * nominal.t_max_G,
            )

        def simulate(x: np.ndarray) -> np.ndarray:
            params = params_from_scale(x)
            first_valid_idx = int(np.argmax(valid_mask))
            init_bg = float(true_cgm[first_valid_idx])
            model = HovorkaPatientModel(
                initial_glucose=init_bg,
                basal_insulin_rate=self.basal_insulin_rate_u_per_hour,
                hovorka_params=params,
            )
            sim_cgm = np.full_like(true_cgm, init_bg, dtype=float)
            sim_cgm[first_valid_idx] = model.get_current_glucose()
            for i in range(first_valid_idx + 1, len(times)):
                event_index = i - 1
                delivered = float(insulin[event_index])
                if self.insulin_semantics == "bolus_only":
                    delivered += self.basal_insulin_rate_u_per_hour * dt / 60.0
                model.update(
                    time_step=dt,
                    delivered_insulin=delivered,
                    carb_intake=float(carbs[event_index]),
                    current_time=float(times[event_index]),
                )
                sim_cgm[i] = model.get_current_glucose()
            return sim_cgm

        def objective_function(x):
            sim_cgm = simulate(np.asarray(x, dtype=float))
            diff = sim_cgm[calibration_mask] - true_cgm[calibration_mask]
            mse = np.mean(diff**2)

            # Weak prior regularization reduces compensation between poorly
            # identifiable parameters when only CGM/event data are available.
            reg = 15.0 * np.sum(np.log(np.maximum(x, 1e-6)) ** 2)

            return mse + reg

        logger.info("Starting bounded research-profile calibration via Powell...")
        res = minimize(
            objective_function,
            x0,
            method='Powell',
            bounds=bounds,
            options={'disp': False, 'xtol': 1e-3, 'ftol': 1e-3, 'maxiter': 120},
        )

        if not res.success:
            logger.warning(f"Optimization may have failed or hit bounds: {res.message}")
        else:
            logger.info("Calibration successful!")

        # Create the final calibrated params
        opt_x = res.x
        final_params = params_from_scale(opt_x)

        final_trace = simulate(opt_x)
        nominal_trace = simulate(x0)
        calibration_rmse = float(
            np.sqrt(np.mean((final_trace[calibration_mask] - true_cgm[calibration_mask]) ** 2))
        )
        validation_rmse = float(
            np.sqrt(np.mean((final_trace[validation_mask] - true_cgm[validation_mask]) ** 2))
        )
        nominal_calibration_rmse = float(
            np.sqrt(np.mean((nominal_trace[calibration_mask] - true_cgm[calibration_mask]) ** 2))
        )
        nominal_validation_rmse = float(
            np.sqrt(np.mean((nominal_trace[validation_mask] - true_cgm[validation_mask]) ** 2))
        )

        # Local finite-difference sensitivity matrix. A high condition number
        # means different parameter combinations produce nearly identical CGM
        # traces and the fitted values must not be interpreted independently.
        sensitivity_columns: list[np.ndarray] = []
        for index in range(len(opt_x)):
            delta = max(0.01, 0.02 * abs(float(opt_x[index])))
            plus = np.array(opt_x, dtype=float)
            minus = np.array(opt_x, dtype=float)
            plus[index] = min(bounds[index][1], plus[index] + delta)
            minus[index] = max(bounds[index][0], minus[index] - delta)
            denominator = plus[index] - minus[index]
            if denominator <= 0.0:
                sensitivity_columns.append(np.zeros(int(np.sum(calibration_mask))))
                continue
            derivative = (simulate(plus) - simulate(minus)) / denominator
            sensitivity_columns.append(derivative[calibration_mask])
        sensitivity_matrix = np.column_stack(sensitivity_columns)
        singular_values = np.linalg.svd(sensitivity_matrix, compute_uv=False)
        condition_number = float(
            np.inf
            if singular_values.size == 0 or singular_values[-1] <= 1e-10
            else singular_values[0] / singular_values[-1]
        )
        boundary_parameters = [
            name
            for name, value, (low, high) in zip(parameter_names, opt_x, bounds)
            if min(abs(float(value) - low), abs(high - float(value))) <= 0.02 * (high - low)
        ]
        warnings: list[str] = []
        if self.observation_semantics == "cgm_observation":
            warnings.append(
                "Calibration compares latent model glucose directly with CGM observations; "
                "sensor lag, noise, and bias can be absorbed into fitted physiology. Treat "
                "the result as a profile fit, not physiological identification."
            )
        if condition_number > 1.0e4:
            warnings.append(
                "Local sensitivity matrix is ill-conditioned; fitted parameters are not separately identifiable."
            )
        if boundary_parameters:
            warnings.append("Optimizer reached or approached bounds: " + ", ".join(boundary_parameters))
        if validation_rmse >= nominal_validation_rmse:
            warnings.append(
                "Held-out RMSE did not improve over the nominal profile; do not promote this calibration."
            )

        self.calibrated_params = final_params
        self.last_report = {
            "optimizer_success": bool(res.success),
            "optimizer_message": str(res.message),
            "calibration_samples": int(np.sum(calibration_mask)),
            "validation_samples": int(np.sum(validation_mask)),
            "calibration_rmse_mgdl": calibration_rmse,
            "validation_rmse_mgdl": validation_rmse,
            "nominal_calibration_rmse_mgdl": nominal_calibration_rmse,
            "nominal_validation_rmse_mgdl": nominal_validation_rmse,
            "validation_improvement_mgdl": nominal_validation_rmse - validation_rmse,
            "parameter_scales": {
                name: float(value) for name, value in zip(parameter_names, opt_x)
            },
            "local_sensitivity_condition_number": condition_number,
            "parameters_near_bounds": boundary_parameters,
            "insulin_semantics": self.insulin_semantics,
            "observation_semantics": self.observation_semantics,
            "event_alignment": "events at row i drive the interval [time_i, time_i+1)",
            "warnings": warnings,
            "interpretation": (
                "bounded aggregate research-profile calibration; held-out data are an audit split, "
                "not an external validation cohort; parameters are not unique physiological estimates"
            ),
        }
        return final_params

    def export_profile(self, filepath: str) -> None:
        """Export the calibrated parameters to JSON for Digital Twin what-if scenarios."""
        if not self.calibrated_params:
            raise ValueError("Model not calibrated yet. Call fit() first.")

        # Convert dataclass to dict via __dict__
        data = {k: v for k, v in self.calibrated_params.__dict__.items()}
        data["calibration_report"] = self.last_report

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Digital Twin profile saved to {filepath}")
