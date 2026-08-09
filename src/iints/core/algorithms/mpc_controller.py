#!/usr/bin/env python3
"""
Model Predictive Controller (MPC) — IINTS-AF
=============================================
Uses an internal ODE model (Bergman Minimal Model) to predict glucose
trajectory and scipy.optimize to calculate the mathematically optimal
insulin dose that minimizes glucose deviation from target.
"""
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize

from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput
from iints.core.patient.bergman_model import BergmanPatientModel
from iints.core.safety.config import (
    CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN,
    CONTROLLER_HIGH_IOB_GUARD_UNITS,
    CONTROLLER_HYPO_GUARD_MGDL,
)


class MPCController(InsulinAlgorithm):
    """
    Research nonlinear MPC prototype using an adapted Bergman model.

    The optimizer proposes a candidate dose. It does not bypass the independent
    supervisor and is not a clinically validated dosing controller.
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(settings)
        self.target_glucose = float(self.settings.get("target_glucose", 120.0))
        self.prediction_horizon_mins = float(self.settings.get("prediction_horizon_mins", 60.0))
        self.step_size_mins = float(self.settings.get("step_size_mins", 5.0))

        self.max_insulin_u_per_step = float(self.settings.get("max_insulin_u_per_step", 1.2))
        self.min_insulin_u_per_step = 0.0
        self.hypo_guard_glucose = float(
            self.settings.get("hypo_guard_glucose", CONTROLLER_HYPO_GUARD_MGDL)
        )
        self.falling_trend_guard = float(
            self.settings.get(
                "falling_trend_guard_mgdl_min",
                CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN,
            )
        )
        self.high_iob_guard_units = float(
            self.settings.get("high_iob_guard_units", CONTROLLER_HIGH_IOB_GUARD_UNITS)
        )
        self.iob_taper_start_units = float(self.settings.get("iob_taper_start_units", 2.0))

        # We keep an internal Bergman model running in parallel to the real patient
        # to maintain the mathematical state (X, I, G).
        self.internal_model = BergmanPatientModel(
            initial_glucose=120.0,
            max_glucose_rate_mgdl_per_min=0.0,
        )
        self.is_initialized = False

    def _sync_internal_model(self, data: AlgorithmInput) -> None:
        """Assimilate observable state without assuming a proposed dose occurred."""
        if not self.is_initialized:
            if data.basal_rate_u_per_hr is not None:
                self.internal_model.basal_insulin_rate = max(
                    0.0, float(data.basal_rate_u_per_hr)
                )
            self.internal_model.reset()
            ode_state = data.patient_state.get("ode_state")
            if (
                isinstance(ode_state, list)
                and len(ode_state) == 14
                and data.patient_state.get("state_schema", "").startswith("bergman")
            ):
                self.internal_model.set_state(data.patient_state)
            self.is_initialized = True

        # Directly observed/derived summaries are assimilated at every tick.
        # This is deterministic state correction, not a Kalman filter.
        self.internal_model.current_glucose = float(data.current_glucose)
        self.internal_model._state[0] = float(data.current_glucose)
        patient_state = data.patient_state
        if "remote_insulin_action" in patient_state:
            self.internal_model._state[1] = max(
                0.0, float(patient_state["remote_insulin_action"])
            )
        if "plasma_insulin_mU_L" in patient_state:
            self.internal_model._state[2] = max(
                0.0, float(patient_state["plasma_insulin_mU_L"])
            )
        for state_key, state_index in (
            ("stomach_solid_mg", 3),
            ("stomach_liquid_mg", 4),
            ("gut_glucose_mg", 5),
            ("subcut_insulin_1_mU", 6),
            ("subcut_insulin_2_mU", 7),
        ):
            if state_key in patient_state:
                self.internal_model._state[state_index] = max(
                    0.0, float(patient_state[state_key])
                )

        self.internal_model.insulin_on_board = max(0.0, float(data.insulin_on_board))
        if data.isf is not None:
            self.internal_model.set_ratio_state(isf=float(data.isf))

    def _cost_function(
        self,
        proposed_doses: np.ndarray,
        current_state: Dict[str, Any],
        current_carb_intake: float,
        basal_units_per_step: float,
    ) -> float:
        """
        Calculates the cost of a specific future sequence of insulin doses.
        """
        # Clone the model state
        sim_model = BergmanPatientModel(
            initial_glucose=120.0,
            max_glucose_rate_mgdl_per_min=0.0,
        )
        sim_model.set_state(current_state)

        cost = 0.0
        penalty_hypo = 50.0  # Heavy penalty for going low
        penalty_hyper = 1.0  # Lighter penalty for going high

        for step_index, dose in enumerate(proposed_doses):
            # Predict step
            step_carbs = current_carb_intake if step_index == 0 else 0.0
            g = sim_model.update(
                time_step=self.step_size_mins,
                delivered_insulin=float(dose) + basal_units_per_step,
                carb_intake=step_carbs,
            )

            # Evaluate cost
            error = g - self.target_glucose
            if error < 0:
                # Hypoglycemia risk
                cost += penalty_hypo * (error ** 2)
                if g < 70.0:
                    cost += 5000.0 * ((70.0 - g) ** 2)
            else:
                # Hyperglycemia risk
                cost += penalty_hyper * (error ** 2)

            if g < 100.0 and dose > 0.0:
                cost += 250.0 * float(dose) ** 2

            # Regularization (prefer less insulin if glucose is stable)
            cost += 0.5 * (dose ** 2)

        return cost

    def _safe_zero_decision(self, reason: str, data: AlgorithmInput) -> Dict[str, Any]:
        self._log_reason(reason, "safety_constraint", 0.0)
        return {
            "total_insulin_delivered": 0.0,
            "bolus_insulin": 0.0,
            "basal_insulin": 0.0,
            "mpc_recommended_units": 0.0,
            "mpc_physics_state": self.internal_model.get_patient_state(),
            "research_only": True,
            "safety_reason": reason,
        }

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        self.why_log = []

        # 1. Sync internal physics model
        self._sync_internal_model(data)

        trend = float(data.glucose_trend_mgdl_min or 0.0)
        predicted_30 = float(data.predicted_glucose_30min or data.current_glucose)
        if (
            data.current_glucose <= self.hypo_guard_glucose
            or predicted_30 <= self.hypo_guard_glucose
            or trend <= self.falling_trend_guard
        ):
            return self._safe_zero_decision(
                "MPC held insulin because glucose is low, predicted low, or falling quickly",
                data,
            )

        horizon_steps = int(self.prediction_horizon_mins / self.step_size_mins)
        current_state = self.internal_model.get_state()

        max_step_dose = self.max_insulin_u_per_step
        if data.insulin_on_board > self.iob_taper_start_units:
            taper_width = max(self.high_iob_guard_units - self.iob_taper_start_units, 0.1)
            taper = 1.0 - ((float(data.insulin_on_board) - self.iob_taper_start_units) / taper_width)
            max_step_dose *= max(0.0, min(1.0, taper))
        if max_step_dose <= 0.05:
            return self._safe_zero_decision(
                "MPC held insulin because active insulin/on-board insulin is already high",
                data,
            )

        basal_units = max(0.0, float(data.basal_rate_u_per_hr or 0.0)) * (
            self.step_size_mins / 60.0
        )
        max_correction_dose = max(0.0, max_step_dose - basal_units)
        if max_correction_dose <= 0.0:
            return self._safe_zero_decision(
                "MPC held insulin because configured basal exceeds the current dose envelope",
                data,
            )

        # 2. Setup Optimizer. Decision variables are correction doses; the
        # configured basal is included separately in every horizon step.
        initial_guess = np.zeros(horizon_steps)
        if data.current_glucose > self.target_glucose:
            isf = float(data.isf or 55.0)
            initial_guess[0] = min(
                max_correction_dose,
                max(0.0, (data.current_glucose - self.target_glucose) / isf * 0.25),
            )
        bounds = [
            (self.min_insulin_u_per_step, max_correction_dose)
            for _ in range(horizon_steps)
        ]

        # 3. Solve MPC optimization problem
        self._log_reason("Running Scipy ODE prediction", "physics_engine", f"Horizon: {horizon_steps} steps")
        result = minimize(
            self._cost_function,
            initial_guess,
            args=(current_state, max(0.0, float(data.carb_intake)), basal_units),
            bounds=bounds,
            method="SLSQP",
            options={"maxiter": 20, "disp": False}
        )

        if result.success:
            optimal_doses = result.x
            correction_dose = float(optimal_doses[0])
            optimal_current_dose = basal_units + correction_dose
            self._log_reason(
                "MPC optimization successful",
                "mpc_solver",
                optimal_current_dose,
            )
        else:
            # Fallback to 0 if optimization fails
            optimal_current_dose = 0.0
            self._log_reason("MPC Optimization failed, defaulting to 0", "mpc_error", 0.0)

        return {
            'total_insulin_delivered': optimal_current_dose,
            'bolus_insulin': max(0.0, optimal_current_dose - basal_units),
            'basal_insulin': min(optimal_current_dose, basal_units),
            'mpc_recommended_units': optimal_current_dose,
            # This is the assimilated pre-decision state. It is deliberately
            # not advanced with a dose that the supervisor/pump may reject.
            'mpc_physics_state': self.internal_model.get_patient_state(),
            'research_only': True,
        }

    def reset(self) -> None:
        super().reset()
        self.is_initialized = False
        self.internal_model.reset()

    def get_algorithm_info(self) -> Dict[str, Any]:
        return {
            'name': 'Adapted Bergman MPC (research prototype)',
            'type': 'Model Predictive Control',
            'description': (
                'Optimizes candidate insulin against an adapted Bergman ODE; '
                'requires independent safety supervision and research validation.'
            ),
        }
