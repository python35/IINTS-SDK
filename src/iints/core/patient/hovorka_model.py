"""
Improved Hovorka Model - IINTS-AF
==================================
Based on standard Hovorka artificial pancreas equations and extended
to match the IINTS simulator's interface.

State vector (10 variables):
0: Q1 (mg) - Accessible glucose
1: Q2 (mg) - Non-accessible glucose
2: S1 (mU) - SubQ insulin pool 1
3: S2 (mU) - SubQ insulin pool 2
4: I (mU/L) - Plasma insulin
5: x1 (1/min) - Insulin action on distribution
6: x2 (1/min) - Insulin action on disposal
7: x3 (1) - Insulin action on EGP
8: D1 (mg) - Stomach carbs
9: D2 (mg) - Gut carbs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class HovorkaParameters:
    """Physiological parameters for the Hovorka Model."""

    body_weight_kg: float = 70.0

    # Insulin absorption
    t_max_I: float = 55.0  # min

    # Carb absorption
    t_max_G: float = 40.0  # min
    A_G: float = 0.8  # bioavailability

    # Insulin kinetics
    k_e: float = 0.138  # elimination rate, 1/min
    V_I_per_kg: float = 0.12  # L/kg

    # Glucose kinetics
    V_G_per_kg: float = 0.16  # L/kg
    k_12: float = 0.066  # 1/min
    EGP_0_per_kg: float = 16.1 * 0.18  # umol/kg/min -> mg/kg/min (1 umol = 0.18 mg)
    F_01c_per_kg: float = 9.7 * 0.18   # umol/kg/min -> mg/kg/min

    # Insulin action (activation/deactivation rates)
    k_a1: float = 0.006  # 1/min
    k_a2: float = 0.06   # 1/min
    k_a3: float = 0.03   # 1/min

    # Insulin sensitivities
    S_IT: float = 51.2e-4  # L/mU/min (effect on distribution)
    S_ID: float = 8.2e-4   # L/mU/min (effect on disposal)
    S_IE: float = 520e-4   # L/mU (effect on EGP)


class HovorkaPatientModel:
    def __init__(
        self,
        basal_insulin_rate: float = 0.8,
        insulin_sensitivity: float = 50.0,
        carb_factor: float = 10.0,
        initial_glucose: float = 120.0,
        glucose_decay_rate: float = 0.05,
        glucose_absorption_rate: float = 0.03,
        insulin_action_duration: float = 300.0,
        insulin_peak_time: float = 75.0,
        meal_mismatch_epsilon: float = 1.0,
        dawn_phenomenon_strength: float = 0.0,
        dawn_start_hour: float = 4.0,
        dawn_end_hour: float = 8.0,
        carb_absorption_duration_minutes: float = 240.0,
        max_glucose_rate_mgdl_per_min: float = 3.0,
        hovorka_params: Optional[HovorkaParameters] = None,
    ) -> None:
        self.basal_insulin_rate = basal_insulin_rate
        self.insulin_sensitivity = insulin_sensitivity
        self.carb_factor = carb_factor
        self.initial_glucose = initial_glucose
        self.glucose_decay_rate = glucose_decay_rate
        self.glucose_absorption_rate = glucose_absorption_rate
        self.insulin_action_duration = insulin_action_duration
        self.insulin_peak_time = insulin_peak_time
        self.meal_mismatch_epsilon = meal_mismatch_epsilon
        self.dawn_phenomenon_strength = dawn_phenomenon_strength
        self.dawn_start_hour = dawn_start_hour
        self.dawn_end_hour = dawn_end_hour
        self.carb_absorption_duration_minutes = carb_absorption_duration_minutes
        self.max_glucose_rate_mgdl_per_min = max_glucose_rate_mgdl_per_min

        self.params = hovorka_params if hovorka_params else HovorkaParameters()

        # Stress and Exercise book-keeping
        self.is_exercising = False
        self.exercise_intensity = 0.0
        self.is_stressed = False
        self.stress_intensity = 0.0

        # Trackers
        self.active_insulin_doses: List[Dict[str, float]] = []
        self.active_carb_intakes: List[Dict[str, float]] = []

        self.current_glucose = initial_glucose
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0

        self.reset()

    def _glucose_volume_dl(self) -> float:
        p = self.params
        return p.V_G_per_kg * p.body_weight_kg * 10.0

    def _default_ode_state(self, glucose_mgdl: Optional[float] = None) -> np.ndarray:
        p = self.params
        V_G_dL = self._glucose_volume_dl()

        glucose = float(self.initial_glucose if glucose_mgdl is None else glucose_mgdl)
        Q1_init = glucose * V_G_dL
        Q2_init = Q1_init * 0.5  # Rough steady-state approximation.

        I_basal = 10.0  # mU/L approximation.
        x1_init = p.S_IT * I_basal
        x2_init = p.S_ID * I_basal
        x3_init = p.S_IE * I_basal

        # State vector: [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2]
        return np.array(
            [Q1_init, Q2_init, 0.0, 0.0, I_basal, x1_init, x2_init, x3_init, 0.0, 0.0],
            dtype=np.float64,
        )

    def _coerce_legacy_ode_state(self, ode_state: np.ndarray) -> np.ndarray:
        """Load older Bergman/custom snapshots into a safe Hovorka state."""
        if ode_state.size == 10:
            return ode_state.astype(np.float64, copy=True)

        glucose = float(self.current_glucose)
        if ode_state.size >= 1 and np.isfinite(ode_state[0]):
            # Older models store glucose in mg/dL as the first element. If a
            # caller passes a Hovorka-like mass, this still keeps a plausible
            # bounded glucose instead of crashing a resume flow.
            candidate = float(ode_state[0])
            glucose = candidate if candidate < 1000.0 else candidate / self._glucose_volume_dl()

        coerced = self._default_ode_state(glucose_mgdl=glucose)
        if ode_state.size >= 3 and np.isfinite(ode_state[2]):
            coerced[4] = max(0.0, float(ode_state[2]))
        if ode_state.size >= 4 and np.isfinite(ode_state[3]):
            coerced[8] = max(0.0, float(ode_state[3]))
        if ode_state.size >= 5 and np.isfinite(ode_state[4]):
            coerced[9] = max(0.0, float(ode_state[4]))
        if ode_state.size >= 7:
            if np.isfinite(ode_state[5]):
                coerced[2] = max(0.0, float(ode_state[5]))
            if np.isfinite(ode_state[6]):
                coerced[3] = max(0.0, float(ode_state[6]))
        return coerced

    def _guard_glucose_transition(self, proposed_glucose: float, time_step: float) -> float:
        if not np.isfinite(proposed_glucose):
            return float(self.current_glucose)
        max_rate = float(self.max_glucose_rate_mgdl_per_min or 0.0)
        if max_rate <= 0.0:
            return float(max(20.0, proposed_glucose))
        max_delta = max_rate * max(float(time_step), 0.0)
        requested_delta = float(proposed_glucose) - float(self.current_glucose)
        bounded_delta = float(np.clip(requested_delta, -max_delta, max_delta))
        return float(max(20.0, self.current_glucose + bounded_delta))

    def reset(self) -> None:
        self._state = self._default_ode_state()
        self.current_glucose = self.initial_glucose
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0
        self.active_insulin_doses = []
        self.active_carb_intakes = []
        self.is_exercising = False
        self.exercise_intensity = 0.0
        self.is_stressed = False
        self.stress_intensity = 0.0

    def start_exercise(self, intensity: float) -> None:
        if not (0.0 <= intensity <= 1.0):
            raise ValueError("Exercise intensity must be between 0.0 and 1.0")
        self.is_exercising = True
        self.exercise_intensity = intensity

    def stop_exercise(self) -> None:
        self.is_exercising = False
        self.exercise_intensity = 0.0

    def start_stress(self, intensity: float) -> None:
        if not (0.0 <= intensity <= 1.0):
            raise ValueError("Stress intensity must be between 0.0 and 1.0")
        self.is_stressed = True
        self.stress_intensity = intensity

    def stop_stress(self) -> None:
        self.is_stressed = False
        self.stress_intensity = 0.0

    def update(
        self,
        time_step: float,
        delivered_insulin: float,
        carb_intake: float = 0.0,
        current_time: Optional[float] = None,
        **kwargs,
    ) -> float:
        true_carbs = carb_intake * self.meal_mismatch_epsilon

        # Track IOB
        if delivered_insulin > 0.001:
            self.active_insulin_doses.append({"amount": delivered_insulin, "age": 0.0})
        for d in self.active_insulin_doses:
            d["age"] += time_step
        self.active_insulin_doses = [
            d for d in self.active_insulin_doses
            if d["age"] <= self.insulin_action_duration
        ]
        self.insulin_on_board = sum(
            d["amount"] * max(0.0, (self.insulin_action_duration - d["age"]) / self.insulin_action_duration)
            for d in self.active_insulin_doses
        )

        # Track COB
        if true_carbs > 0:
            self.active_carb_intakes.append({"amount": true_carbs, "time_since_intake": 0.0})
        for c in self.active_carb_intakes:
            c["time_since_intake"] += time_step
        self.active_carb_intakes = [
            c for c in self.active_carb_intakes
            if c["time_since_intake"] <= self.carb_absorption_duration_minutes
        ]
        self.carbs_on_board = sum(
            c["amount"] * max(0.0, 1.0 - c["time_since_intake"] / self.carb_absorption_duration_minutes)
            for c in self.active_carb_intakes
        )

        # Meals into stomach (D1)
        if true_carbs > 0:
            self._state[8] += true_carbs * 1000.0  # g to mg

        # Insulin rate
        insulin_rate = (delivered_insulin * 1000.0) / max(time_step, 0.001)  # mU/min

        # Solve ODE
        ct = current_time if current_time is not None else 0.0
        sol = solve_ivp(
            fun=lambda t, y: self._ode(t, y, insulin_rate, ct),
            t_span=(0.0, time_step),
            y0=self._state,
            method="RK45",
            max_step=1.0,
            rtol=1e-6,
            atol=1e-8,
        )

        self._state = sol.y[:, -1].copy()

        # Derive glucose from Q1
        V_G_dL = self._glucose_volume_dl()
        raw_glucose = self._state[0] / V_G_dL

        self.current_glucose = self._guard_glucose_transition(float(raw_glucose), time_step)

        # Override Q1 to match bounded glucose
        self._state[0] = self.current_glucose * V_G_dL

        # Clamp positive
        for i in range(len(self._state)):
            self._state[i] = max(0.0, self._state[i])

        return self.current_glucose

    def get_current_glucose(self) -> float:
        return self.current_glucose

    def trigger_event(self, event_type: str, value: Any) -> None:
        if event_type == "exercise":
            self.start_exercise(float(value))
        elif event_type in {"stress", "illness"}:
            self.start_stress(float(value))

    def get_patient_state(self) -> Dict[str, float]:
        return {
            "current_glucose": self.current_glucose,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "basal_rate_u_per_hr": self.basal_insulin_rate,
            "isf": self.insulin_sensitivity,
            "icr": self.carb_factor,
            "dia_minutes": self.insulin_action_duration,
            "max_glucose_rate_mgdl_per_min": self.max_glucose_rate_mgdl_per_min,
        }

    def get_ratio_state(self) -> Dict[str, float]:
        return {
            "basal_rate_u_per_hr": self.basal_insulin_rate,
            "isf": self.insulin_sensitivity,
            "icr": self.carb_factor,
            "dia_minutes": self.insulin_action_duration,
        }

    def set_ratio_state(
        self,
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        basal_rate: Optional[float] = None,
        dia_minutes: Optional[float] = None,
    ) -> None:
        if isf is not None:
            self.insulin_sensitivity = float(isf)
        if icr is not None:
            self.carb_factor = float(icr)
        if basal_rate is not None:
            self.basal_insulin_rate = float(basal_rate)
        if dia_minutes is not None:
            self.insulin_action_duration = float(dia_minutes)

    def get_state(self) -> Dict[str, Any]:
        return {
            "ode_state": self._state.tolist(),
            "current_glucose": self.current_glucose,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "active_insulin_doses": self.active_insulin_doses,
            "active_carb_intakes": self.active_carb_intakes,
            "is_exercising": self.is_exercising,
            "exercise_intensity": self.exercise_intensity,
            "is_stressed": self.is_stressed,
            "stress_intensity": self.stress_intensity,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if "ode_state" in state:
            ode_state = np.array(state["ode_state"], dtype=np.float64)
            self._state = self._coerce_legacy_ode_state(ode_state)
        self.current_glucose = state.get("current_glucose", self.current_glucose)
        self.insulin_on_board = state.get("insulin_on_board", self.insulin_on_board)
        self.carbs_on_board = state.get("carbs_on_board", self.carbs_on_board)
        self.active_insulin_doses = state.get("active_insulin_doses", [])
        self.active_carb_intakes = state.get("active_carb_intakes", [])
        self.is_exercising = state.get("is_exercising", False)
        self.exercise_intensity = state.get("exercise_intensity", 0.0)
        self.is_stressed = state.get("is_stressed", False)
        self.stress_intensity = state.get("stress_intensity", 0.0)

    # ------------------------------------------------------------------
    # ODE right-hand-side
    # ------------------------------------------------------------------

    def _ode(
        self,
        t: float,
        y: np.ndarray,
        u_insulin: float,
        current_time: float,
    ) -> np.ndarray:
        Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2 = y
        p = self.params

        V_I = p.V_I_per_kg * p.body_weight_kg
        V_G_dL = self._glucose_volume_dl()

        G = Q1 / V_G_dL

        # Carbohydrate absorption
        dD1_dt = -D1 / p.t_max_G
        dD2_dt = D1 / p.t_max_G - D2 / p.t_max_G
        U_G = (D2 / p.t_max_G) * p.A_G  # mg/min

        # Insulin absorption
        dS1_dt = u_insulin - S1 / p.t_max_I
        dS2_dt = S1 / p.t_max_I - S2 / p.t_max_I
        U_I = S2 / p.t_max_I  # mU/min

        # Plasma insulin
        dI_dt = U_I / V_I - p.k_e * I

        # Stress and Exercise modulation (approximated for Hovorka)
        stress_sens_multiplier = 1.0 - 0.7 * self.stress_intensity if self.is_stressed else 1.0
        stress_EGP_multiplier = 1.0 + 0.5 * self.stress_intensity if self.is_stressed else 1.0

        ex_sens_multiplier = 1.0 + 2.0 * self.exercise_intensity if self.is_exercising else 1.0

        overall_sens = stress_sens_multiplier * ex_sens_multiplier

        k_b1 = p.S_IT * p.k_a1 * overall_sens
        k_b2 = p.S_ID * p.k_a2 * overall_sens
        k_b3 = p.S_IE * p.k_a3 * overall_sens

        # Insulin action
        dx1_dt = -p.k_a1 * x1 + k_b1 * I
        dx2_dt = -p.k_a2 * x2 + k_b2 * I
        dx3_dt = -p.k_a3 * x3 + k_b3 * I

        # Glucose kinetics
        F_01c = p.F_01c_per_kg * p.body_weight_kg
        EGP_0 = p.EGP_0_per_kg * p.body_weight_kg * stress_EGP_multiplier

        # Renal clearance
        F_R = max(0.0, 0.003 * (G - 162.0) * V_G_dL) if G > 162.0 else 0.0

        # Dawn phenomenon
        dawn = 0.0
        if self.dawn_phenomenon_strength > 0:
            minutes_in_day = current_time % 1440
            ds = self.dawn_start_hour * 60
            de = self.dawn_end_hour * 60
            if ds <= minutes_in_day <= de:
                # Add dawn phenomenon to endogenous production
                dawn = (self.dawn_phenomenon_strength / 60.0) * V_G_dL

        dQ1_dt = (
            -(F_01c + F_R)
            - x1 * Q1
            + p.k_12 * Q2
            + EGP_0 * max(0.0, 1 - x3)
            + U_G
            + dawn
        )
        dQ2_dt = x1 * Q1 - (p.k_12 + x2) * Q2

        return np.array(
            [dQ1_dt, dQ2_dt, dS1_dt, dS2_dt, dI_dt, dx1_dt, dx2_dt, dx3_dt, dD1_dt, dD2_dt]
        )
