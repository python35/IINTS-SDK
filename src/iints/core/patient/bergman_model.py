"""
Bergman Minimal Model — IINTS-AF
==================================
ODE-based patient model inspired by the Bergman Minimal Model with an
additional gut absorption compartment for realistic carbohydrate dynamics.

This provides a more physiologically accurate glucose simulation than the
default ``CustomPatientModel``, at the cost of higher computational load
(uses ``scipy.integrate.solve_ivp``).

The model tracks four state variables:

* **G** — plasma glucose concentration (mg/dL)
* **X** — remote insulin action (1/min)
* **I** — plasma insulin concentration (mU/L)
* **Q_gut** — gut glucose mass (mg)

References
----------
* Bergman, R. N. et al. (1979). Quantitative estimation of insulin
  sensitivity. *Am J Physiol*, 236(6), E667–E677.
* Dalla Man, C. et al. (2007). Meal Simulation Model of the Glucose-
  Insulin System. *IEEE Trans Biomed Eng*, 54(10), 1740–1749.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class BergmanParameters:
    """Physiological parameters for the Bergman Minimal Model."""

    # --- Glucose sub-system ---
    p1: float = 0.028       # 1/min  — insulin-independent glucose uptake
    p2: float = 0.025       # 1/min  — rate of remote insulin degradation
    p3: float = 5.0e-6      # (mU/L)^-1 min^-2 — insulin action gain
    Gb: float = 120.0       # mg/dL  — basal glucose concentration
    Vg: float = 1.569       # dL/kg  — glucose distribution volume

    # --- Insulin sub-system ---
    n: float = 0.23         # 1/min  — fractional insulin degradation
    Ib: float = 7.0         # mU/L   — basal plasma insulin
    Vi: float = 0.05        # L/kg   — insulin distribution volume
    gamma: float = 0.004    # (mU/L)/(mg/dL)/min — endogenous secretion gain
    h: float = 80.0         # mg/dL  — secretion glucose threshold

    # --- Gut absorption ---
    tau_meal: float = 40.0  # min    — gastric emptying time constant
    k_abs: float = 0.05     # 1/min  — intestinal absorption rate constant
    f_bio: float = 0.90     # —      — bioavailability (fraction absorbed)

    # --- Patient physical ---
    body_weight_kg: float = 70.0


class BergmanPatientModel:
    """
    ODE-based patient model providing the same interface as
    ``CustomPatientModel`` for drop-in use with the IINTS Simulator.
    """

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
        bergman_params: Optional[BergmanParameters] = None,
    ) -> None:
        # Store clinical knobs (for ratio queries and compatibility)
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

        # Bergman ODE parameters
        self.params = bergman_params if bergman_params else BergmanParameters(Gb=initial_glucose)

        # Exercise book-keeping
        self.is_exercising = False
        self.exercise_intensity = 0.0
        self.exercise_glucose_consumption_rate = 1.5  # mg/dL per min at max

        # Dose/carb trackers for IOB/COB (same format as CustomPatientModel)
        self.active_insulin_doses: List[Dict[str, float]] = []
        self.active_carb_intakes: List[Dict[str, float]] = []

        # Derived scalar state
        self.current_glucose = initial_glucose
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0
        self.meal_effect_delay = 30  # kept for API compat

        # ODE state vector: [G, X, I, Q_gut]
        self._state = np.array([
            initial_glucose,       # G  (mg/dL)
            0.0,                   # X  (1/min)
            self.params.Ib,        # I  (mU/L)
            0.0,                   # Q_gut (mg)
        ], dtype=np.float64)

        self.reset()

    # ------------------------------------------------------------------
    # Public interface (mirrors CustomPatientModel exactly)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to initial conditions."""
        self._state = np.array([
            self.initial_glucose, 0.0, self.params.Ib, 0.0,
        ], dtype=np.float64)
        self.current_glucose = self.initial_glucose
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0
        self.active_insulin_doses = []
        self.active_carb_intakes = []
        self.is_exercising = False
        self.exercise_intensity = 0.0

    def start_exercise(self, intensity: float) -> None:
        if not (0.0 <= intensity <= 1.0):
            raise ValueError("Exercise intensity must be between 0.0 and 1.0")
        self.is_exercising = True
        self.exercise_intensity = intensity

    def stop_exercise(self) -> None:
        self.is_exercising = False
        self.exercise_intensity = 0.0

    def update(
        self,
        time_step: float,
        delivered_insulin: float,
        carb_intake: float = 0.0,
        current_time: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Advance the model by *time_step* minutes and return new glucose."""
        true_carbs = carb_intake * self.meal_mismatch_epsilon

        # --- Track IOB (same bookkeeping as CustomPatientModel) ---
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

        # --- Track COB ---
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

        # --- Inject carbs into gut compartment ---
        # Add carbs as glucose mass (mg) into Q_gut
        if true_carbs > 0:
            self._state[3] += true_carbs * 1000.0  # g -> mg

        # --- Prepare exogenous insulin rate ---
        # Convert Units to mU, spread evenly over time_step (mU/min)
        insulin_rate = (delivered_insulin * 1000.0) / max(time_step, 0.001)

        # --- Solve ODE ---
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
        # Floor glucose at 20 mg/dL (physiological minimum)
        self._state[0] = max(20.0, self._state[0])
        # Clamp non-negative for other compartments
        self._state[1] = max(0.0, self._state[1])
        self._state[2] = max(0.0, self._state[2])
        self._state[3] = max(0.0, self._state[3])

        self.current_glucose = float(self._state[0])
        return self.current_glucose

    def get_current_glucose(self) -> float:
        return self.current_glucose

    def trigger_event(self, event_type: str, value: Any) -> None:
        pass  # handled by the simulator

    def get_patient_state(self) -> Dict[str, float]:
        return {
            "current_glucose": self.current_glucose,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "basal_rate_u_per_hr": self.basal_insulin_rate,
            "isf": self.insulin_sensitivity,
            "icr": self.carb_factor,
            "dia_minutes": self.insulin_action_duration,
            "plasma_insulin_mU_L": float(self._state[2]),
            "remote_insulin_action": float(self._state[1]),
            "gut_glucose_mg": float(self._state[3]),
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
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if "ode_state" in state:
            self._state = np.array(state["ode_state"], dtype=np.float64)
        self.current_glucose = state.get("current_glucose", self.current_glucose)
        self.insulin_on_board = state.get("insulin_on_board", self.insulin_on_board)
        self.carbs_on_board = state.get("carbs_on_board", self.carbs_on_board)
        self.active_insulin_doses = state.get("active_insulin_doses", [])
        self.active_carb_intakes = state.get("active_carb_intakes", [])
        self.is_exercising = state.get("is_exercising", False)
        self.exercise_intensity = state.get("exercise_intensity", 0.0)

    # ------------------------------------------------------------------
    # ODE right-hand-side
    # ------------------------------------------------------------------

    def _ode(
        self,
        t: float,
        y: np.ndarray,
        u_insulin_mu_per_min: float,
        current_time: float,
    ) -> np.ndarray:
        G, X, I, Q_gut = y
        p = self.params

        Vg_abs = p.Vg * p.body_weight_kg   # dL
        Vi_abs = p.Vi * p.body_weight_kg    # L

        # --- Glucose rate of appearance from gut ---
        Ra = (p.k_abs * Q_gut) / Vg_abs  # mg/dL/min

        # --- Dawn phenomenon ---
        dawn = 0.0
        if self.dawn_phenomenon_strength > 0:
            minutes_in_day = current_time % 1440
            ds = self.dawn_start_hour * 60
            de = self.dawn_end_hour * 60
            if ds <= minutes_in_day <= de:
                dawn = self.dawn_phenomenon_strength / 60.0  # mg/dL/min

        # --- Exercise ---
        exercise = 0.0
        if self.is_exercising:
            exercise = self.exercise_intensity * self.exercise_glucose_consumption_rate

        # --- dG/dt ---
        dGdt = -(p.p1 + X) * G + p.p1 * p.Gb + Ra + dawn - exercise

        # --- dX/dt ---
        dXdt = -p.p2 * X + p.p3 * max(I - p.Ib, 0.0)

        # --- dI/dt ---
        # Endogenous pancreatic secretion (blunted in T1D, but kept for generality)
        secretion = p.gamma * max(G - p.h, 0.0)
        dIdt = -p.n * (I - p.Ib) + secretion + u_insulin_mu_per_min / Vi_abs

        # --- dQ_gut/dt ---
        # Q_gut decays as glucose is absorbed into plasma
        dQ_gut_dt = -p.k_abs * Q_gut

        return np.array([dGdt, dXdt, dIdt, dQ_gut_dt])
