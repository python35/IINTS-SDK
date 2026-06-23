"""
Bergman Minimal Model — IINTS-AF
==================================
ODE-based patient model inspired by the Bergman Minimal Model with an
additional gut absorption compartment for realistic carbohydrate dynamics.

This provides a more physiologically accurate glucose simulation than the
default ``CustomPatientModel``, at the cost of higher computational load
(uses ``scipy.integrate.solve_ivp``).

The model tracks 13 state variables:

* **G** — plasma glucose concentration (mg/dL)
* **X** — remote insulin action (1/min)
* **I** — plasma insulin concentration (mU/L)
* **Q_stomach** — stomach glucose mass waiting for gastric emptying (mg)
* **Q_gut** — intestinal glucose mass available for absorption (mg)
* **S1/S2** — SubQ insulin pools
* **Y1/Y2/Gamma/x_gluc** — Glucagon subQ & plasma kinetics
* **HAAF** — Hypoglycemia-Associated Autonomic Failure memory

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

from .physiology import smooth_threshold_excess


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
    gamma: float = 0.0      # (mU/L)/(mg/dL)/min — endogenous secretion gain (0 for T1D default)
    h: float = 80.0         # mg/dL  — secretion glucose threshold
    k_a: float = 0.018      # 1/min  — subcutaneous insulin absorption rate constant

    # --- Exogenous Glucagon ---
    t_max_glucagon: float = 30.0  # min
    k_e_glucagon: float = 0.1     # 1/min
    V_glucagon_per_kg: float = 0.2 # L/kg
    k_a_glucagon: float = 0.05    # 1/min
    S_glucagon: float = 0.02      # sensitivity

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
        basal_glucose_target: Optional[float] = None,
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
        bergman_params: Optional[BergmanParameters] = None,
    ) -> None:
        # Store clinical knobs (for ratio queries and compatibility)
        self.basal_insulin_rate = basal_insulin_rate
        self.insulin_sensitivity = insulin_sensitivity
        self.carb_factor = carb_factor
        self.initial_glucose = initial_glucose
        self.basal_glucose_target = basal_glucose_target
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

        # Bergman ODE parameters
        basal_target = float(initial_glucose) if basal_glucose_target is None else float(basal_glucose_target)
        gb_default = min(float(initial_glucose), 115.0) if basal_glucose_target is None else float(np.clip(basal_target, 80.0, 220.0))
        self.params = bergman_params if bergman_params else BergmanParameters(
            Gb=gb_default,
            tau_meal=max(45.0, min(float(carb_absorption_duration_minutes) / 3.0, 100.0)),
            k_abs=max(0.015, min(float(glucose_absorption_rate), 0.035)),
        )

        # Exercise book-keeping
        self.is_exercising = False
        self.exercise_intensity = 0.0
        self.exercise_glucose_consumption_rate = 1.5  # mg/dL per min at max

        # Stress book-keeping
        self.is_stressed = False
        self.stress_intensity = 0.0

        # Dose/carb trackers for IOB/COB (same format as CustomPatientModel)
        self.active_insulin_doses: List[Dict[str, float]] = []
        self.active_carb_intakes: List[Dict[str, float]] = []

        # Derived scalar state
        self.current_glucose = initial_glucose
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0
        self.last_delivered_insulin_units = 0.0
        self.last_delivered_glucagon_mg = 0.0
        self.meal_effect_delay = 30  # kept for API compat

        # ODE state vector: [G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF]
        self._state = np.array([
            initial_glucose,       # 0: G  (mg/dL)
            0.0,                   # 1: X  (1/min)
            self.params.Ib,        # 2: I  (mU/L)
            0.0,                   # 3: Q_sto1 (mg) - Solid Stomach
            0.0,                   # 4: Q_sto2 (mg) - Liquid Stomach
            0.0,                   # 5: Q_gut (mg)  - Intestine
            0.0,                   # 6: S1 (mU)
            0.0,                   # 7: S2 (mU)
            0.0,                   # 8: Y1 (pg) - Glucagon subQ 1
            0.0,                   # 9: Y2 (pg) - Glucagon subQ 2
            0.0,                   # 10: Gamma (pg/mL) - Plasma Glucagon
            0.0,                   # 11: x_gluc (1) - Glucagon action on EGP
            0.0,                   # 12: HAAF (1) - Memory
        ], dtype=np.float64)

        self.reset()

    def _guard_glucose_transition(self, proposed_glucose: float, time_step: float) -> float:
        """Bound solver output to a plausible research-simulator transition."""
        if not np.isfinite(proposed_glucose):
            return float(self.current_glucose)
        max_rate = float(getattr(self, "max_glucose_rate_mgdl_per_min", 0.0) or 0.0)
        if max_rate <= 0.0:
            return float(max(20.0, proposed_glucose))
        max_delta = max_rate * max(float(time_step), 0.0)
        requested_delta = float(proposed_glucose) - float(self.current_glucose)
        bounded_delta = float(np.clip(requested_delta, -max_delta, max_delta))
        return float(max(20.0, self.current_glucose + bounded_delta))

    # ------------------------------------------------------------------
    # Public interface (mirrors CustomPatientModel exactly)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to initial conditions."""
        self._state = np.array([
            self.initial_glucose, 0.0, self.params.Ib, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float64)
        self.current_glucose = self.initial_glucose
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0
        self.last_delivered_insulin_units = 0.0
        self.last_delivered_glucagon_mg = 0.0
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
        delivered_glucagon_mg: float = 0.0,
        current_time: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Advance the model by *time_step* minutes and return new glucose."""
        true_carbs = carb_intake * self.meal_mismatch_epsilon
        self.last_delivered_insulin_units = max(0.0, float(delivered_insulin))
        self.last_delivered_glucagon_mg = max(0.0, float(delivered_glucagon_mg))

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

        # --- Inject carbs into stomach compartment ---
        if true_carbs > 0:
            bioavailability = max(0.0, min(float(self.params.f_bio), 1.0))
            self._state[3] += true_carbs * bioavailability * 1000.0  # g -> mg (into solid stomach Q_sto1)

        # --- Prepare exogenous insulin/glucagon rate ---
        insulin_rate = (delivered_insulin * 1000.0) / max(time_step, 0.001)
        glucagon_rate = (delivered_glucagon_mg * 1e6) / max(time_step, 0.001) # pg/min

        # --- Solve ODE ---
        ct = current_time if current_time is not None else 0.0
        sol = solve_ivp(
            fun=lambda t, y: self._ode(t, y, insulin_rate, glucagon_rate, ct),
            t_span=(0.0, time_step),
            y0=self._state,
            method="RK45",
            max_step=1.0,
            rtol=1e-6,
            atol=1e-8,
        )

        self._state = sol.y[:, -1].copy()
        self._state[0] = self._guard_glucose_transition(float(self._state[0]), time_step)
        # Clamp non-negative for other compartments
        for i in range(1, len(self._state)):
            self._state[i] = max(0.0, self._state[i])
        self._state[12] = float(np.clip(self._state[12], 0.0, 1.0))

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
            "stomach_glucose_mg": float(self._state[3] + self._state[4]),
            "stomach_solid_mg": float(self._state[3]),
            "stomach_liquid_mg": float(self._state[4]),
            "gut_glucose_mg": float(self._state[5]),
            "subcut_insulin_1_mU": float(self._state[6]),
            "subcut_insulin_2_mU": float(self._state[7]),
            "plasma_glucagon_pg_ml": float(self._state[10]),
            "haaf_metric": float(self._state[12]),
            "max_glucose_rate_mgdl_per_min": self.max_glucose_rate_mgdl_per_min,
            "delivered_insulin": self.last_delivered_insulin_units,
            "last_delivered_insulin_units": self.last_delivered_insulin_units,
            "delivered_insulin_iob": self.insulin_on_board,
            "active_insulin": float(self._state[2]),
            "insulin_effect": float(self._state[1]),
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
            "last_delivered_insulin_units": self.last_delivered_insulin_units,
            "last_delivered_glucagon_mg": self.last_delivered_glucagon_mg,
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
            # Handle legacy snapshot coercions to 13-state vector
            if ode_state.size == 4:
                ode_state = np.array(
                    [ode_state[0], ode_state[1], ode_state[2], 0.0, 0.0, ode_state[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                )
            elif ode_state.size == 5:
                ode_state = np.array(
                    [ode_state[0], ode_state[1], ode_state[2], ode_state[3], 0.0, ode_state[4], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                )
            elif ode_state.size == 7:
                ode_state = np.array(
                    [ode_state[0], ode_state[1], ode_state[2], ode_state[3], 0.0, ode_state[4], ode_state[5], ode_state[6], 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                )
            elif ode_state.size == 8:
                ode_state = np.array(
                    [ode_state[0], ode_state[1], ode_state[2], ode_state[3], ode_state[4], ode_state[5], ode_state[6], ode_state[7], 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                )
            self._state = ode_state
        self.current_glucose = state.get("current_glucose", self.current_glucose)
        self.insulin_on_board = state.get("insulin_on_board", self.insulin_on_board)
        self.carbs_on_board = state.get("carbs_on_board", self.carbs_on_board)
        self.last_delivered_insulin_units = state.get(
            "last_delivered_insulin_units",
            state.get("delivered_insulin", self.last_delivered_insulin_units),
        )
        self.last_delivered_glucagon_mg = state.get("last_delivered_glucagon_mg", 0.0)
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
        u_insulin_mu_per_min: float,
        u_glucagon_pg_per_min: float,
        current_time: float,
    ) -> np.ndarray:
        G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF = y
        p = self.params

        Vg_abs = p.Vg * p.body_weight_kg   # dL
        Vi_abs = p.Vi * p.body_weight_kg    # L
        V_glucagon = p.V_glucagon_per_kg * p.body_weight_kg

        # --- Glucose rate of appearance from gut ---
        Ra = (p.k_abs * Q_gut) / Vg_abs  # mg/dL/min

        # --- Exogenous Glucagon Kinetics (Bi-hormonal PK/PD) ---
        dY1_dt = u_glucagon_pg_per_min - Y1 / p.t_max_glucagon
        dY2_dt = Y1 / p.t_max_glucagon - Y2 / p.t_max_glucagon
        U_Gamma = Y2 / p.t_max_glucagon
        dGamma_dt = U_Gamma / V_glucagon - p.k_e_glucagon * Gamma
        dx_gluc_dt = -p.k_a_glucagon * x_gluc + p.S_glucagon * p.k_a_glucagon * Gamma

        # --- Dawn phenomenon ---
        dawn = 0.0
        if self.dawn_phenomenon_strength > 0:
            minutes_in_day = current_time % 1440
            ds = self.dawn_start_hour * 60
            de = self.dawn_end_hour * 60
            if ds <= minutes_in_day <= de:
                dawn = self.dawn_phenomenon_strength / 60.0  # mg/dL/min

        # --- Exercise Physiologic Impact ---
        exercise_p1_multiplier = 1.0
        exercise_p3_multiplier = 1.0
        exercise_glucose_uptake = 0.0
        if self.is_exercising:
            exercise_p1_multiplier = 1.0 + 2.0 * self.exercise_intensity
            exercise_p3_multiplier = 1.0 + 2.0 * self.exercise_intensity
            exercise_glucose_uptake = self.exercise_intensity * self.exercise_glucose_consumption_rate

        # --- Stress Physiologic Impact ---
        stress_p1_multiplier = 1.0
        stress_p3_multiplier = 1.0
        stress_Gb_multiplier = 1.0
        if self.is_stressed:
            stress_p1_multiplier = 1.0 - 0.2 * self.stress_intensity
            stress_p3_multiplier = 1.0 - 0.7 * self.stress_intensity
            stress_Gb_multiplier = 1.0 + 0.5 * self.stress_intensity

        # --- Endogenous Rescue & HAAF ---
        hypo_delta = max(0.0, 70.0 - G)
        rescue_multiplier = 1.0 + (hypo_delta / 10.0) * (1.0 - HAAF)

        # HAAF Memory Dynamics
        k_haaf_build = 0.005
        k_haaf_decay = 1.0 / (24 * 60)
        dHAAF_dt = k_haaf_build * hypo_delta * (1.0 - HAAF) - k_haaf_decay * HAAF

        p1_eff = p.p1 * exercise_p1_multiplier * stress_p1_multiplier
        p3_eff = p.p3 * exercise_p3_multiplier * stress_p3_multiplier
        
        # Gb is multiplied by stress, rescue adrenaline, and exogenous glucagon action
        Gb_eff = p.Gb * stress_Gb_multiplier * rescue_multiplier * max(0.0, 1.0 + x_gluc)

        # --- Physiological Renal Clearance ---
        softplus_diff = smooth_threshold_excess(G, threshold=162.0, splay=10.0)
        F_R = 0.003 * softplus_diff

        # --- dG/dt ---
        dGdt = -(p1_eff + X) * G + p1_eff * Gb_eff + Ra + dawn - exercise_glucose_uptake - F_R

        # --- dX/dt ---
        dXdt = -p.p2 * X + p3_eff * max(I - p.Ib, 0.0)

        # --- dS1/dt, dS2/dt (Subcutaneous Insulin Absorption) ---
        dS1dt = u_insulin_mu_per_min - p.k_a * S1
        dS2dt = p.k_a * S1 - p.k_a * S2

        # Rate of appearance of insulin into plasma (mU/min)
        Ra_I = p.k_a * S2

        # --- dI/dt ---
        secretion = p.gamma * max(G - p.h, 0.0)
        dIdt = -p.n * (I - p.Ib) + secretion + Ra_I / Vi_abs

        # --- Dalla Man Multi-compartment Meal Kinetcs ---
        gastric_emptying_rate = 1.0 / max(float(p.tau_meal), 1.0)
        solid_to_liquid_rate = gastric_emptying_rate * 1.5 
        dQ_sto1_dt = -solid_to_liquid_rate * Q_sto1
        dQ_sto2_dt = solid_to_liquid_rate * Q_sto1 - gastric_emptying_rate * Q_sto2
        dQ_gut_dt = gastric_emptying_rate * Q_sto2 - p.k_abs * Q_gut

        return np.array([dGdt, dXdt, dIdt, dQ_sto1_dt, dQ_sto2_dt, dQ_gut_dt, dS1dt, dS2dt, dY1_dt, dY2_dt, dGamma_dt, dx_gluc_dt, dHAAF_dt])
