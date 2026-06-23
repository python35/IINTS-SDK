"""
Improved Hovorka Model - IINTS-AF
==================================
Based on standard Hovorka artificial pancreas equations and extended
to match the IINTS simulator's interface.

State vector (19 variables):
0: Q1 (mg) - Accessible glucose
1: Q2 (mg) - Non-accessible glucose
2: S1 (mU) - SubQ insulin pool 1
3: S2 (mU) - SubQ insulin pool 2
4: I (mU/L) - Plasma insulin
5: x1 (1/min) - Insulin action on distribution
6: x2 (1/min) - Insulin action on disposal
7: x3 (1) - Insulin action on EGP
8: D1 (mg) - Stomach Solid carbs
9: D2 (mg) - Stomach Liquid carbs
10: D3 (mg) - Gut carbs
11: H_stress (1) - Adrenaline/Cortisol pseudo-hormone
12: H_exercise (1) - Endorphin/AMPK pseudo-hormone
13: Y1 (pg/mL) - SubQ Glucagon pool 1
14: Y2 (pg/mL) - SubQ Glucagon pool 2
15: Gamma (pg/mL) - Plasma Glucagon
16: x_gluc (1) - Glucagon action on EGP
17: HAAF (1) - Hypoglycemia-Associated Autonomic Failure (Memory)
18: GLUT4_active (1) - Non-Insulin-Mediated Glucose Uptake via Exercise
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .physiology import smooth_threshold_excess


@dataclass
class HovorkaParameters:
    """Physiological parameters for the Hovorka Model."""

    body_weight_kg: float = 70.0

    # Insulin absorption
    t_max_I: float = 55.0  # min

    # Carb absorption (Dalla Man params)
    k_min: float = 0.008  # 1/min
    k_max: float = 0.05   # 1/min
    A_G: float = 0.8  # bioavailability

    # Insulin type (Physics PK)
    insulin_type: str = "novolog" # Options: fiasp, novolog, regular
    t_max_I_override: Optional[float] = None

    # Glucagon PK/PD
    t_max_glucagon: float = 30.0 # min
    k_e_glucagon: float = 0.1 # 1/min
    V_glucagon_per_kg: float = 0.2 # L/kg
    k_a_glucagon: float = 0.05 # 1/min (activation rate on EGP)
    S_glucagon: float = 0.02 # Sensitivity of liver to glucagon

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
        hovorka_params: Optional[HovorkaParameters] = None,
    ) -> None:
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
        self.last_delivered_insulin_units = 0.0
        self.last_delivered_glucagon_mg = 0.0

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

        # State vector: [Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2, D3, H_stress, H_exercise, Y1, Y2, Gamma, x_gluc, HAAF, GLUT4_active]
        return np.array(
            [Q1_init, Q2_init, 0.0, 0.0, I_basal, x1_init, x2_init, x3_init, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    def _coerce_legacy_ode_state(self, ode_state: np.ndarray) -> np.ndarray:
        """Load older Bergman/custom snapshots into a safe Hovorka state."""
        if ode_state.size == 19:
            return ode_state.astype(np.float64, copy=True)
        if ode_state.size in {10, 11, 12, 13, 18}:
            coerced = self._default_ode_state()
            coerced[: ode_state.size] = ode_state.astype(np.float64, copy=False)
            return coerced

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
            coerced[8] = max(0.0, float(ode_state[4])) # fallback to D1
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
        true_carbs = carb_intake * self.meal_mismatch_epsilon
        self.last_delivered_insulin_units = max(0.0, float(delivered_insulin))
        self.last_delivered_glucagon_mg = max(0.0, float(delivered_glucagon_mg))

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

        # Insulin and Glucagon rates
        insulin_rate = (delivered_insulin * 1000.0) / max(time_step, 0.001)  # mU/min
        glucagon_rate = (delivered_glucagon_mg * 1e6) / max(time_step, 0.001) # pg/min

        # Solve ODE
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

        # Derive glucose from Q1
        V_G_dL = self._glucose_volume_dl()
        raw_glucose = self._state[0] / V_G_dL

        self.current_glucose = self._guard_glucose_transition(float(raw_glucose), time_step)

        # Override Q1 to match bounded glucose
        self._state[0] = self.current_glucose * V_G_dL

        # Clamp positive
        for i in range(len(self._state)):
            self._state[i] = max(0.0, self._state[i])
        self._state[17] = float(np.clip(self._state[17], 0.0, 1.0))

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
            "delivered_insulin": self.last_delivered_insulin_units,
            "last_delivered_insulin_units": self.last_delivered_insulin_units,
            "delivered_insulin_iob": self.insulin_on_board,
            "active_insulin": float(self._state[4]),
            "plasma_glucagon_pg_ml": float(self._state[15]),
            "haaf_metric": float(self._state[17]),
            "glut4_active": float(self._state[18]),
            "insulin_effect": float(self._state[5] + self._state[6] + self._state[7]),
            "plasma_insulin_mU_L": float(self._state[4]),
            "remote_insulin_action_x1": float(self._state[5]),
            "remote_insulin_action_x2": float(self._state[6]),
            "remote_insulin_action_x3": float(self._state[7]),
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
            self._state = self._coerce_legacy_ode_state(ode_state)
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
        u_insulin: float,
        u_glucagon: float,
        current_time: float,
    ) -> np.ndarray:
        Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2, D3, H_stress, H_exercise, Y1, Y2, Gamma, x_gluc, HAAF, GLUT4_active = y
        p = self.params

        V_I = p.V_I_per_kg * p.body_weight_kg

        # Dehydration alters V_G. For extreme science, stress slightly decreases volume.
        dehydration_factor = 1.0 - 0.05 * H_stress
        V_G_dL = self._glucose_volume_dl() * dehydration_factor

        G = Q1 / V_G_dL

        # Biochemistry: Dalla Man 3-compartment meal kinetics
        # GLP-1 Incretin Non-linear Feedback:
        # High glucose in Gut (D3) triggers GLP-1, slowing down stomach emptying.
        D_total = D1 + D2
        if D_total > 0:
            k_empt_base = p.k_min + (p.k_max - p.k_min) / 2.0 * (1.0 - np.tanh(5 * (D_total - 50000) / 50000))
            # GLP-1 effect: up to 50% reduction in k_empt based on D3
            glp1_inhibition = 1.0 / (1.0 + (D3 / 20000.0)**2)
            k_empt = k_empt_base * glp1_inhibition
        else:
            k_empt = p.k_min

        k_solid = p.k_max  # Solid to liquid conversion
        k_abs = 0.05 # Intestinal absorption

        dD1_dt = -k_solid * D1
        dD2_dt = k_solid * D1 - k_empt * D2
        dD3_dt = k_empt * D2 - k_abs * D3
        U_G = (k_abs * D3) * p.A_G  # mg/min

        # Physics/Pharmacokinetics: Insulin Diffusion
        # Determine t_max_I based on insulin_type
        if p.t_max_I_override is not None:
            t_max_I = p.t_max_I_override
        elif p.insulin_type == "fiasp":
            t_max_I = 35.0
        elif p.insulin_type == "regular":
            t_max_I = 90.0
        else:
            t_max_I = 55.0 # novolog

        dS1_dt = u_insulin - S1 / t_max_I
        dS2_dt = S1 / t_max_I - S2 / t_max_I
        U_I = S2 / t_max_I  # mU/min

        # Plasma insulin
        dI_dt = U_I / V_I - p.k_e * I

        # Exogenous Glucagon Kinetics (Bi-hormonal PK/PD)
        V_glucagon = p.V_glucagon_per_kg * p.body_weight_kg
        dY1_dt = u_glucagon - Y1 / p.t_max_glucagon
        dY2_dt = Y1 / p.t_max_glucagon - Y2 / p.t_max_glucagon
        U_Gamma = Y2 / p.t_max_glucagon
        dGamma_dt = U_Gamma / V_glucagon - p.k_e_glucagon * Gamma
        dx_gluc_dt = -p.k_a_glucagon * x_gluc + p.S_glucagon * p.k_a_glucagon * Gamma

        # Endocrinology: Hormonal Kinetics (Adrenaline/Cortisol)
        target_stress = self.stress_intensity if self.is_stressed else 0.0
        target_exercise = self.exercise_intensity if self.is_exercising else 0.0

        dH_stress_dt = (target_stress - H_stress) / 20.0  # 20 min model time constant.
        dH_exercise_dt = (target_exercise - H_exercise) / 10.0  # 10 min model time constant.

        stress_sens_multiplier = 1.0 - 0.7 * H_stress
        stress_EGP_multiplier = 1.0 + 0.5 * H_stress
        ex_sens_multiplier = 1.0 + 2.0 * H_exercise

        overall_sens = stress_sens_multiplier * ex_sens_multiplier

        k_b1 = p.S_IT * p.k_a1 * overall_sens
        k_b2 = p.S_ID * p.k_a2 * overall_sens
        k_b3 = p.S_IE * p.k_a3 * overall_sens

        # Insulin action
        dx1_dt = -p.k_a1 * x1 + k_b1 * I
        dx2_dt = -p.k_a2 * x2 + k_b2 * I
        dx3_dt = -p.k_a3 * x3 + k_b3 * I

        # Celbiology: GLUT4 Translocation Kinetics (Exercise)
        # Exercise brings GLUT4 to the membrane independently of insulin (NIMGU)
        k_glut4_activation = 0.05
        k_glut4_deactivation = 0.01
        dGLUT4_active_dt = k_glut4_activation * H_exercise * (1.0 - GLUT4_active) - k_glut4_deactivation * GLUT4_active

        # Glucose kinetics
        # Fourier-series circadian EGP term for dawn/cortisol/GH research.
        # It is intentionally gated by dawn_phenomenon_strength so default
        # Hovorka runs do not drift from an unexplained always-on oscillator.
        time_of_day_min = current_time % 1440
        dawn_midpoint_min = ((self.dawn_start_hour + self.dawn_end_hour) / 2.0) * 60.0
        phase_shift = 2.0 * np.pi * (time_of_day_min - dawn_midpoint_min) / 1440.0
        circadian_wave = 0.15 * np.cos(phase_shift) + 0.05 * np.cos(2.0 * phase_shift)
        dawn_scale = float(np.clip(self.dawn_phenomenon_strength / 20.0, 0.0, 1.0))
        circadian_EGP = 1.0 + dawn_scale * circadian_wave

        F_01c = p.F_01c_per_kg * p.body_weight_kg

        # Endogenous Rescue & HAAF
        # When G < 70, body naturally spikes EGP to survive (Adrenaline/Glucagon burst).
        # But HAAF blunts this response.
        hypo_delta = max(0.0, 70.0 - G)
        rescue_multiplier = 1.0 + (hypo_delta / 10.0) * (1.0 - HAAF)

        # HAAF Memory Dynamics
        # Builds up quickly when low, decays slowly (24h) when normal
        k_haaf_build = 0.005
        k_haaf_decay = 1.0 / (24 * 60)
        dHAAF_dt = k_haaf_build * hypo_delta * (1.0 - HAAF) - k_haaf_decay * HAAF

        EGP_0 = p.EGP_0_per_kg * p.body_weight_kg * stress_EGP_multiplier * rescue_multiplier * circadian_EGP

        # Physiological Renal Clearance (Sigmoid GFR curve instead of hard cutoff)
        # Smoothly increases glucosuria above 162 mg/dL
        softplus_diff = smooth_threshold_excess(G, threshold=162.0, splay=10.0)
        F_R = 0.003 * V_G_dL * softplus_diff

        # Mass balance ODEs for Glucose Compartments
        # F_01c gets enhanced by active GLUT4 (Non-Insulin-Mediated Glucose Uptake)
        NIMGU = F_01c * (1.0 + 1.5 * GLUT4_active)

        dQ1_dt = (
            -(NIMGU + F_R)
            - x1 * Q1
            + p.k_12 * Q2
            + EGP_0 * max(0.0, 1 - x3 + x_gluc)
            + U_G
        )
        dQ2_dt = x1 * Q1 - (p.k_12 + x2) * Q2

        return np.array(
            [dQ1_dt, dQ2_dt, dS1_dt, dS2_dt, dI_dt, dx1_dt, dx2_dt, dx3_dt, dD1_dt, dD2_dt, dD3_dt, dH_stress_dt, dH_exercise_dt, dY1_dt, dY2_dt, dGamma_dt, dx_gluc_dt, dHAAF_dt, dGLUT4_active_dt]
        )
