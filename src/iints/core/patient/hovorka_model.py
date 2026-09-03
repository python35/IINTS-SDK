"""
Adapted Hovorka Research Model - IINTS-AF
==========================================
Based on published Hovorka artificial-pancreas equations and extended with
explicit research stressors to match the IINTS simulator interface. The
extensions are not part of the canonical Hovorka model and are not clinically
validated patient physiology.

State vector (19 variables):
0: Q1 (mg) - Accessible glucose
1: Q2 (mg) - Non-accessible glucose
2: S1 (mU) - SubQ insulin pool 1
3: S2 (mU) - SubQ insulin pool 2
4: I (mU/L) - Plasma insulin
5: x1 (1/min) - Insulin action on distribution
6: x2 (1/min) - Insulin action on disposal
7: x3 (1) - Insulin action on EGP
8: D1 (mg) - First meal-absorption compartment
9: D2 (mg) - Second meal-absorption compartment
10: D3 (mg) - Reserved legacy meal mass (migrated into D2 on restore)
11: H_stress (1) - Adrenaline/Cortisol pseudo-hormone
12: H_exercise (1) - Endorphin/AMPK pseudo-hormone
13: Y1 (pg) - SubQ Glucagon pool 1
14: Y2 (pg) - SubQ Glucagon pool 2
15: Gamma (pg/mL) - Plasma Glucagon
16: x_gluc (1) - Glucagon action on EGP
17: HAAF (1) - Hypoglycemia-Associated Autonomic Failure (Memory)
18: GLUT4_active (1) - Non-Insulin-Mediated Glucose Uptake via Exercise
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .compartments import HOVORKA_COMPARTMENTS, compartment_schema
from .models import PatientModelDomainError
from .physiology import (
    antecedent_hypoglycemia_memory_derivative,
    counterregulatory_rescue_multiplier,
    dawn_glucose_rate_mgdl_min,
    dawn_insulin_sensitivity_multiplier,
    glucagon_mg_to_pg,
    smooth_threshold_excess,
    validated_activity_events,
    validated_snapshot_bool,
    validated_snapshot_scalar,
)


@dataclass
class HovorkaParameters:
    """Physiological parameters for the Hovorka Model."""

    body_weight_kg: float = 70.0

    # Insulin absorption
    t_max_I: float = 55.0  # min

    # Adapted meal-chain parameters
    k_min: float = 0.008  # 1/min
    k_max: float = 0.05   # 1/min
    A_G: float = 0.8  # bioavailability
    t_max_G: float = 40.0  # min; published two-compartment meal time constant

    # Predefined research absorption profiles
    insulin_type: str = "novolog" # Options: fiasp, novolog, regular
    t_max_I_override: Optional[float] = None

    # Glucagon PK/PD. k1 = 1/t_max_glucagon; k2 and apparent clearance use
    # representative values within the Wendt et al. T1D parameter ranges.
    t_max_glucagon: float = 25.0  # min; k1 = 0.04 1/min
    k_e_glucagon: float = 0.165  # 1/min; second transfer/elimination rate k2
    glucagon_clearance_ml_kg_min: float = 120.0
    glucagon_ec50_pg_ml: float = 350.0
    k_a_glucagon: float = 0.05  # 1/min effect-compartment rate
    S_glucagon: float = 1.0  # maximum fractional EGP increase

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

    def __post_init__(self) -> None:
        if self.insulin_type not in {"fiasp", "novolog", "regular"}:
            raise ValueError(
                "insulin_type must be one of: fiasp, novolog, regular"
            )
        numeric = {
            name: float(value)
            for name, value in vars(self).items()
            if name != "insulin_type" and value is not None
        }
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("Hovorka parameters must all be finite")
        positive = (
            "body_weight_kg", "t_max_I", "k_min", "k_max", "t_max_G",
            "t_max_glucagon", "k_e_glucagon",
            "glucagon_clearance_ml_kg_min", "glucagon_ec50_pg_ml",
            "k_a_glucagon", "k_e", "V_I_per_kg", "V_G_per_kg", "k_12",
            "EGP_0_per_kg", "F_01c_per_kg", "k_a1", "k_a2", "k_a3",
            "S_IT", "S_ID", "S_IE",
        )
        for name in positive:
            if numeric[name] <= 0.0:
                raise ValueError(f"Hovorka parameter {name} must be positive")
        if not 0.0 <= numeric["A_G"] <= 1.0:
            raise ValueError("Hovorka parameter A_G must be between 0 and 1")
        if numeric["S_glucagon"] < 0.0:
            raise ValueError("Hovorka parameter S_glucagon must be non-negative")
        if self.t_max_I_override is not None and float(self.t_max_I_override) <= 0.0:
            raise ValueError("t_max_I_override must be positive when provided")


class HovorkaPatientModel:
    def __init__(
        self,
        basal_insulin_rate: float = 0.8,
        insulin_sensitivity: float = 50.0,
        carb_factor: float = 10.0,
        initial_glucose: float = 120.0,
        basal_glucose_target: Optional[float] = None,
        glucose_decay_rate: float = 0.05,
        glucose_absorption_rate: float = 0.025,
        insulin_action_duration: float = 300.0,
        insulin_peak_time: float = 75.0,
        meal_mismatch_epsilon: float = 1.0,
        dawn_phenomenon_strength: float = 0.0,
        dawn_insulin_resistance_fraction: float = 0.0,
        dawn_start_hour: float = 4.0,
        dawn_end_hour: float = 8.0,
        molecular_affinity_scalar: float = 1.0,
        muscle_sensitivity_scalar: float = 1.0,
        liver_sensitivity_scalar: float = 1.0,
        carb_absorption_duration_minutes: float = 240.0,
        max_glucose_rate_mgdl_per_min: float = 3.0,
        hovorka_params: Optional[HovorkaParameters] = None,
    ) -> None:
        values = {
            "basal_insulin_rate": float(basal_insulin_rate),
            "insulin_sensitivity": float(insulin_sensitivity),
            "carb_factor": float(carb_factor),
            "initial_glucose": float(initial_glucose),
            "glucose_decay_rate": float(glucose_decay_rate),
            "glucose_absorption_rate": float(glucose_absorption_rate),
            "insulin_action_duration": float(insulin_action_duration),
            "insulin_peak_time": float(insulin_peak_time),
            "meal_mismatch_epsilon": float(meal_mismatch_epsilon),
            "dawn_phenomenon_strength": float(dawn_phenomenon_strength),
            "dawn_insulin_resistance_fraction": float(dawn_insulin_resistance_fraction),
            "dawn_start_hour": float(dawn_start_hour),
            "dawn_end_hour": float(dawn_end_hour),
            "molecular_affinity_scalar": float(molecular_affinity_scalar),
            "muscle_sensitivity_scalar": float(muscle_sensitivity_scalar),
            "liver_sensitivity_scalar": float(liver_sensitivity_scalar),
            "carb_absorption_duration_minutes": float(carb_absorption_duration_minutes),
            "max_glucose_rate_mgdl_per_min": float(max_glucose_rate_mgdl_per_min),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Hovorka model inputs must all be finite")
        positive = (
            "insulin_sensitivity", "carb_factor", "initial_glucose",
            "glucose_absorption_rate",
            "insulin_action_duration", "insulin_peak_time",
            "meal_mismatch_epsilon",
            "carb_absorption_duration_minutes",
        )
        for name in positive:
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if values["basal_insulin_rate"] < 0.0:
            raise ValueError("basal_insulin_rate must be non-negative")
        for name in (
            "molecular_affinity_scalar",
            "muscle_sensitivity_scalar",
            "liver_sensitivity_scalar",
        ):
            if values[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if values["glucose_decay_rate"] < 0.0:
            raise ValueError("glucose_decay_rate must be non-negative")
        if values["molecular_affinity_scalar"] > 2.0:
            raise ValueError("molecular_affinity_scalar must not exceed 2.0")
        if values["dawn_phenomenon_strength"] < 0.0:
            raise ValueError("dawn_phenomenon_strength must be non-negative")
        if not 0.0 <= values["dawn_insulin_resistance_fraction"] < 1.0:
            raise ValueError(
                "dawn_insulin_resistance_fraction must satisfy 0 <= fraction < 1"
            )
        if values["max_glucose_rate_mgdl_per_min"] < 0.0:
            raise ValueError("max_glucose_rate_mgdl_per_min must be non-negative")
        if not 0.0 <= values["dawn_start_hour"] < values["dawn_end_hour"] <= 24.0:
            raise ValueError(
                "dawn hours must satisfy 0 <= start < end <= 24"
            )
        if basal_glucose_target is not None:
            target = float(basal_glucose_target)
            if not np.isfinite(target) or target < 20.0:
                raise ValueError(
                    "basal_glucose_target must be finite and at least 20 mg/dL"
                )

        self.basal_insulin_rate = values["basal_insulin_rate"]
        self.molecular_affinity_scalar = values["molecular_affinity_scalar"]
        self.muscle_sensitivity_scalar = values["muscle_sensitivity_scalar"]
        self.liver_sensitivity_scalar = values["liver_sensitivity_scalar"]
        # Keep the configured clinical ratio separate from the experimental
        # molecular scalar. The scalar is applied exactly once in the ODE.
        self.insulin_sensitivity = values["insulin_sensitivity"]
        self.carb_factor = values["carb_factor"]
        self.initial_glucose = values["initial_glucose"]
        self.basal_glucose_target = basal_glucose_target
        self.glucose_decay_rate = values["glucose_decay_rate"]
        self.glucose_absorption_rate = values["glucose_absorption_rate"]
        self.insulin_action_duration = values["insulin_action_duration"]
        self.insulin_peak_time = values["insulin_peak_time"]
        self.meal_mismatch_epsilon = values["meal_mismatch_epsilon"]
        self.dawn_phenomenon_strength = values["dawn_phenomenon_strength"]
        self.dawn_insulin_resistance_fraction = values[
            "dawn_insulin_resistance_fraction"
        ]
        self.dawn_start_hour = values["dawn_start_hour"]
        self.dawn_end_hour = values["dawn_end_hour"]
        self.carb_absorption_duration_minutes = values["carb_absorption_duration_minutes"]
        self.max_glucose_rate_mgdl_per_min = values["max_glucose_rate_mgdl_per_min"]

        self.params = hovorka_params if hovorka_params else HovorkaParameters(
            t_max_G=1.0 / max(float(glucose_absorption_rate), 1e-6)
        )
        self._clinical_sensitivity_scale = values["insulin_sensitivity"] / 50.0
        basal_reference = (
            values["initial_glucose"]
            if basal_glucose_target is None
            else float(basal_glucose_target)
        )
        self._basal_parameter_scale = self._derive_basal_parameter_scale(
            basal_reference
        )

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
        self._last_unsupported_event: Optional[Dict[str, Any]] = None

        self.reset()

    def _glucose_volume_dl(self) -> float:
        p = self.params
        return p.V_G_per_kg * p.body_weight_kg * 10.0

    def _insulin_tmax_minutes(self) -> float:
        p = self.params
        if p.t_max_I_override is not None:
            return max(float(p.t_max_I_override), 1.0)
        return {
            "fiasp": 35.0,
            "regular": 90.0,
        }.get(p.insulin_type, 55.0)

    def _derive_basal_parameter_scale(self, glucose_mgdl: float) -> float:
        """Match the configured basal infusion to a fasting steady state.

        Published Hovorka parameters describe a population model, while the
        public SDK accepts a patient-specific basal rate. This explicit scalar
        reconciles those two inputs at the reference ISF (50 mg/dL/U) instead
        of hiding a large startup transient in the initial conditions.
        """

        p = self.params
        basal_input = max(float(self.basal_insulin_rate), 0.0) * 1000.0 / 60.0
        if basal_input <= 0.0:
            return 1.0

        V_I = p.V_I_per_kg * p.body_weight_kg
        basal_insulin = basal_input / max(V_I * p.k_e, 1e-9)
        Q1 = max(float(glucose_mgdl), 20.0) * self._glucose_volume_dl()
        F_01 = p.F_01c_per_kg * p.body_weight_kg
        F_01c = F_01 * min(1.0, max(0.0, float(glucose_mgdl) / 81.0))
        renal = 0.003 * self._glucose_volume_dl() * smooth_threshold_excess(
            float(glucose_mgdl), threshold=162.0, splay=10.0
        )
        EGP_0 = p.EGP_0_per_kg * p.body_weight_kg

        def residual(scale: float) -> float:
            x1 = p.S_IT * basal_insulin * scale
            x2 = p.S_ID * basal_insulin * scale
            x3 = p.S_IE * basal_insulin * scale
            Q2 = x1 * Q1 / max(p.k_12 + x2, 1e-9)
            return (
                -(F_01c + renal)
                - x1 * Q1
                + p.k_12 * Q2
                + EGP_0 * max(0.0, 1.0 - x3)
            )

        low, high = 0.0, 1.0
        while residual(high) > 0.0 and high < 64.0:
            high *= 2.0
        if residual(high) > 0.0:
            return 1.0
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            if residual(midpoint) > 0.0:
                low = midpoint
            else:
                high = midpoint
        return 0.5 * (low + high)

    def _default_ode_state(self, glucose_mgdl: Optional[float] = None) -> np.ndarray:
        p = self.params
        V_G_dL = self._glucose_volume_dl()

        glucose = float(self.initial_glucose if glucose_mgdl is None else glucose_mgdl)
        Q1_init = glucose * V_G_dL

        basal_input = max(float(self.basal_insulin_rate), 0.0) * 1000.0 / 60.0
        t_max_I = self._insulin_tmax_minutes()
        S1_init = basal_input * t_max_I
        S2_init = basal_input * t_max_I
        V_I = p.V_I_per_kg * p.body_weight_kg
        I_basal = basal_input / max(V_I * p.k_e, 1e-9)
        sensitivity = (
            self._basal_parameter_scale
            * self._clinical_sensitivity_scale
            * self.molecular_affinity_scalar
        )
        x1_init = p.S_IT * sensitivity * I_basal
        x2_init = p.S_ID * sensitivity * self.muscle_sensitivity_scalar * I_basal
        x3_init = p.S_IE * sensitivity * self.liver_sensitivity_scalar * I_basal
        Q2_init = x1_init * Q1_init / max(p.k_12 + x2_init, 1e-9)

        # State vector follows the schema documented in the module docstring.
        return np.array(
            [Q1_init, Q2_init, S1_init, S2_init, I_basal, x1_init, x2_init, x3_init, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    def _coerce_legacy_ode_state(self, ode_state: np.ndarray) -> np.ndarray:
        """Load older Bergman/custom snapshots into a safe Hovorka state."""
        if ode_state.size == 19:
            return ode_state.astype(np.float64, copy=True)
        if ode_state.size == 18:
            return np.append(ode_state.astype(np.float64, copy=True), 0.0)
        if ode_state.size != 7:
            raise ValueError(
                f"Unsupported Hovorka ODE snapshot length {ode_state.size}; "
                "expected 7, 18, or 19"
            )
        if not np.all(np.isfinite(ode_state)):
            raise ValueError("Legacy Hovorka snapshot contains non-finite values")
        glucose = float(ode_state[0])
        if glucose < 20.0:
            raise ValueError(
                "Legacy seven-state snapshots must store glucose in mg/dL "
                "as their first value"
            )
        if np.any(ode_state[2:] < 0.0):
            raise ValueError("Legacy Hovorka snapshot contains a negative compartment")

        coerced = self._default_ode_state(glucose_mgdl=glucose)
        # The historical seven-state custom model used
        # [G, X, I, Q_sto, Q_gut, S1, S2]. These quantities are mapped by
        # meaning rather than copied positionally into the 19-state model.
        if ode_state.size == 7:
            coerced[4] = float(ode_state[2])
            coerced[8] = float(ode_state[3])
            coerced[10] = float(ode_state[4])
            coerced[2] = float(ode_state[5])
            coerced[3] = float(ode_state[6])
        return coerced

    def _guard_glucose_transition(self, proposed_glucose: float, time_step: float) -> float:
        if not np.isfinite(proposed_glucose):
            raise PatientModelDomainError(
                "Hovorka ODE produced non-finite glucose",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        if proposed_glucose < 0.0:
            raise PatientModelDomainError(
                f"Hovorka ODE produced negative glucose: {proposed_glucose}",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        max_rate = float(self.max_glucose_rate_mgdl_per_min or 0.0)
        elapsed = max(float(time_step), 1e-9)
        rate = abs(float(proposed_glucose) - float(self.current_glucose)) / elapsed
        if max_rate > 0.0 and rate > max_rate + 1e-9:
            raise PatientModelDomainError(
                f"Hovorka ODE glucose rate {rate:.3f} mg/dL/min exceeds "
                f"configured validation limit {max_rate:.3f}",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        return float(proposed_glucose)

    def reset(self) -> None:
        self._state = self._default_ode_state()
        self._last_input_rates: Tuple[float, float] = (0.0, 0.0)
        self._last_ode_time = 0.0
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
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Hovorka update arguments: {names}")
        if not np.isfinite(time_step) or float(time_step) <= 0.0:
            raise ValueError("time_step must be a finite positive number of minutes")
        if not np.isfinite(delivered_insulin) or float(delivered_insulin) < 0.0:
            raise ValueError("delivered_insulin must be finite and non-negative")
        if not np.isfinite(carb_intake) or float(carb_intake) < 0.0:
            raise ValueError("carb_intake must be finite and non-negative")
        if not np.isfinite(delivered_glucagon_mg) or float(delivered_glucagon_mg) < 0.0:
            raise ValueError("delivered_glucagon_mg must be finite and non-negative")
        if current_time is not None and not np.isfinite(current_time):
            raise ValueError("current_time must be finite when provided")
        previous_state = copy.deepcopy(self.get_state())
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
        glucagon_rate = glucagon_mg_to_pg(delivered_glucagon_mg) / max(time_step, 0.001)

        # Solve ODE
        ct = current_time if current_time is not None else 0.0
        # Kept so flux_snapshot reports the rates and clock this step actually
        # integrated with, instead of repeating the unit conversions above at
        # every call site.
        self._last_input_rates = (float(insulin_rate), float(glucagon_rate))
        self._last_ode_time = float(ct) + float(time_step)
        try:
            sol = solve_ivp(
                fun=lambda t, y: self._ode(
                    t,
                    y,
                    insulin_rate,
                    glucagon_rate,
                    float(ct) + float(t),
                ),
                t_span=(0.0, time_step),
                y0=self._state,
                method="RK45",
                max_step=1.0,
                rtol=1e-6,
                atol=1e-8,
            )

            if (
                not sol.success
                or sol.y.shape[1] == 0
                or not np.all(np.isfinite(sol.y[:, -1]))
            ):
                raise RuntimeError(f"Hovorka ODE integration failed: {sol.message}")
            self._state = sol.y[:, -1].copy()

            # Derive glucose from Q1.
            raw_glucose = self._state[0] / self._glucose_volume_dl()
            self.current_glucose = self._guard_glucose_transition(
                float(raw_glucose), time_step
            )

            if np.any(self._state[1:] < -1e-6):
                minimum = float(np.min(self._state[1:]))
                raise RuntimeError(
                    f"Hovorka ODE produced a negative compartment state: {minimum}"
                )
            for index in (11, 12, 17, 18):
                value = float(self._state[index])
                if value > 1.0 + 1e-6:
                    raise RuntimeError(
                        f"Hovorka ODE produced a fraction state above 1: {value}"
                    )
            # Remove only sub-micro numerical integration noise.
            for i in range(len(self._state)):
                self._state[i] = max(0.0, self._state[i])
            for index in (11, 12, 17, 18):
                self._state[index] = min(1.0, self._state[index])

            return self.current_glucose
        except Exception:
            self.set_state(previous_state)
            raise

    def get_current_glucose(self) -> float:
        return self.current_glucose

    def trigger_event(self, event_type: str, value: Any) -> None:
        if event_type == "exercise":
            self.start_exercise(float(value))
        elif event_type in {"stress", "illness"}:
            self.start_stress(float(value))
        else:
            self._last_unsupported_event = {
                "event_type": str(event_type),
                "value": value,
            }

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
            if not np.isfinite(isf) or float(isf) <= 0.0:
                raise ValueError("isf must be finite and positive")
            self.insulin_sensitivity = float(isf)
            self._clinical_sensitivity_scale = self.insulin_sensitivity / 50.0
        if icr is not None:
            if not np.isfinite(icr) or float(icr) <= 0.0:
                raise ValueError("icr must be finite and positive")
            self.carb_factor = float(icr)
        if basal_rate is not None:
            if not np.isfinite(basal_rate) or float(basal_rate) < 0.0:
                raise ValueError("basal_rate must be finite and non-negative")
            self.basal_insulin_rate = float(basal_rate)
        if dia_minutes is not None:
            if not np.isfinite(dia_minutes) or float(dia_minutes) <= 0.0:
                raise ValueError("dia_minutes must be finite and positive")
            self.insulin_action_duration = float(dia_minutes)

    def describe_compartments(self) -> Dict[str, Any]:
        """Return the compartment and flux schema this model integrates.

        The schema names each state's unit, whether it is a physical content or a
        dimensionless action variable, and whether it comes from the published
        equations or is one of this project's extensions. Consumers that draw a
        compartment diagram must read the schema from the model rather than
        assume it, because the three patient backends have different state
        vectors.
        """

        return compartment_schema("hovorka")

    def get_compartment_state(self) -> Dict[str, float]:
        """Return the current ODE state keyed by compartment name."""

        return {
            item.key: float(self._state[item.state_index])
            for item in HOVORKA_COMPARTMENTS
        }

    def flux_snapshot(
        self,
        insulin_rate_mu_per_min: Optional[float] = None,
        glucagon_rate_pg_per_min: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """Return the instantaneous transfer rates at the current state.

        This evaluates the ODE once more at the state the integrator finished
        on, so the values come from the same expressions that produced the
        trajectory instead of a second implementation that could drift from it.
        The result is an instantaneous rate at one instant, not a mass
        transferred over the preceding step.

        The delivery rates and clock default to the ones the last ``update``
        actually integrated with. Callers therefore never repeat the unit
        conversion from delivered units to mU/min, which would be a second place
        for it to be wrong.
        """

        insulin_rate, glucagon_rate = self._last_input_rates
        clock = self._last_ode_time if current_time is None else float(current_time)
        record: Dict[str, float] = {}
        self._ode(
            0.0,
            self._state,
            insulin_rate if insulin_rate_mu_per_min is None else float(insulin_rate_mu_per_min),
            glucagon_rate if glucagon_rate_pg_per_min is None else float(glucagon_rate_pg_per_min),
            clock,
            record=record,
        )
        return record

    def get_state(self) -> Dict[str, Any]:
        return {
            "state_schema": "hovorka_iints_v3_19",
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
            "last_unsupported_event": getattr(self, "_last_unsupported_event", None),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        loaded_ode_state = False
        if "ode_state" in state:
            ode_state = np.array(state["ode_state"], dtype=np.float64)
            self._state = self._coerce_legacy_ode_state(ode_state)
            if state.get("state_schema") != "hovorka_iints_v3_19":
                # v2 used D1/D2/D3 as a three-stage heuristic. Preserve total
                # undelivered meal mass while migrating to the published
                # Hovorka D1/D2 chain.
                self._state[9] += self._state[10]
                self._state[10] = 0.0
            glucagon_k2 = max(self.params.k_e_glucagon, 1e-9)
            clearance_ml_min = (
                self.params.glucagon_clearance_ml_kg_min
                * self.params.body_weight_kg
            )
            self._state[15] = (
                glucagon_k2 * self._state[14] / max(clearance_ml_min, 1e-9)
            )
            if not np.all(np.isfinite(self._state)):
                raise ValueError("Hovorka ODE snapshot contains non-finite values")
            if np.any(self._state < 0.0):
                raise ValueError("Hovorka ODE snapshot contains a negative compartment")
            for index in (11, 12, 17, 18):
                if self._state[index] > 1.0:
                    raise ValueError(
                        "Hovorka ODE snapshot contains a fraction state above 1"
                    )
            loaded_ode_state = True

        if loaded_ode_state:
            restored_glucose = float(self._state[0] / self._glucose_volume_dl())
            supplied_glucose = state.get("current_glucose")
            if supplied_glucose is not None and not np.isclose(
                float(supplied_glucose), restored_glucose, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    "Hovorka snapshot is inconsistent: current_glucose does not "
                    "match accessible glucose mass Q1"
                )
            self.current_glucose = restored_glucose
        else:
            self.current_glucose = validated_snapshot_scalar(
                state.get("current_glucose", self.current_glucose),
                name="current_glucose",
                minimum=20.0,
            )
        self.insulin_on_board = validated_snapshot_scalar(
            state.get("insulin_on_board", self.insulin_on_board),
            name="insulin_on_board",
            minimum=0.0,
        )
        self.carbs_on_board = validated_snapshot_scalar(
            state.get("carbs_on_board", self.carbs_on_board),
            name="carbs_on_board",
            minimum=0.0,
        )
        self.last_delivered_insulin_units = validated_snapshot_scalar(
            state.get(
                "last_delivered_insulin_units",
                state.get("delivered_insulin", self.last_delivered_insulin_units),
            ),
            name="last_delivered_insulin_units",
            minimum=0.0,
        )
        self.last_delivered_glucagon_mg = validated_snapshot_scalar(
            state.get("last_delivered_glucagon_mg", 0.0),
            name="last_delivered_glucagon_mg",
            minimum=0.0,
        )
        self.active_insulin_doses = validated_activity_events(
            state.get("active_insulin_doses", []),
            name="active_insulin_doses",
            age_key="age",
        )
        self.active_carb_intakes = validated_activity_events(
            state.get("active_carb_intakes", []),
            name="active_carb_intakes",
            age_key="time_since_intake",
        )
        self.is_exercising = validated_snapshot_bool(
            state.get("is_exercising", False), name="is_exercising"
        )
        self.exercise_intensity = validated_snapshot_scalar(
            state.get("exercise_intensity", 0.0),
            name="exercise_intensity",
            minimum=0.0,
            maximum=1.0,
        )
        self.is_stressed = validated_snapshot_bool(
            state.get("is_stressed", False), name="is_stressed"
        )
        self.stress_intensity = validated_snapshot_scalar(
            state.get("stress_intensity", 0.0),
            name="stress_intensity",
            minimum=0.0,
            maximum=1.0,
        )
        self._last_unsupported_event = state.get("last_unsupported_event")

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
        record: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Right-hand side of the patient ODE.

        When ``record`` is supplied, the individual transfer terms are written
        into it under the flux keys of ``patient.compartments``. Recording only
        copies subexpressions that are computed anyway, so an integration with
        recording enabled is numerically identical to one without it. The
        integrator evaluates this function several times per step, so a snapshot
        taken this way is an instantaneous rate at one state, never a
        step-averaged flow.
        """

        Q1, Q2, S1, S2, I, x1, x2, x3, D1, D2, D3, H_stress, H_exercise, Y1, Y2, Gamma, x_gluc, HAAF, GLUT4_active = y
        p = self.params

        V_I = p.V_I_per_kg * p.body_weight_kg

        # The published Hovorka glucose distribution volume is fixed. Fluid
        # balance/dehydration requires a separate validated compartment and is
        # therefore not inferred from the generic stress state.
        V_G_dL = self._glucose_volume_dl()

        G = Q1 / V_G_dL

        # Published Hovorka two-compartment meal absorption model. D1 and D2
        # contain meal mass in mg; bioavailability is applied to appearance.
        meal_rate = 1.0 / max(float(p.t_max_G), 1.0)
        dD1_dt = -meal_rate * D1
        dD2_dt = meal_rate * D1 - meal_rate * D2
        dD3_dt = 0.0
        U_G = meal_rate * D2 * p.A_G  # mg/min

        # Two-depot insulin absorption with a predefined t_max profile.
        t_max_I = self._insulin_tmax_minutes()

        dS1_dt = u_insulin - S1 / t_max_I
        dS2_dt = S1 / t_max_I - S2 / t_max_I
        U_I = S2 / t_max_I  # mU/min

        # Plasma insulin
        dI_dt = U_I / V_I - p.k_e * I

        # Exogenous Glucagon Kinetics (Bi-hormonal PK/PD)
        glucagon_k1 = 1.0 / max(p.t_max_glucagon, 1.0)
        glucagon_k2 = max(p.k_e_glucagon, 1e-9)
        glucagon_clearance_ml_min = (
            p.glucagon_clearance_ml_kg_min * p.body_weight_kg
        )
        dY1_dt = u_glucagon - glucagon_k1 * Y1
        dY2_dt = glucagon_k1 * Y1 - glucagon_k2 * Y2
        glucagon_concentration = (
            glucagon_k2 * Y2 / max(glucagon_clearance_ml_min, 1e-9)
        )
        dGamma_dt = (
            glucagon_k2 * dY2_dt / max(glucagon_clearance_ml_min, 1e-9)
        )
        glucagon_activation = glucagon_concentration / (
            max(p.glucagon_ec50_pg_ml, 1e-9) + glucagon_concentration
        )
        dx_gluc_dt = p.k_a_glucagon * (
            p.S_glucagon * glucagon_activation - x_gluc
        )

        # First-order scenario stress/exercise states. They summarize effects;
        # they are not measured adrenaline, cortisol, or AMPK concentrations.
        target_stress = self.stress_intensity if self.is_stressed else 0.0
        target_exercise = self.exercise_intensity if self.is_exercising else 0.0

        dH_stress_dt = (target_stress - H_stress) / 20.0  # 20 min model time constant.
        dH_exercise_dt = (target_exercise - H_exercise) / 10.0  # 10 min model time constant.

        stress_sens_multiplier = 1.0 - 0.7 * H_stress
        stress_EGP_multiplier = 1.0 + 0.5 * H_stress
        ex_sens_multiplier = 1.0 + 2.0 * H_exercise

        # Dawn resistance scales all three insulin actions, because it enters
        # through k_b1/k_b2/k_b3 -- that is, through S_IT, S_ID and S_IE, the
        # transport, disposal and EGP-suppression sensitivities alike.
        dawn_sens_multiplier = dawn_insulin_sensitivity_multiplier(
            current_time,
            peak_resistance_fraction=self.dawn_insulin_resistance_fraction,
            start_hour=self.dawn_start_hour,
            end_hour=self.dawn_end_hour,
        )

        overall_sens = (
            stress_sens_multiplier * ex_sens_multiplier * dawn_sens_multiplier
        )

        affinity = self.molecular_affinity_scalar
        patient_sensitivity = self._basal_parameter_scale * self._clinical_sensitivity_scale
        k_b1 = p.S_IT * p.k_a1 * overall_sens * affinity * patient_sensitivity
        k_b2 = p.S_ID * p.k_a2 * overall_sens * affinity * patient_sensitivity * self.muscle_sensitivity_scalar
        k_b3 = p.S_IE * p.k_a3 * overall_sens * affinity * patient_sensitivity * self.liver_sensitivity_scalar

        # Insulin action
        dx1_dt = -p.k_a1 * x1 + k_b1 * I
        dx2_dt = -p.k_a2 * x2 + k_b2 * I
        dx3_dt = -p.k_a3 * x3 + k_b3 * I

        # Heuristic exercise-uptake state inspired by insulin-independent
        # skeletal-muscle glucose uptake; not a molecular GLUT4 assay model.
        k_glut4_activation = 0.05
        k_glut4_deactivation = 0.01
        dGLUT4_active_dt = k_glut4_activation * H_exercise * (1.0 - GLUT4_active) - k_glut4_deactivation * GLUT4_active

        # Phenomenological dawn perturbation. The public setting has one
        # consistent unit (mg/dL/hour) across all patient backends.
        dawn_rate = dawn_glucose_rate_mgdl_min(
            current_time,
            peak_strength_mgdl_per_hour=self.dawn_phenomenon_strength,
            start_hour=self.dawn_start_hour,
            end_hour=self.dawn_end_hour,
        )
        dawn_flux = dawn_rate * V_G_dL

        F_01 = p.F_01c_per_kg * p.body_weight_kg
        # Canonical Hovorka low-glucose branch: insulin-independent uptake
        # decreases proportionally below 4.5 mmol/L (approximately 81 mg/dL).
        F_01c = F_01 * min(1.0, max(0.0, G / 81.0))

        # Endogenous Rescue & HAAF
        # When G < 70, body naturally spikes EGP to survive (Adrenaline/Glucagon burst).
        # But HAAF blunts this response.
        rescue_multiplier = counterregulatory_rescue_multiplier(G, HAAF)
        dHAAF_dt = antecedent_hypoglycemia_memory_derivative(G, HAAF)

        EGP_0 = (
            p.EGP_0_per_kg
            * p.body_weight_kg
            * stress_EGP_multiplier
            * rescue_multiplier
        )

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
            + dawn_flux
        )
        dQ2_dt = x1 * Q1 - (p.k_12 + x2) * Q2

        if record is not None:
            record.update({
                "insulin_infusion": float(u_insulin),
                "insulin_depot_transfer": float(S1 / t_max_I),
                "insulin_appearance": float(U_I),
                "insulin_elimination": float(p.k_e * I),
                "insulin_action_x1": float(dx1_dt),
                "insulin_action_x2": float(dx2_dt),
                "insulin_action_x3": float(dx3_dt),
                "meal_transfer": float(meal_rate * D1),
                "glucose_appearance": float(U_G),
                "glucose_to_periphery": float(x1 * Q1),
                "glucose_from_periphery": float(p.k_12 * Q2),
                "peripheral_disposal": float(x2 * Q2),
                "nimgu": float(NIMGU),
                "renal_clearance": float(F_R),
                "endogenous_production": float(EGP_0 * max(0.0, 1 - x3 + x_gluc)),
                "dawn_flux": float(dawn_flux),
                "glucagon_infusion": float(u_glucagon),
                "glucagon_depot_transfer": float(glucagon_k1 * Y1),
                "glucagon_appearance": float(glucagon_concentration),
                "glucagon_action": float(dx_gluc_dt),
            })

        return np.array(
            [dQ1_dt, dQ2_dt, dS1_dt, dS2_dt, dI_dt, dx1_dt, dx2_dt, dx3_dt, dD1_dt, dD2_dt, dD3_dt, dH_stress_dt, dH_exercise_dt, dY1_dt, dY2_dt, dGamma_dt, dx_gluc_dt, dHAAF_dt, dGLUT4_active_dt]
        )
