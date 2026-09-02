"""
Bergman Minimal Model — IINTS-AF
==================================
ODE-based patient model inspired by the Bergman Minimal Model with an
adapted gut absorption chain for delayed carbohydrate appearance.

It is a more mechanistic research option than ``CustomPatientModel``, but the
extensions and parameterization are not independently clinically validated.
It uses ``scipy.integrate.solve_ivp`` and has a higher computational cost.

The model tracks 14 state variables:

* **G** — plasma glucose concentration (mg/dL)
* **X** — remote insulin action (1/min)
* **I** — plasma insulin concentration (mU/L)
* **Q_stomach** — stomach glucose mass waiting for gastric emptying (mg)
* **Q_gut** — intestinal glucose mass available for absorption (mg)
* **S1/S2** — SubQ insulin pools
* **Y1/Y2/Gamma/x_gluc** — Glucagon subQ & plasma kinetics
* **HAAF** — Hypoglycemia-Associated Autonomic Failure memory
* **M_graft** — optional stem-cell/islet graft survival fraction

References
----------
* Bergman, R. N. et al. (1979). Quantitative estimation of insulin
  sensitivity. *Am J Physiol*, 236(6), E667–E677.
* Dalla Man, C. et al. (2007). Meal Simulation Model of the Glucose-
  Insulin System. *IEEE Trans Biomed Eng*, 54(10), 1740–1749.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .compartments import BERGMAN_COMPARTMENTS, compartment_schema
from .models import PatientModelDomainError
from .physiology import (
    antecedent_hypoglycemia_memory_derivative,
    counterregulatory_rescue_multiplier,
    dawn_glucose_rate_mgdl_min,
    glucagon_mg_to_pg,
    smooth_threshold_excess,
    validated_activity_events,
    validated_snapshot_bool,
    validated_snapshot_scalar,
)


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
    # Literature-informed glucagon PK medians/ranges from Wendt et al. The
    # first absorption rate is represented as 1 / t_max_glucagon for backward
    # configuration compatibility.
    t_max_glucagon: float = 25.0  # min; k1 = 0.04 1/min
    k_e_glucagon: float = 0.165   # 1/min; second transfer/elimination rate k2
    glucagon_clearance_ml_kg_min: float = 120.0  # apparent clearance
    glucagon_ec50_pg_ml: float = 350.0
    k_a_glucagon: float = 0.05    # 1/min; effect-compartment rate
    S_glucagon: float = 1.0       # maximum fractional EGP increase

    # --- Stem Cell Graft (Research) ---
    # Abstract functional graft fraction; 100% is not a clinical cure claim.
    stem_cell_engraftment_percent: float = 0.0
    stem_cell_subq_fraction: float = 0.0       # 0.0 = PV (immediate), 1.0 = subQ (delayed via S1)
    immune_rejection_rate: float = 0.0         # 1/min decay of graft mass

    # --- Gut absorption ---
    tau_meal: float = 40.0  # min    — gastric emptying time constant
    k_abs: float = 0.05     # 1/min  — intestinal absorption rate constant
    f_bio: float = 0.90     # —      — bioavailability (fraction absorbed)

    # --- Patient physical ---
    body_weight_kg: float = 70.0

    def __post_init__(self) -> None:
        numeric = {
            name: float(value)
            for name, value in vars(self).items()
        }
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("Bergman parameters must all be finite")
        positive = (
            "p1", "p2", "p3", "Gb", "Vg", "n", "Vi", "k_a",
            "t_max_glucagon", "k_e_glucagon",
            "glucagon_clearance_ml_kg_min", "glucagon_ec50_pg_ml",
            "k_a_glucagon", "tau_meal", "k_abs", "body_weight_kg",
        )
        for name in positive:
            if numeric[name] <= 0.0:
                raise ValueError(f"Bergman parameter {name} must be positive")
        non_negative = (
            "Ib", "gamma", "h", "S_glucagon", "immune_rejection_rate",
        )
        for name in non_negative:
            if numeric[name] < 0.0:
                raise ValueError(
                    f"Bergman parameter {name} must be non-negative"
                )
        if not 0.0 <= numeric["f_bio"] <= 1.0:
            raise ValueError("Bergman parameter f_bio must be between 0 and 1")
        if not 0.0 <= numeric["stem_cell_engraftment_percent"] <= 100.0:
            raise ValueError(
                "stem_cell_engraftment_percent must be between 0 and 100"
            )
        if not 0.0 <= numeric["stem_cell_subq_fraction"] <= 1.0:
            raise ValueError("stem_cell_subq_fraction must be between 0 and 1")


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
        stem_cell_engraftment_percent: float = 0.0,
        stem_cell_subq_fraction: float = 0.0,
        immune_rejection_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Bergman model arguments: {names}")
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
            "dawn_start_hour": float(dawn_start_hour),
            "dawn_end_hour": float(dawn_end_hour),
            "carb_absorption_duration_minutes": float(carb_absorption_duration_minutes),
            "max_glucose_rate_mgdl_per_min": float(max_glucose_rate_mgdl_per_min),
            "stem_cell_engraftment_percent": float(stem_cell_engraftment_percent),
            "stem_cell_subq_fraction": float(stem_cell_subq_fraction),
            "immune_rejection_rate": float(immune_rejection_rate),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Bergman model inputs must all be finite")
        positive = (
            "insulin_sensitivity", "carb_factor", "initial_glucose",
            "glucose_absorption_rate",
            "insulin_action_duration", "insulin_peak_time",
            "meal_mismatch_epsilon", "carb_absorption_duration_minutes",
        )
        for name in positive:
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if values["basal_insulin_rate"] < 0.0:
            raise ValueError("basal_insulin_rate must be non-negative")
        if values["glucose_decay_rate"] < 0.0:
            raise ValueError("glucose_decay_rate must be non-negative")
        if values["dawn_phenomenon_strength"] < 0.0:
            raise ValueError("dawn_phenomenon_strength must be non-negative")
        if values["max_glucose_rate_mgdl_per_min"] < 0.0:
            raise ValueError("max_glucose_rate_mgdl_per_min must be non-negative")
        if not 0.0 <= values["dawn_start_hour"] < values["dawn_end_hour"] <= 24.0:
            raise ValueError(
                "dawn hours must satisfy 0 <= start < end <= 24"
            )
        if not 0.0 <= values["stem_cell_engraftment_percent"] <= 100.0:
            raise ValueError(
                "stem_cell_engraftment_percent must be between 0 and 100"
            )
        if not 0.0 <= values["stem_cell_subq_fraction"] <= 1.0:
            raise ValueError("stem_cell_subq_fraction must be between 0 and 1")
        if values["immune_rejection_rate"] < 0.0:
            raise ValueError("immune_rejection_rate must be non-negative")
        if basal_glucose_target is not None:
            target = float(basal_glucose_target)
            if not np.isfinite(target) or target < 20.0:
                raise ValueError(
                    "basal_glucose_target must be finite and at least 20 mg/dL"
                )

        # Store clinical knobs (for ratio queries and compatibility)
        self.basal_insulin_rate = values["basal_insulin_rate"]
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
        self.dawn_start_hour = values["dawn_start_hour"]
        self.dawn_end_hour = values["dawn_end_hour"]
        self.carb_absorption_duration_minutes = values["carb_absorption_duration_minutes"]
        self.max_glucose_rate_mgdl_per_min = values["max_glucose_rate_mgdl_per_min"]

        # Bergman ODE parameters
        gb_default = (
            float(initial_glucose)
            if basal_glucose_target is None
            else float(basal_glucose_target)
        )
        if bergman_params is not None:
            if any(
                value != 0.0
                for value in (
                    stem_cell_engraftment_percent,
                    stem_cell_subq_fraction,
                    immune_rejection_rate,
                )
            ):
                raise ValueError(
                    "Pass stem-cell parameters either through bergman_params "
                    "or constructor arguments, not both"
                )
            self.params = bergman_params
        else:
            self.params = BergmanParameters(
                p1=0.028 * max(float(glucose_decay_rate), 1e-6) / 0.05,
                p3=5.0e-6 * max(float(insulin_sensitivity), 1e-6) / 50.0,
                Gb=gb_default,
                k_a=1.0 / max(float(insulin_peak_time), 1.0),
                tau_meal=max(45.0, min(float(carb_absorption_duration_minutes) / 3.0, 100.0)),
                k_abs=float(glucose_absorption_rate),
                stem_cell_engraftment_percent=stem_cell_engraftment_percent,
                stem_cell_subq_fraction=stem_cell_subq_fraction,
                immune_rejection_rate=immune_rejection_rate,
                gamma=0.005 if stem_cell_engraftment_percent > 0 else 0.0,
            )
        self._p3_per_isf = self.params.p3 / max(float(insulin_sensitivity), 1e-6)
        self._refresh_basal_steady_state()

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
        self._last_unsupported_event: Optional[Dict[str, Any]] = None
        self.meal_effect_delay = 30  # kept for API compat

        # ODE state vector:
        # [G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF, M_graft]
        self._state = np.array([
            initial_glucose,       # 0: G  (mg/dL)
            0.0,                   # 1: X  (1/min)
            self._reference_insulin_mU_L,  # 2: I  (mU/L)
            0.0,                   # 3: Q_sto1 (mg) - Solid Stomach
            0.0,                   # 4: Q_sto2 (mg) - Liquid Stomach
            0.0,                   # 5: Q_gut (mg)  - Intestine
            self._basal_depot_mU,  # 6: S1 (mU)
            self._basal_depot_mU,  # 7: S2 (mU)
            0.0,                   # 8: Y1 (pg) - Glucagon subQ 1
            0.0,                   # 9: Y2 (pg) - Glucagon subQ 2
            0.0,                   # 10: Gamma (pg/mL) - Plasma Glucagon
            0.0,                   # 11: x_gluc (1) - Glucagon action on EGP
            0.0,                   # 12: HAAF (1) - Memory
            (self.params.stem_cell_engraftment_percent / 100.0),  # 13: M_graft (1) - Graft Mass
        ], dtype=np.float64)

        self.reset()

    def _refresh_basal_steady_state(self) -> None:
        """Derive pump-supported fasting insulin states in consistent units."""

        p = self.params
        self._basal_input_mU_per_min = (
            max(float(self.basal_insulin_rate), 0.0) * 1000.0 / 60.0
        )
        insulin_volume_l = p.Vi * p.body_weight_kg
        if self._basal_input_mU_per_min > 0.0:
            self._reference_insulin_mU_L = self._basal_input_mU_per_min / max(
                insulin_volume_l * p.n, 1e-9
            )
        else:
            self._reference_insulin_mU_L = max(float(p.Ib), 0.0)
        self._basal_depot_mU = self._basal_input_mU_per_min / max(p.k_a, 1e-9)

    def _guard_glucose_transition(self, proposed_glucose: float, time_step: float) -> float:
        """Validate solver output without altering physiological mass balance."""
        if not np.isfinite(proposed_glucose):
            raise PatientModelDomainError(
                "Bergman ODE produced non-finite glucose",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        if proposed_glucose < 0.0:
            raise PatientModelDomainError(
                f"Bergman ODE produced negative glucose: {proposed_glucose}",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        max_rate = float(getattr(self, "max_glucose_rate_mgdl_per_min", 0.0) or 0.0)
        elapsed = max(float(time_step), 1e-9)
        rate = abs(float(proposed_glucose) - float(self.current_glucose)) / elapsed
        if max_rate > 0.0 and rate > max_rate + 1e-9:
            raise PatientModelDomainError(
                f"Bergman ODE glucose rate {rate:.3f} mg/dL/min exceeds "
                f"configured validation limit {max_rate:.3f}",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        return float(proposed_glucose)

    def _bounded_fraction_state_indices(self) -> tuple[int, ...]:
        """Return state indices whose interpretation is limited to [0, 1]."""

        return (12, 13)  # antecedent-hypoglycemia memory, graft mass fraction

    # ------------------------------------------------------------------
    # Public interface (mirrors CustomPatientModel exactly)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to initial conditions."""
        self._last_input_rates = (0.0, 0.0)
        self._last_ode_time = 0.0
        self._refresh_basal_steady_state()
        self._state = np.array([
            self.initial_glucose,
            0.0,
            self._reference_insulin_mU_L,
            0.0,
            0.0,
            0.0,
            self._basal_depot_mU,
            self._basal_depot_mU,
            0.0, 0.0, 0.0, 0.0, 0.0, (self.params.stem_cell_engraftment_percent / 100.0)
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
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Bergman update arguments: {names}")
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
        glucagon_rate = glucagon_mg_to_pg(delivered_glucagon_mg) / max(time_step, 0.001)

        # --- Solve ODE ---
        ct = current_time if current_time is not None else 0.0
        # Kept so flux_snapshot reports the rates and clock this step actually
        # integrated with, rather than repeating the unit conversions above at
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
                raise RuntimeError(f"Bergman ODE integration failed: {sol.message}")
            self._state = sol.y[:, -1].copy()
            self._state[0] = self._guard_glucose_transition(
                float(self._state[0]), time_step
            )
            # X is a deviation-from-basal action state and may legitimately be
            # negative when plasma insulin falls below its reference value.
            if np.any(self._state[2:] < -1e-6):
                minimum = float(np.min(self._state[2:]))
                raise RuntimeError(
                    f"Bergman ODE produced a negative compartment state: {minimum}"
                )
            for index in self._bounded_fraction_state_indices():
                value = float(self._state[index])
                if value > 1.0 + 1e-6:
                    raise RuntimeError(
                        f"Bergman ODE produced a fraction state above 1: {value}"
                    )
            # Remove only sub-micro numerical integration noise.
            for i in range(2, len(self._state)):
                self._state[i] = max(0.0, self._state[i])
            for index in self._bounded_fraction_state_indices():
                self._state[index] = min(1.0, self._state[index])

            self.current_glucose = float(self._state[0])
            return self.current_glucose
        except Exception:
            self.set_state(previous_state)
            raise

    def get_current_glucose(self) -> float:
        return self.current_glucose

    def trigger_event(self, event_type: str, value: Any) -> None:
        # Scenario events are applied by the simulator through the explicit
        # start/stop methods. Retain the last unsupported event for audit
        # visibility instead of silently claiming that it changed physiology.
        self._last_unsupported_event = {"event_type": str(event_type), "value": value}

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
            "reference_insulin_mU_L": self._reference_insulin_mU_L,
            "remote_insulin_action": float(self._state[1]),
            "stomach_glucose_mg": float(self._state[3] + self._state[4]),
            "stomach_solid_mg": float(self._state[3]),
            "stomach_liquid_mg": float(self._state[4]),
            "gut_glucose_mg": float(self._state[5]),
            "subcut_insulin_1_mU": float(self._state[6]),
            "subcut_insulin_2_mU": float(self._state[7]),
            "plasma_glucagon_pg_ml": float(self._state[10]),
            "haaf_metric": float(self._state[12]),
            "stem_cell_graft_mass_fraction": float(self._state[13]),
            "stem_cell_engraftment_percent": float(self.params.stem_cell_engraftment_percent),
            "stem_cell_subq_fraction": float(self.params.stem_cell_subq_fraction),
            "immune_rejection_rate_per_min": float(self.params.immune_rejection_rate),
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
            if not np.isfinite(isf) or float(isf) <= 0.0:
                raise ValueError("isf must be finite and positive")
            self.insulin_sensitivity = float(isf)
            self.params.p3 = self._p3_per_isf * self.insulin_sensitivity
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
        """Return the compartment schema this model integrates.

        The Bergman state vector differs from the Hovorka one -- it integrates
        glucose as a concentration and has no separate accessible/peripheral
        glucose mass pair -- so consumers must read the schema from the model
        instead of assuming a single layout across patient backends. No flux
        table is published for this backend yet, so a diagram drawn from it
        shows contents without transfer arrows.
        """

        return compartment_schema("bergman")

    def get_compartment_state(self) -> Dict[str, float]:
        """Return the current ODE state keyed by compartment name."""

        return {
            item.key: float(self._state[item.state_index])
            for item in BERGMAN_COMPARTMENTS
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
        trajectory. The result is an instantaneous rate at one instant, not a
        mass transferred over the preceding step. Glucose terms are per volume
        because this backend integrates glucose as a concentration.

        The delivery rates and clock default to the ones the last ``update``
        actually integrated with, so callers never repeat the unit conversion
        from delivered units to mU/min.
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
            "state_schema": "bergman_iints_v2_14",
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
            # Handle legacy snapshot coercions to 13-state vector
            if ode_state.size == 4:
                ode_state = np.array(
                    [ode_state[0], ode_state[1], ode_state[2], 0.0, 0.0, ode_state[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (self.params.stem_cell_engraftment_percent / 100.0)],
                    dtype=np.float64,
                )
            elif ode_state.size == 5:
                ode_state = np.array(
                    [ode_state[0], ode_state[1], ode_state[2], ode_state[3], 0.0, ode_state[4], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (self.params.stem_cell_engraftment_percent / 100.0)],
                    dtype=np.float64,
                )
            elif ode_state.size == 7:
                ode_state = np.array(
                    [ode_state[0], ode_state[1], ode_state[2], ode_state[3], 0.0, ode_state[4], ode_state[5], ode_state[6], 0.0, 0.0, 0.0, 0.0, 0.0, (self.params.stem_cell_engraftment_percent / 100.0)],
                    dtype=np.float64,
                )
            elif ode_state.size == 8:
                ode_state = np.array(
                    [ode_state[0], ode_state[1], ode_state[2], ode_state[3], ode_state[4], ode_state[5], ode_state[6], ode_state[7], 0.0, 0.0, 0.0, 0.0, 0.0, (self.params.stem_cell_engraftment_percent / 100.0)],
                    dtype=np.float64,
                )
            elif ode_state.size == 13:
                ode_state = np.array(
                    list(ode_state) + [(self.params.stem_cell_engraftment_percent / 100.0)],
                    dtype=np.float64,
                )
            if ode_state.size != 14:
                raise ValueError(
                    f"Unsupported Bergman ODE snapshot length {ode_state.size}; expected 14"
                )
            if not np.all(np.isfinite(ode_state)):
                raise ValueError("Bergman ODE snapshot contains non-finite values")
            if ode_state[0] < 0.0 or np.any(ode_state[2:] < 0.0):
                raise ValueError("Bergman ODE snapshot contains a negative mass or concentration")
            for index in self._bounded_fraction_state_indices():
                if ode_state[index] > 1.0:
                    raise ValueError(
                        "Bergman ODE snapshot contains a fraction state above 1"
                    )
            self._state = ode_state
            clearance_ml_min = (
                self.params.glucagon_clearance_ml_kg_min
                * self.params.body_weight_kg
            )
            self._state[10] = (
                self.params.k_e_glucagon * self._state[9]
                / max(clearance_ml_min, 1e-9)
            )
            loaded_ode_state = True

        if loaded_ode_state:
            restored_glucose = float(self._state[0])
            supplied_glucose = state.get("current_glucose")
            if supplied_glucose is not None and not np.isclose(
                float(supplied_glucose), restored_glucose, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    "Bergman snapshot is inconsistent: current_glucose does not "
                    "match ode_state[0]"
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
        u_insulin_mu_per_min: float,
        u_glucagon_pg_per_min: float,
        current_time: float,
        record: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Right-hand side of the extended minimal model.

        When ``record`` is given, the transfer terms already computed here are
        copied into it under the keys declared in ``BERGMAN_FLUXES``. Reporting
        reuses these values instead of recomputing them elsewhere, which would
        create a second implementation free to drift from the equations that
        actually produced the trajectory. The integration itself is unaffected.
        """

        G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF, M_graft = y
        p = self.params

        Vg_abs = p.Vg * p.body_weight_kg   # dL
        Vi_abs = p.Vi * p.body_weight_kg    # L
        glucagon_clearance_ml_min = (
            p.glucagon_clearance_ml_kg_min * p.body_weight_kg
        )

        # --- Glucose rate of appearance from gut ---
        Ra = (p.k_abs * Q_gut) / Vg_abs  # mg/dL/min

        # --- Exogenous Glucagon Kinetics (Bi-hormonal PK/PD) ---
        glucagon_k1 = 1.0 / max(p.t_max_glucagon, 1.0)
        glucagon_k2 = max(p.k_e_glucagon, 1e-9)
        dY1_dt = u_glucagon_pg_per_min - glucagon_k1 * Y1
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

        # --- Dawn phenomenon ---
        dawn = dawn_glucose_rate_mgdl_min(
            current_time,
            peak_strength_mgdl_per_hour=self.dawn_phenomenon_strength,
            start_hour=self.dawn_start_hour,
            end_hour=self.dawn_end_hour,
        )

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
        rescue_multiplier = counterregulatory_rescue_multiplier(G, HAAF)
        dHAAF_dt = antecedent_hypoglycemia_memory_derivative(G, HAAF)

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
        dXdt = -p.p2 * X + p3_eff * (I - self._reference_insulin_mU_L)

        # --- Stem Cell / Islet Secretion ---
        # Gamma acts as base secretion rate for a 100% functional pancreas
        # M_graft acts as the survival fraction multiplier.
        total_secretion_concentration = p.gamma * M_graft * max(G - p.h, 0.0)
        
        secretion_mass = total_secretion_concentration * Vi_abs
        secretion_subq = secretion_mass * p.stem_cell_subq_fraction
        secretion_plasma = secretion_mass * (1.0 - p.stem_cell_subq_fraction)

        # --- dS1/dt, dS2/dt (Subcutaneous Insulin Absorption) ---
        # Any SubQ implanted islet cells release into S1
        dS1dt = u_insulin_mu_per_min + secretion_subq - p.k_a * S1
        dS2dt = p.k_a * S1 - p.k_a * S2

        # Rate of appearance of insulin into plasma (mU/min)
        Ra_I = p.k_a * S2

        # --- dI/dt ---
        # Any PV/Hepatic implanted islet cells release directly into plasma
        dIdt = -p.n * I + (secretion_plasma + Ra_I) / Vi_abs
        
        # --- dM_graft/dt ---
        dM_graft_dt = -p.immune_rejection_rate * M_graft

        # --- Adapted three-stage meal absorption chain ---
        gastric_emptying_rate = 1.0 / max(float(p.tau_meal), 1.0)
        solid_to_liquid_rate = gastric_emptying_rate * 1.5 
        dQ_sto1_dt = -solid_to_liquid_rate * Q_sto1
        dQ_sto2_dt = solid_to_liquid_rate * Q_sto1 - gastric_emptying_rate * Q_sto2
        dQ_gut_dt = gastric_emptying_rate * Q_sto2 - p.k_abs * Q_gut

        if record is not None:
            record.update(
                {
                    "gastric_liquefaction": float(solid_to_liquid_rate * Q_sto1),
                    "gastric_emptying": float(gastric_emptying_rate * Q_sto2),
                    "glucose_appearance": float(Ra),
                    "insulin_infusion": float(u_insulin_mu_per_min),
                    "insulin_depot_transfer": float(p.k_a * S1),
                    "insulin_appearance": float(Ra_I),
                    "insulin_elimination": float(p.n * I),
                    "insulin_action": float(dXdt),
                    "glucose_uptake": float((p1_eff + X) * G),
                    "basal_production": float(p1_eff * Gb_eff),
                    "renal_clearance": float(F_R),
                    "exercise_uptake": float(exercise_glucose_uptake),
                    "dawn_flux": float(dawn),
                    "glucagon_infusion": float(u_glucagon_pg_per_min),
                    "glucagon_depot_transfer": float(glucagon_k1 * Y1),
                    "glucagon_appearance": float(glucagon_k2 * Y2 / max(glucagon_clearance_ml_min, 1e-9)),
                    "glucagon_action": float(dx_gluc_dt),
                    "islet_secretion_subq": float(secretion_subq),
                    "islet_secretion_plasma": float(secretion_plasma / Vi_abs),
                    "graft_rejection": float(p.immune_rejection_rate * M_graft),
                }
            )

        return np.array([dGdt, dXdt, dIdt, dQ_sto1_dt, dQ_sto2_dt, dQ_gut_dt, dS1dt, dS2dt, dY1_dt, dY2_dt, dGamma_dt, dx_gluc_dt, dHAAF_dt, dM_graft_dt])
