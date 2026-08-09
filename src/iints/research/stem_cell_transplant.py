from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from iints.core.patient.bergman_model import BergmanPatientModel


TransplantPlacement = Literal["portal", "subcutaneous", "encapsulated"]


@dataclass(frozen=True)
class PlacementPreset:
    """Tissue-site preset for a research stem-cell/islet graft simulation."""

    initial_vascularization: float
    initial_oxygenation: float
    initial_inflammation: float
    immune_visibility: float
    direct_plasma_fraction: float
    insulin_release_delay_minutes: float
    fibrosis_sensitivity: float


PLACEMENT_PRESETS: dict[TransplantPlacement, PlacementPreset] = {
    "portal": PlacementPreset(
        initial_vascularization=0.55,
        initial_oxygenation=0.70,
        initial_inflammation=0.75,
        immune_visibility=1.0,
        direct_plasma_fraction=0.85,
        insulin_release_delay_minutes=5.0,
        fibrosis_sensitivity=0.4,
    ),
    "subcutaneous": PlacementPreset(
        initial_vascularization=0.15,
        initial_oxygenation=0.35,
        initial_inflammation=0.35,
        immune_visibility=0.75,
        direct_plasma_fraction=0.0,
        insulin_release_delay_minutes=45.0,
        fibrosis_sensitivity=0.7,
    ),
    "encapsulated": PlacementPreset(
        initial_vascularization=0.10,
        initial_oxygenation=0.30,
        initial_inflammation=0.25,
        immune_visibility=0.25,
        direct_plasma_fraction=0.0,
        insulin_release_delay_minutes=75.0,
        fibrosis_sensitivity=1.4,
    ),
}


@dataclass(frozen=True)
class StemCellTransplantParameters:
    """
    Deterministic research parameters for a simplified transplant graft.

    The model is intended for algorithm stress testing and education. It is not
    a patient-specific transplant forecast and must not be used for clinical
    decision-making.
    """

    placement: TransplantPlacement = "portal"
    initial_cell_mass: float = 1.0
    initial_maturation_fraction: float = 0.30
    immunosuppression_effect: float = 0.0
    encapsulation_effect: float = 0.0
    maturation_rate_per_min: float = 1.0 / (14.0 * 24.0 * 60.0)
    vascularization_rate_per_min: float = 1.0 / (10.0 * 24.0 * 60.0)
    oxygen_time_constant_minutes: float = 180.0
    innate_decay_per_min: float = 1.0 / (3.0 * 24.0 * 60.0)
    adaptive_activation_rate_per_min: float = 2.0e-5
    adaptive_decay_per_min: float = 1.0 / (21.0 * 24.0 * 60.0)
    fibrosis_rate_per_min: float = 6.0e-6
    fibrosis_resolution_per_min: float = 1.0 / (90.0 * 24.0 * 60.0)
    hypoxia_death_rate_per_min: float = 2.5e-4
    inflammatory_death_rate_per_min: float = 1.5e-4
    adaptive_death_rate_per_min: float = 1.0e-4
    dedifferentiation_rate_per_min: float = 1.0e-4
    glucose_threshold_mgdl: float = 90.0
    glucose_slope_mgdl: float = 25.0
    max_secretion_units_per_min: float = 0.006

    def __post_init__(self) -> None:
        if self.placement not in PLACEMENT_PRESETS:
            raise ValueError(f"Unknown transplant placement: {self.placement}")
        numeric = {
            name: float(value)
            for name, value in asdict(self).items()
            if name != "placement"
        }
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("transplant parameters must all be finite")
        if numeric["initial_cell_mass"] < 0.0:
            raise ValueError("initial_cell_mass must be non-negative")
        for name in (
            "initial_maturation_fraction",
            "immunosuppression_effect",
            "encapsulation_effect",
        ):
            if not 0.0 <= numeric[name] <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        non_negative = (
            "maturation_rate_per_min",
            "vascularization_rate_per_min",
            "innate_decay_per_min",
            "adaptive_activation_rate_per_min",
            "adaptive_decay_per_min",
            "fibrosis_rate_per_min",
            "fibrosis_resolution_per_min",
            "hypoxia_death_rate_per_min",
            "inflammatory_death_rate_per_min",
            "adaptive_death_rate_per_min",
            "dedifferentiation_rate_per_min",
            "max_secretion_units_per_min",
        )
        for name in non_negative:
            if numeric[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "oxygen_time_constant_minutes",
            "glucose_threshold_mgdl",
            "glucose_slope_mgdl",
        ):
            if numeric[name] <= 0.0:
                raise ValueError(f"{name} must be positive")

    def with_placement_defaults(self) -> "StemCellTransplantParameters":
        if self.placement == "encapsulated" and self.encapsulation_effect == 0.0:
            return replace(self, encapsulation_effect=0.80)
        return self


@dataclass(frozen=True)
class StemCellTransplantState:
    immature_mass: float
    functional_mass: float
    vascularization: float
    oxygenation: float
    innate_inflammation: float
    adaptive_immunity: float
    fibrosis: float
    insulin_delay_pool_units: float
    time_minutes: float = 0.0

    def __post_init__(self) -> None:
        numeric = {name: float(value) for name, value in asdict(self).items()}
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("transplant state must contain only finite values")
        for name in (
            "immature_mass",
            "functional_mass",
            "insulin_delay_pool_units",
            "time_minutes",
        ):
            if numeric[name] < 0.0:
                raise ValueError(f"transplant state {name} must be non-negative")
        for name in (
            "vascularization",
            "oxygenation",
            "innate_inflammation",
            "adaptive_immunity",
            "fibrosis",
        ):
            if not 0.0 <= numeric[name] <= 1.0:
                raise ValueError(f"transplant state {name} must be between 0 and 1")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class StemCellTransplantStep:
    state: StemCellTransplantState
    glucose_stimulus: float
    oxygen_stress: float
    secreted_insulin_units: float
    released_insulin_units: float
    direct_plasma_units: float
    subcutaneous_units: float
    effective_beta_mass: float

    def to_dict(self) -> dict[str, float]:
        payload = self.state.to_dict()
        payload.update(
            {
                "glucose_stimulus": float(self.glucose_stimulus),
                "oxygen_stress": float(self.oxygen_stress),
                "secreted_insulin_units": float(self.secreted_insulin_units),
                "released_insulin_units": float(self.released_insulin_units),
                "direct_plasma_units": float(self.direct_plasma_units),
                "subcutaneous_units": float(self.subcutaneous_units),
                "effective_beta_mass": float(self.effective_beta_mass),
            }
        )
        return payload


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _validated_fraction_step(value: float, *, name: str) -> float:
    """Project solver-scale roundoff only; reject material state overshoot."""

    numeric = float(value)
    if not np.isfinite(numeric):
        raise RuntimeError(f"transplant step produced non-finite {name}")
    if numeric < -1e-9 or numeric > 1.0 + 1e-9:
        raise RuntimeError(
            f"transplant step moved {name} outside [0, 1]: {numeric}"
        )
    return _clip01(numeric)


def _validated_nonnegative_step(value: float, *, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise RuntimeError(f"transplant step produced non-finite {name}")
    if numeric < -1e-9:
        raise RuntimeError(f"transplant step produced negative {name}: {numeric}")
    return max(0.0, numeric)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = float(np.exp(-value))
        return float(1.0 / (1.0 + z))
    z = float(np.exp(value))
    return float(z / (1.0 + z))


class StemCellTransplantModel:
    """Multi-compartment graft model for IINTS research simulations."""

    def __init__(
        self,
        parameters: StemCellTransplantParameters | None = None,
        state: StemCellTransplantState | None = None,
    ) -> None:
        self.parameters = (parameters or StemCellTransplantParameters()).with_placement_defaults()
        self.placement = PLACEMENT_PRESETS[self.parameters.placement]
        self.state = state or self.initial_state(self.parameters)

    @staticmethod
    def initial_state(parameters: StemCellTransplantParameters) -> StemCellTransplantState:
        params = parameters.with_placement_defaults()
        preset = PLACEMENT_PRESETS[params.placement]
        initial_mass = float(params.initial_cell_mass)
        maturation = float(params.initial_maturation_fraction)
        return StemCellTransplantState(
            immature_mass=initial_mass * (1.0 - maturation),
            functional_mass=initial_mass * maturation,
            vascularization=_clip01(preset.initial_vascularization),
            oxygenation=_clip01(preset.initial_oxygenation),
            innate_inflammation=_clip01(preset.initial_inflammation),
            adaptive_immunity=0.0,
            fibrosis=0.0,
            insulin_delay_pool_units=0.0,
            time_minutes=0.0,
        )

    def step(self, glucose_mgdl: float, dt_minutes: float) -> StemCellTransplantStep:
        params = self.parameters
        preset = self.placement
        state = self.state
        glucose = float(glucose_mgdl)
        dt = float(dt_minutes)
        if not np.isfinite(glucose) or not 20.0 <= glucose <= 600.0:
            raise ValueError("glucose_mgdl must be finite and inside [20, 600]")
        if not np.isfinite(dt) or not 0.0 < dt <= 60.0:
            raise ValueError("dt_minutes must be finite and inside (0, 60]")

        immunosuppression = params.immunosuppression_effect
        encapsulation = params.encapsulation_effect
        immune_visibility = preset.immune_visibility * (1.0 - 0.75 * encapsulation)

        inflammation = state.innate_inflammation
        adaptive = state.adaptive_immunity
        fibrosis = state.fibrosis
        vascularization = state.vascularization
        oxygenation = state.oxygenation
        immature = state.immature_mass
        functional = state.functional_mass

        d_vascularization = (
            params.vascularization_rate_per_min
            * (1.0 - vascularization)
            * (1.0 - 0.75 * fibrosis)
            * (1.0 - 0.35 * inflammation)
        )
        vascularization_next = _validated_fraction_step(
            vascularization + dt * d_vascularization,
            name="vascularization",
        )

        oxygen_target = _clip01(
            0.10
            + 0.90 * vascularization_next
            - 0.35 * fibrosis
            - 0.10 * max(0.0, immature + functional - 1.0)
        )
        oxygenation_next = _validated_fraction_step(
            oxygenation
            + dt * (oxygen_target - oxygenation)
            / params.oxygen_time_constant_minutes,
            name="oxygenation",
        )
        oxygen_stress = max(0.0, 0.45 - oxygenation_next)

        inflammation_next = _validated_fraction_step(
            inflammation - dt * params.innate_decay_per_min * inflammation,
            name="innate_inflammation",
        )
        adaptive_drive = (
            params.adaptive_activation_rate_per_min
            * immune_visibility
            * max(0.0, immature + functional)
            * (1.0 - immunosuppression)
        )
        adaptive_decay = params.adaptive_decay_per_min * adaptive * (0.3 + immunosuppression)
        adaptive_next = _validated_fraction_step(
            adaptive + dt * (adaptive_drive - adaptive_decay),
            name="adaptive_immunity",
        )

        fibrosis_drive = (
            params.fibrosis_rate_per_min
            * preset.fibrosis_sensitivity
            * (inflammation_next + adaptive_next)
            * (0.35 + encapsulation)
        )
        fibrosis_next = _validated_fraction_step(
            fibrosis
            + dt
            * (fibrosis_drive - params.fibrosis_resolution_per_min * fibrosis),
            name="fibrosis",
        )

        death_rate = (
            params.hypoxia_death_rate_per_min * oxygen_stress
            + params.inflammatory_death_rate_per_min * inflammation_next * (1.0 - 0.65 * immunosuppression)
            + params.adaptive_death_rate_per_min * adaptive_next * (1.0 - 0.80 * immunosuppression)
        )
        maturation_rate = params.maturation_rate_per_min * oxygenation_next * (1.0 - 0.50 * inflammation_next)
        dediff_rate = params.dedifferentiation_rate_per_min * oxygen_stress

        matured = min(immature, maturation_rate * immature * dt)
        immature_next = _validated_nonnegative_step(
            immature - matured - death_rate * immature * dt,
            name="immature_mass",
        )
        functional_next = _validated_nonnegative_step(
            functional + matured - death_rate * functional * dt - dediff_rate * functional * dt,
            name="functional_mass",
        )

        glucose_stimulus = _sigmoid(
            (glucose - params.glucose_threshold_mgdl)
            / params.glucose_slope_mgdl
        )
        function_factor = oxygenation_next * (1.0 - 0.55 * fibrosis_next)
        effective_beta_mass = _validated_nonnegative_step(
            functional_next * function_factor,
            name="effective_beta_mass",
        )
        secreted_units = _validated_nonnegative_step(
            params.max_secretion_units_per_min
            * effective_beta_mass
            * glucose_stimulus
            * dt,
            name="secreted_insulin_units",
        )

        pool = state.insulin_delay_pool_units + secreted_units
        release_fraction = dt / (dt + preset.insulin_release_delay_minutes)
        released_units = pool * release_fraction
        pool_next = _validated_nonnegative_step(
            pool - released_units,
            name="insulin_delay_pool_units",
        )

        direct_units = released_units * preset.direct_plasma_fraction
        subq_units = released_units * (1.0 - preset.direct_plasma_fraction)

        self.state = StemCellTransplantState(
            immature_mass=immature_next,
            functional_mass=functional_next,
            vascularization=vascularization_next,
            oxygenation=oxygenation_next,
            innate_inflammation=inflammation_next,
            adaptive_immunity=adaptive_next,
            fibrosis=fibrosis_next,
            insulin_delay_pool_units=pool_next,
            time_minutes=state.time_minutes + dt,
        )
        return StemCellTransplantStep(
            state=self.state,
            glucose_stimulus=glucose_stimulus,
            oxygen_stress=oxygen_stress,
            secreted_insulin_units=secreted_units,
            released_insulin_units=released_units,
            direct_plasma_units=direct_units,
            subcutaneous_units=subq_units,
            effective_beta_mass=effective_beta_mass,
        )


def _normalize_meal_schedule(meal_schedule: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    meals: dict[int, float] = {}
    for meal in meal_schedule:
        minute = meal.get("start_time", meal.get("time", meal.get("minute")))
        carbs = meal.get("value", meal.get("carbs", meal.get("carb_grams")))
        if minute is None or carbs is None:
            raise ValueError("Meal entries must include a time/start_time/minute and carbs/value.")
        minute_key = int(minute)
        meals[minute_key] = meals.get(minute_key, 0.0) + float(carbs)
    return meals


def _add_direct_plasma_insulin(patient: BergmanPatientModel, insulin_units: float) -> None:
    if insulin_units <= 0.0:
        return
    vi_abs_l = patient.params.Vi * patient.params.body_weight_kg
    patient._state[2] += (float(insulin_units) * 1000.0) / max(vi_abs_l, 0.001)


def run_stem_cell_transplant_simulation(
    *,
    duration_minutes: int = 1440,
    time_step_minutes: int = 5,
    initial_glucose: float = 120.0,
    parameters: StemCellTransplantParameters | None = None,
    meal_schedule: Sequence[Mapping[str, Any]] = (),
    basal_insulin_units_per_hour: float = 0.0,
) -> pd.DataFrame:
    """
    Couple the multi-compartment graft model to a Bergman virtual patient.

    The coupling is intentionally transparent: portal release is added to the
    plasma insulin state, while delayed/subcutaneous release goes through the
    normal two-depot insulin absorption pathway.
    """
    dt = int(time_step_minutes)
    duration = int(duration_minutes)
    if dt <= 0 or dt > 60 or dt != time_step_minutes:
        raise ValueError("time_step_minutes must be an integer inside [1, 60]")
    if duration < 0 or duration != duration_minutes:
        raise ValueError("duration_minutes must be a non-negative integer")
    initial = float(initial_glucose)
    basal_rate = float(basal_insulin_units_per_hour)
    if not np.isfinite(initial) or not 20.0 <= initial <= 600.0:
        raise ValueError("initial_glucose must be finite and inside [20, 600]")
    if not np.isfinite(basal_rate) or basal_rate < 0.0:
        raise ValueError(
            "basal_insulin_units_per_hour must be finite and non-negative"
        )

    patient = BergmanPatientModel(
        initial_glucose=initial,
        basal_insulin_rate=basal_rate,
    )
    graft = StemCellTransplantModel(parameters=parameters)
    meals = _normalize_meal_schedule(meal_schedule)
    records: list[dict[str, float | str]] = []

    for minute in range(0, duration + 1, dt):
        carb_intake = meals.get(minute, 0.0)
        graft_step = graft.step(patient.current_glucose, dt)
        _add_direct_plasma_insulin(patient, graft_step.direct_plasma_units)
        basal_units = basal_rate * dt / 60.0
        delivered_subq = basal_units + graft_step.subcutaneous_units
        glucose = patient.update(
            time_step=float(dt),
            delivered_insulin=delivered_subq,
            carb_intake=carb_intake,
            current_time=float(minute),
        )
        patient_state = patient.get_patient_state()
        row: dict[str, float | str] = {
            "time_minutes": float(minute),
            "glucose_mgdl": float(glucose),
            "carbs_grams": float(carb_intake),
            "placement": graft.parameters.placement,
            "plasma_insulin_mU_L": float(patient_state["plasma_insulin_mU_L"]),
            "subcutaneous_insulin_units": float(delivered_subq),
        }
        row.update(graft_step.to_dict())
        records.append(row)

    return pd.DataFrame.from_records(records)
