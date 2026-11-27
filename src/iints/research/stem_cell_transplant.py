from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from iints.research.provenance import warn_unsourced
import pandas as pd

from iints.core.patient.bergman_model import BergmanPatientModel
from iints.research.c_peptide import CPeptideModel, CPeptideParameters


TransplantPlacement = Literal["portal", "subcutaneous", "encapsulated"]

# Islet grafts are dosed in islet equivalents (IEQ, normalised to a 150 um
# islet). To give the dimensionless graft mass a unit, this is the IEQ/kg that
# counts as "one reference beta-cell mass". It follows from the order-of-
# magnitude estimate of ~1e6 islet equivalents in an adult pancreas; it is a
# stated convention, not a measured constant, and it is flagged in
# docs/DEFECT_REGISTER_NL.md so it can be replaced with a sourced value.
REFERENCE_BETA_MASS_IEQ_PER_KG = 14000.0

# Named function presets. The difference between them is a first-phase
# deficit, which is the most consistently reported functional difference
# between stem-cell-derived and primary islets. The stage-6 value is an
# explicit assumption to be calibrated against published C-peptide curves, not
# a measurement.
ISLET_FUNCTION_PRESETS: dict[str, dict[str, float]] = {
    "primary_islet": {"first_phase_fraction": 1.0},
    "sc_islet_stage6": {"first_phase_fraction": 0.25},
}


def mass_fraction_from_islet_equivalents(
    islet_equivalents_per_kg: float,
    *,
    reference_ieq_per_kg: float = REFERENCE_BETA_MASS_IEQ_PER_KG,
) -> float:
    """Convert a clinical IEQ/kg dose into the model's graft-mass fraction."""
    if islet_equivalents_per_kg < 0.0:
        raise ValueError("islet_equivalents_per_kg must be non-negative")
    if reference_ieq_per_kg <= 0.0:
        raise ValueError("reference_ieq_per_kg must be positive")
    if reference_ieq_per_kg == REFERENCE_BETA_MASS_IEQ_PER_KG:
        warn_unsourced("REFERENCE_BETA_MASS_IEQ_PER_KG")
    return float(islet_equivalents_per_kg) / float(reference_ieq_per_kg)


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
    # The costs of immunosuppression. Both default to zero, so existing runs
    # are unchanged, but with them at zero a run that sets
    # immunosuppression_effect represents only the benefit. Calcineurin and
    # mTOR inhibitors are reported to be directly toxic to beta cells and to
    # impair their secretory response; these two parameters are the places to
    # express that once a source fixes their size.
    immunosuppression_beta_toxicity_per_min: float = 0.0
    immunosuppression_secretory_impairment: float = 0.0
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

    # --- secretion scale (G1) -------------------------------------------
    # The graft mass is a fraction of one reference beta-cell mass (see
    # REFERENCE_BETA_MASS_IEQ_PER_KG). What turns that fraction into insulin is
    # the pair below. `basal_secretion_u_per_kg_day` is the basal secretion a
    # full reference mass produces at the glucose threshold, where the sigmoid
    # sits at one half; the saturating ceiling is therefore twice that rate.
    # Reported basal secretion in adults without diabetes is roughly
    # 0.2-0.4 U/kg/day out of a 0.4-0.8 U/kg/day total, so 0.30 is mid-range.
    # Pass `max_secretion_units_per_min` to override the scale directly; the
    # historical default was 0.006 U/min, which is about five times too low
    # for a reference mass and left the mass unit undefined.
    body_weight_kg: float = 70.0
    basal_secretion_u_per_kg_day: float = 0.30
    max_secretion_units_per_min: float | None = None

    # --- first phase (G2) -----------------------------------------------
    # Glucose-stimulated secretion is biphasic: a readily releasable pool
    # empties within minutes of a rising glucose, then a sustained second
    # phase follows the glucose level. `first_phase_fraction` scales the
    # dynamic component only, so it separates a first-phase deficit from
    # overall graft mass or placement delay.
    first_phase_fraction: float = 1.0
    first_phase_capacity_u_per_kg: float = 0.03
    readily_releasable_refill_minutes: float = 25.0
    potentiation_gain_per_mgdl: float = 0.02

    def __post_init__(self) -> None:
        if self.placement not in PLACEMENT_PRESETS:
            raise ValueError(f"Unknown transplant placement: {self.placement}")
        numeric = {
            name: float(value)
            for name, value in asdict(self).items()
            if name != "placement" and value is not None
        }
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("transplant parameters must all be finite")
        if numeric["initial_cell_mass"] < 0.0:
            raise ValueError("initial_cell_mass must be non-negative")
        for name in (
            "initial_maturation_fraction",
            "immunosuppression_effect",
            "immunosuppression_secretory_impairment",
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
            "basal_secretion_u_per_kg_day",
            "first_phase_capacity_u_per_kg",
            "potentiation_gain_per_mgdl",
            "immunosuppression_beta_toxicity_per_min",
        )
        for name in non_negative:
            if numeric[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_secretion_units_per_min is not None:
            if float(self.max_secretion_units_per_min) < 0.0:
                raise ValueError("max_secretion_units_per_min must be non-negative")
        if not 0.0 <= float(self.first_phase_fraction) <= 1.0:
            raise ValueError("first_phase_fraction must be between 0 and 1")
        for name in ("body_weight_kg", "readily_releasable_refill_minutes"):
            if numeric[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
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

    def with_islet_function(self, preset: str) -> "StemCellTransplantParameters":
        """Apply a named function preset from ``ISLET_FUNCTION_PRESETS``."""
        if preset not in ISLET_FUNCTION_PRESETS:
            raise ValueError(
                f"Unknown islet function preset: {preset}. "
                f"Known: {sorted(ISLET_FUNCTION_PRESETS)}")
        return replace(self, **ISLET_FUNCTION_PRESETS[preset])

    @property
    def effective_max_secretion_units_per_min(self) -> float:
        """Saturating secretion rate of one reference beta-cell mass (U/min)."""
        if self.max_secretion_units_per_min is not None:
            return float(self.max_secretion_units_per_min)
        basal_u_per_min = (float(self.basal_secretion_u_per_kg_day)
                           * float(self.body_weight_kg) / 1440.0)
        return 2.0 * basal_u_per_min

    @property
    def readily_releasable_capacity_units(self) -> float:
        """Insulin held in the readily releasable pool at a full reference mass."""
        return (float(self.first_phase_capacity_u_per_kg)
                * float(self.body_weight_kg))


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
    readily_releasable_units: float = 0.0
    time_minutes: float = 0.0

    def __post_init__(self) -> None:
        numeric = {name: float(value) for name, value in asdict(self).items()}
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("transplant state must contain only finite values")
        for name in (
            "immature_mass",
            "functional_mass",
            "insulin_delay_pool_units",
            "readily_releasable_units",
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
    first_phase_units: float = 0.0
    second_phase_units: float = 0.0
    glucose_rate_mgdl_per_min: float = 0.0
    c_peptide_pmol_l: float = 0.0

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
                "first_phase_units": float(self.first_phase_units),
                "second_phase_units": float(self.second_phase_units),
                "glucose_rate_mgdl_per_min": float(self.glucose_rate_mgdl_per_min),
                "c_peptide_pmol_l": float(self.c_peptide_pmol_l),
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
        c_peptide: CPeptideModel | None = None,
    ) -> None:
        self.parameters = (parameters or StemCellTransplantParameters()).with_placement_defaults()
        self.placement = PLACEMENT_PRESETS[self.parameters.placement]
        self.state = state or self.initial_state(self.parameters)
        # C-peptide is co-secreted equimolar with insulin, so the same release
        # drives it. It exists here to make graft function comparable with the
        # endpoint clinical studies report (G3).
        self.c_peptide = c_peptide or CPeptideModel(
            CPeptideParameters(body_weight_kg=self.parameters.body_weight_kg))
        self._previous_glucose_mgdl: float | None = None
        self._warn_about_unsourced_constants()

    def _warn_about_unsourced_constants(self) -> None:
        """Say out loud which numbers in this run are guesses.

        Each constant warns once per process. Relative comparisons at a fixed
        parameter set are unaffected by these values; absolute insulin,
        C-peptide and IEQ/kg figures are not.
        """
        params = self.parameters
        defaults = StemCellTransplantParameters()
        if params.potentiation_gain_per_mgdl == defaults.potentiation_gain_per_mgdl:
            warn_unsourced("potentiation_gain_per_mgdl")
        stage6 = ISLET_FUNCTION_PRESETS["sc_islet_stage6"]["first_phase_fraction"]
        if float(params.first_phase_fraction) == stage6:
            warn_unsourced(
                "ISLET_FUNCTION_PRESETS['sc_islet_stage6'].first_phase_fraction")
        if (params.immunosuppression_effect > 0.0
                and params.immunosuppression_beta_toxicity_per_min == 0.0
                and params.immunosuppression_secretory_impairment == 0.0):
            warn_unsourced("immunosuppression_effect")

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
            readily_releasable_units=(
                params.readily_releasable_capacity_units * initial_mass * maturation),
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
            # The cost side: the drug that protects the graft also damages it.
            # Zero by default, so this term only appears once a user sets it.
            + params.immunosuppression_beta_toxicity_per_min * immunosuppression
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
        function_factor = (
            oxygenation_next
            * (1.0 - 0.55 * fibrosis_next)
            # Impaired secretion under immunosuppression; zero by default.
            * (1.0 - params.immunosuppression_secretory_impairment * immunosuppression)
        )
        effective_beta_mass = _validated_nonnegative_step(
            functional_next * function_factor,
            name="effective_beta_mass",
        )

        # Second phase: sustained, follows the glucose level.
        second_phase_units = _validated_nonnegative_step(
            params.effective_max_secretion_units_per_min
            * effective_beta_mass
            * glucose_stimulus
            * dt,
            name="second_phase_units",
        )

        # First phase (G2): a readily releasable pool that empties on a rising
        # glucose and refills with its own time constant. Release is
        # proportional to the pool content, so the burst is self-limiting and
        # exhausts - which is what a blunted first phase has to be able to
        # express. `first_phase_fraction` scales only this component.
        if self._previous_glucose_mgdl is None:
            glucose_rate = 0.0
        else:
            glucose_rate = (glucose - self._previous_glucose_mgdl) / dt
        pool_capacity = (params.readily_releasable_capacity_units
                         * effective_beta_mass)
        refilled = state.readily_releasable_units + dt * (
            (pool_capacity - state.readily_releasable_units)
            / params.readily_releasable_refill_minutes)
        refilled = max(0.0, refilled)
        first_phase_demand = (params.potentiation_gain_per_mgdl
                              * max(0.0, glucose_rate)
                              * refilled
                              * params.first_phase_fraction
                              * dt)
        first_phase_units = min(refilled, max(0.0, first_phase_demand))
        readily_releasable_next = _validated_nonnegative_step(
            refilled - first_phase_units,
            name="readily_releasable_units",
        )
        self._previous_glucose_mgdl = glucose

        secreted_units = _validated_nonnegative_step(
            second_phase_units + first_phase_units,
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
            readily_releasable_units=readily_releasable_next,
            time_minutes=state.time_minutes + dt,
        )
        # The released amount is what reaches the circulation, so it is also
        # what drives C-peptide; the site delay therefore applies to both.
        c_peptide_pmol_l = self.c_peptide.step(released_units, dt)
        return StemCellTransplantStep(
            state=self.state,
            glucose_stimulus=glucose_stimulus,
            oxygen_stress=oxygen_stress,
            secreted_insulin_units=secreted_units,
            released_insulin_units=released_units,
            direct_plasma_units=direct_units,
            subcutaneous_units=subq_units,
            effective_beta_mass=effective_beta_mass,
            first_phase_units=first_phase_units,
            second_phase_units=second_phase_units,
            glucose_rate_mgdl_per_min=glucose_rate,
            c_peptide_pmol_l=c_peptide_pmol_l,
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
    """Deprecated: adds insulin to the plasma state as an impulse.

    Kept so existing scripts keep working. New code should pass
    ``endogenous_plasma_insulin_units`` to ``BergmanPatientModel.update``,
    which applies the same insulin as a rate inside the derivative and is
    therefore step-size independent (defect G5).
    """
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
    max_coupling_step_minutes: float | None = 1.0,
) -> pd.DataFrame:
    """
    Couple the multi-compartment graft model to a Bergman virtual patient.

    The coupling is intentionally transparent: directly appearing (portal)
    release enters the plasma insulin derivative as a rate, while delayed or
    subcutaneous release goes through the normal two-depot absorption pathway.

    ``time_step_minutes`` is the reporting interval: one row per interval
    ``[time_minutes, interval_end_minutes)``. ``max_coupling_step_minutes``
    is the integration interval of the coupled system, which is what the
    accuracy depends on, because the graft reads glucose once per integration
    step. Keeping it at one minute makes results at different reporting
    intervals comparable, which is what an experiment protocol needs; pass
    ``None`` to integrate one step per reported row instead, which is faster
    and step-size dependent.

    Per row, insulin flows (secreted, released, direct, subcutaneous, first and
    second phase) are summed over the interval, while levels and graft states
    are taken at the end of the interval.
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
    coupling_step = (None if max_coupling_step_minutes is None
                     else float(max_coupling_step_minutes))
    if coupling_step is not None and (not np.isfinite(coupling_step)
                                      or coupling_step <= 0.0):
        raise ValueError(
            "max_coupling_step_minutes must be finite and positive when given")

    patient = BergmanPatientModel(
        initial_glucose=initial,
        basal_insulin_rate=basal_rate,
    )
    graft = StemCellTransplantModel(parameters=parameters)
    meals = _normalize_meal_schedule(meal_schedule)
    records: list[dict[str, float | str]] = []

    # G4: one row per integrated interval [time_minutes, interval_end_minutes).
    # The loop used to run to `duration + 1`, so the integrated horizon was
    # `duration + time_step` and two runs at different step sizes covered
    # different amounts of time - the same defect as D2 in the simulator.
    flow_fields = (
        "secreted_insulin_units",
        "released_insulin_units",
        "direct_plasma_units",
        "subcutaneous_units",
        "first_phase_units",
        "second_phase_units",
    )
    for minute in range(0, duration, dt):
        step_minutes = float(min(dt, duration - minute))
        carb_intake = meals.get(minute, 0.0)
        # G6: integrate at `coupling_step` or finer, report once per interval.
        if coupling_step is None:
            substeps = 1
        else:
            substeps = max(1, int(np.ceil(step_minutes / coupling_step - 1e-9)))
        inner_minutes = step_minutes / substeps
        delivered_subq = 0.0
        flows = {name: 0.0 for name in flow_fields}
        for index in range(substeps):
            graft_step = graft.step(patient.current_glucose, inner_minutes)
            basal_units = basal_rate * inner_minutes / 60.0
            subcutaneous_units = basal_units + graft_step.subcutaneous_units
            # G5: directly appearing insulin goes in as a rate through the ODE,
            # not as a jump in the plasma state.
            glucose = patient.update(
                time_step=inner_minutes,
                delivered_insulin=subcutaneous_units,
                carb_intake=carb_intake if index == 0 else 0.0,
                current_time=float(minute) + index * inner_minutes,
                endogenous_plasma_insulin_units=graft_step.direct_plasma_units,
            )
            delivered_subq += subcutaneous_units
            step_values = graft_step.to_dict()
            for name in flow_fields:
                flows[name] += float(step_values[name])
        patient_state = patient.get_patient_state()
        row: dict[str, float | str] = dict(graft_step.to_dict())
        row.update(flows)
        # G7: the graft keeps its own clock, and a state field of the same name
        # used to overwrite the interval column silently. The authoritative
        # columns are written last; the graft clock keeps its own name.
        row["graft_time_minutes"] = float(row.pop("time_minutes", 0.0))
        row.update({
            "time_minutes": float(minute),
            "interval_end_minutes": float(minute) + step_minutes,
            "glucose_mgdl": float(glucose),
            "carbs_grams": float(carb_intake),
            "placement": graft.parameters.placement,
            "plasma_insulin_mU_L": float(patient_state["plasma_insulin_mU_L"]),
            "subcutaneous_insulin_units": float(delivered_subq),
        })
        records.append(row)

    return pd.DataFrame.from_records(records)
