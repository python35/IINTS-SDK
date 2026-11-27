"""Declarative Pydantic v2 contracts for IINTS Simulation SDK 2.0.

Defines reproducible, version-auditable schemas for virtual patients, meal/stress
scenarios, sensor fault models (including blank vs stale-value display behavior),
forecasting/control algorithms, and pre-registered missing-data evaluation rules.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

__all__ = [
    "RESEARCH_ONLY_DISCLAIMER",
    "MealEventSpec",
    "PatientSpec",
    "ScenarioSpec",
    "SensorFaultSpec",
    "AlgorithmSpec",
    "MissingDataContract",
    "EvaluationSpec",
    "ExperimentSpec",
    "load_experiment_spec",
    "dump_experiment_yaml",
]

RESEARCH_ONLY_DISCLAIMER = (
    "Pre-clinical in-silico research and education only. Neither IINTS Simulation SDK 2.0 "
    "nor any bundled patient model has clinical or regulatory clearance, and no output "
    "may be used for real-world insulin dosing."
)


class MealEventSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minute: float = Field(..., ge=0.0, description="Simulation minute of meal intake.")
    carbs_g: float = Field(..., ge=0.0, le=300.0, description="True physical carbohydrates ingested (g).")
    announced_carbs_g: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=300.0,
        description="Carbohydrates entered into the pump/forecaster (None -> defaults to carbs_g).",
    )
    bolus_delay_min: float = Field(default=0.0, ge=-60.0, le=120.0)


class PatientSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["hovorka_equilibrium", "iints_ode", "bergman", "simglucose"] = "hovorka_equilibrium"
    profile: str = "virtual_adult_01"
    split: Literal["dev", "unseen"] = "dev"
    initial_glucose_mgdl: float = Field(default=135.0, ge=40.0, le=350.0)
    basal_u_per_h: float = Field(default=1.6, ge=0.1, le=8.0)
    sigma: Optional[float] = Field(
        default=None,
        gt=0.01,
        le=1.5,
        description="Single-knob Hovorka sensitivity; None derives fasting equilibrium at initial_glucose.",
    )
    insulin_sensitivity_mgdl_per_u: float = Field(default=45.0, gt=5.0, le=250.0)
    carb_factor_g_per_u: float = Field(default=10.0, gt=2.0, le=60.0)
    meal_tau_min: float = Field(default=50.0, gt=10.0, le=150.0)
    bioavailability: float = Field(default=0.80, gt=0.2, le=1.0)
    max_glucose_rate_mgdl_per_min: float = Field(default=8.0, gt=0.5, le=20.0)


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = "standard_day"
    duration_hours: float = Field(default=24.0, gt=1.0, le=336.0)
    step_minutes: float = Field(default=5.0, gt=0.5, le=15.0)
    warmup_hours: float = Field(
        default=6.0,
        ge=0.0,
        le=72.0,
        description="Initial simulation/training warm-up hours before primary evaluation window.",
    )
    meal_schedule: Literal[
        "standard_day",
        "high_carb_stress",
        "late_fat_protein",
        "nocturnal_descent",
        "custom",
    ] = "standard_day"
    meals: List[MealEventSpec] = Field(default_factory=list)
    meal_time_jitter_min: float = Field(default=15.0, ge=0.0, le=90.0)
    carb_announcement_error_sd: float = Field(default=0.10, ge=0.0, le=0.8)
    bolus_relative_error_sd: float = Field(default=0.10, ge=0.0, le=0.8)
    unannounced_meal_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    regime: Literal["in_distribution", "ood_stress"] = "in_distribution"


class SensorFaultSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "cgm_v1"
    condition_id: str = "clean_reference"
    isf_tau_minutes: float = Field(default=6.0, gt=0.5, le=30.0)
    lag_minutes: int = Field(default=0, ge=0, le=25)
    noise_sd_mgdl: float = Field(default=6.0, ge=0.0, le=40.0)
    noise_ar1_phi: float = Field(default=0.75, ge=0.0, lt=0.995)
    bias_mgdl: float = Field(default=0.0, ge=-50.0, le=50.0)
    drift_mgdl_per_day: float = Field(default=0.0, ge=-40.0, le=40.0)
    dropout_minutes: int = Field(default=0, ge=0, le=240)
    dropout_at_hour: float = Field(default=8.5, ge=0.0, le=336.0)
    recurring_dropout_every_hours: Optional[float] = Field(default=None, gt=0.5, le=72.0)
    random_dropout_prob: float = Field(default=0.0, ge=0.0, le=0.8)
    compression_low_at_hour: Optional[float] = Field(default=None, ge=0.0, le=336.0)
    compression_low_minutes: int = Field(default=0, ge=0, le=120)
    compression_low_drop_mgdl: float = Field(default=25.0, ge=0.0, le=80.0)
    comm_delay_steps: int = Field(default=0, ge=0, le=12)
    display_mode: Literal["nan_blank", "stale_hold"] = Field(
        default="nan_blank",
        description=(
            "'nan_blank' emits NaN during dropout; 'stale_hold' continues displaying "
            "the last outdated CGM value (for UI/human-factors hazard benchmarking)."
        ),
    )


class AlgorithmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["glucose_forecaster"] = "glucose_forecaster"
    model_id: str = "persistence_v1"
    params: Dict[str, Any] = Field(default_factory=dict)


class MissingDataContract(BaseModel):
    """Pre-registered rules governing what forecasters may observe and how evaluation works."""

    model_config = ConfigDict(extra="forbid")

    allow_last_valid_observation: bool = True
    expose_stale_flag_to_algorithm: bool = True
    expose_insulin_and_announced_carbs: bool = True
    impute_before_algorithm: Literal["none", "ffill"] = "none"
    reference_target: Literal["true_plasma_glucose", "clean_cgm"] = "true_plasma_glucose"
    history_window_minutes: int = Field(default=180, ge=30, le=1440)


class EvaluationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_horizon_minutes: int = Field(default=30, ge=5, le=240)
    extra_horizons_minutes: List[int] = Field(default_factory=list)
    recovery_window_minutes: int = Field(default=30, ge=5, le=180)
    metrics: List[str] = Field(
        default_factory=lambda: [
            "mae",
            "rmse",
            "dropout_mae",
            "dropout_recovery_mae",
            "clarke_ab_pct",
            "coverage_90_pct",
        ]
    )
    missing_data_contract: MissingDataContract = Field(default_factory=MissingDataContract)

    @field_validator("metrics")
    @classmethod
    def _non_empty_metrics(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("evaluation.metrics must contain at least one metric")
        return value

    def all_horizons(self) -> List[int]:
        ordered: List[int] = []
        for h in [self.prediction_horizon_minutes, *self.extra_horizons_minutes]:
            if int(h) not in ordered:
                ordered.append(int(h))
        return ordered


class ExperimentSpec(BaseModel):
    """Complete single- or multi-algorithm reproducible SDK 2.0 experiment specification."""

    model_config = ConfigDict(extra="forbid")

    id: str = "cgm_dropout_001"
    seed: int = 42
    patient: PatientSpec = Field(default_factory=PatientSpec)
    scenario: ScenarioSpec = Field(default_factory=ScenarioSpec)
    sensor: SensorFaultSpec = Field(default_factory=SensorFaultSpec)
    algorithm: Optional[AlgorithmSpec] = None
    algorithms: List[AlgorithmSpec] = Field(default_factory=list)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)

    def resolved_algorithms(self) -> List[AlgorithmSpec]:
        items: List[AlgorithmSpec] = []
        if self.algorithm is not None:
            items.append(self.algorithm)
        items.extend(self.algorithms)
        if not items:
            items = [
                AlgorithmSpec(model_id="persistence_v1"),
                AlgorithmSpec(model_id="ar2_v1"),
                AlgorithmSpec(model_id="ridge_ml_v1"),
                AlgorithmSpec(model_id="hovorka_graybox_v2"),
            ]
        seen = set()
        unique: List[AlgorithmSpec] = []
        for spec in items:
            if spec.model_id not in seen:
                seen.add(spec.model_id)
                unique.append(spec)
        return unique


def load_experiment_spec(source: Union[str, Path, Dict[str, Any]]) -> ExperimentSpec:
    """Load an ``ExperimentSpec`` from YAML/JSON path or nested ``{'experiment': ...}`` dict."""
    if isinstance(source, (str, Path)):
        raw = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    else:
        raw = dict(source)
    if "experiment" in raw and isinstance(raw["experiment"], dict):
        merged = dict(raw["experiment"])
        for key in ("patient", "scenario", "sensor", "algorithm", "algorithms", "evaluation"):
            if key in raw and key not in merged:
                merged[key] = raw[key]
        raw = merged
    return ExperimentSpec.model_validate(raw)


def dump_experiment_yaml(spec: ExperimentSpec) -> str:
    payload = {"experiment": spec.model_dump(mode="json", exclude_none=True)}
    return yaml.safe_dump(payload, sort_keys=False)
