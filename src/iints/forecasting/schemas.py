"""Canonical Pydantic v2 schemas for `iints.forecasting` (IINTS Neural Forecasting v1).

Defines standardized contracts for imported CGM rows, dataset quality reports,
explicit 12-sample (55-minute elapsed span) window specifications, Model Registry
entries, and read-only research prediction records (`"clinical_use": False`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "FORECASTING_DISCLAIMER",
    "BenchmarkTier",
    "PredictionRecord",
    "SubjectQualitySummary",
    "DataQualityReport",
    "WindowSpec",
    "ModelRegistryEntry",
    "ForecastingExperimentConfig",
]

FORECASTING_DISCLAIMER = (
    "Experimental research model only (clinical_use=False). Never use predictions "
    "from iints.forecasting to program an insulin pump or guide real insulin dosing."
)

BenchmarkTier = Literal[
    "primary_ohiot1dm",
    "synthetic_software_validation",
    "personal_exploratory_only",
]


class PredictionRecord(BaseModel):
    """Standardized output record emitted by every `iints.forecasting` predictor."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    timestamp: str
    prediction_horizon_minutes: int = Field(..., ge=5, le=240)
    predicted_glucose_mgdl: float
    input_quality: Literal["complete", "locf_imputed", "degraded"] = "complete"
    source: Literal["research_model"] = "research_model"
    clinical_use: Literal[False] = False


class SubjectQualitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    source_split: Literal["train", "test"]
    raw_rows: int
    valid_rows: int
    duplicate_timestamps: int
    out_of_bounds_rows: int
    missing_value_rows: int
    gaps_over_threshold: int
    max_gap_minutes: float
    median_interval_minutes: float
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None


class DataQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    is_synthetic: bool
    benchmark_tier: BenchmarkTier
    source_provenance: str
    n_subjects: int
    total_raw_rows: int
    total_valid_rows: int
    total_duplicates_flagged: int
    total_out_of_bounds_flagged: int
    total_missing_values_flagged: int
    total_gaps_over_threshold: int
    subjects: List[SubjectQualitySummary]


class WindowSpec(BaseModel):
    """Exact sampling convention for univariate CGM windowing.

    With ``n_input_samples=12`` and ``sample_interval_minutes=5.0``:
    - Samples are ``[G_{t-55}, G_{t-50}, ..., G_{t-5}, G_t]`` (12 five-minute bins,
      commonly referred to as 60 minutes of 5-minute CGM history).
    - The exact elapsed time between the first sample ``t-55`` and the origin
      sample ``t`` is ``elapsed_span_minutes = (12 - 1) * 5.0 = 55.0 minutes``.
    - The prediction target is ``G_{t + prediction_horizon_minutes}`` (e.g. ``t + 30 min``,
      6 five-minute intervals strictly after ``t``).
    """

    model_config = ConfigDict(extra="forbid")

    n_input_samples: int = Field(default=12, ge=3, le=144)
    history_minutes: int = Field(default=60, ge=15, le=720)
    prediction_horizon_minutes: int = Field(default=30, ge=5, le=180)
    sample_interval_minutes: float = Field(default=5.0, gt=0.5, le=15.0)
    step_tolerance_minutes: float = Field(default=1.5, gt=0.1, le=4.0)
    target_tolerance_minutes: float = Field(default=2.5, gt=0.1, le=5.0)
    validation_fraction: float = Field(default=0.20, gt=0.05, lt=0.50)

    @model_validator(mode="after")
    def _sync_samples_and_history(self) -> "WindowSpec":
        implied = int(round(self.history_minutes / self.sample_interval_minutes))
        # If caller explicitly changed history_minutes while leaving default n_input_samples=12, sync it
        if self.n_input_samples == 12 and implied != 12:
            object.__setattr__(self, "n_input_samples", implied)
        elif implied != self.n_input_samples:
            raise ValueError(
                f"Inconsistent WindowSpec: n_input_samples={self.n_input_samples} at "
                f"sample_interval_minutes={self.sample_interval_minutes} implies "
                f"history_minutes={int(round(self.n_input_samples * self.sample_interval_minutes))}, "
                f"got {self.history_minutes}."
            )
        return self

    @property
    def input_length(self) -> int:
        return int(self.n_input_samples)

    @property
    def elapsed_span_minutes(self) -> float:
        """Exact elapsed time between ``X[:, 0]`` (``t - 55m``) and ``X[:, -1]`` (``t``)."""
        return float((self.n_input_samples - 1) * self.sample_interval_minutes)


class ModelRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    version: str = "1.0.0"
    architecture: Literal["persistence", "ridge", "gru", "tcn"]
    dataset_id: str
    is_synthetic: bool
    benchmark_tier: BenchmarkTier
    dataset_sha256: str
    input_length: int
    elapsed_input_span_minutes: float
    features: List[str] = Field(default_factory=lambda: ["glucose"])
    prediction_horizon_minutes: int = 30
    created_utc: str
    training_config: Dict[str, Any]
    split_counts: Dict[str, int]
    checkpoint_file: str
    checkpoint_sha256: str
    validation_metrics: Dict[str, float]
    test_metrics_available: bool = False
    clinical_validation: Literal["none"] = "none"
    disclaimer: str = FORECASTING_DISCLAIMER


class ForecastingExperimentConfig(BaseModel):
    """Declarative YAML configuration matching the IINTS Neural Forecasting v1 spec."""

    model_config = ConfigDict(extra="forbid")

    project: str = "iints-neural-forecasting"
    version: int = 1
    dataset: Literal["ohiot1dm", "iints_virtual", "carelink"] = "ohiot1dm"
    n_input_samples: int = 12
    history_minutes: int = 60
    prediction_horizon_minutes: int = 30
    sample_interval_minutes: float = 5.0
    validation_fraction: float = 0.20
    features: List[Literal["glucose"]] = Field(default_factory=lambda: ["glucose"])
    ridge_alphas: List[float] = Field(default_factory=lambda: [0.1, 1.0, 10.0, 50.0, 200.0, 1000.0])
    primary_metric: Literal["mae"] = "mae"
    secondary_metrics: List[str] = Field(
        default_factory=lambda: ["rmse", "median_absolute_error", "clarke_ab_pct"]
    )
