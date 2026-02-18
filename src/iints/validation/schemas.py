from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

LATEST_SCHEMA_VERSION = "1.1"

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class StressEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_time: int = Field(ge=0)
    event_type: Literal["meal", "missed_meal", "sensor_error", "exercise", "exercise_end", "ratio_change"]
    value: Optional[float] = None
    reported_value: Optional[float] = None
    absorption_delay_minutes: int = Field(default=0, ge=0)
    duration: int = Field(default=0, ge=0)
    isf: Optional[float] = Field(default=None, gt=0)
    icr: Optional[float] = Field(default=None, gt=0)
    basal_rate: Optional[float] = Field(default=None, ge=0)
    dia_minutes: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_required_fields(self) -> "StressEventModel":
        if self.event_type in {"meal", "missed_meal"}:
            if self.value is None or self.value <= 0:
                raise ValueError("meal value must be > 0 grams")
        if self.event_type == "exercise":
            if self.value is None or not (0.0 <= self.value <= 1.0):
                raise ValueError("exercise value must be between 0.0 and 1.0")
        if self.event_type == "sensor_error":
            if self.value is None:
                raise ValueError("sensor_error requires a value")
        if self.event_type == "ratio_change":
            if all(
                val is None
                for val in (self.isf, self.icr, self.basal_rate, self.dia_minutes)
            ):
                raise ValueError("ratio_change requires at least one ratio value (isf/icr/basal_rate/dia_minutes)")
        return self


class ScenarioModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_name: str = Field(min_length=1)
    schema_version: str = Field(default=LATEST_SCHEMA_VERSION, min_length=1)
    scenario_version: str = Field(min_length=1)
    description: Optional[str] = None
    stress_events: List[StressEventModel] = Field(default_factory=list)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _normalize_schema_version(cls, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return LATEST_SCHEMA_VERSION
        return str(value)

    @field_validator("scenario_version", mode="before")
    @classmethod
    def _normalize_version(cls, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            raise ValueError("scenario_version is required")
        return str(value)


class PatientConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basal_insulin_rate: float = Field(default=0.8, ge=0.0, le=3.0)
    insulin_sensitivity: float = Field(default=50.0, ge=10.0, le=200.0)
    carb_factor: float = Field(default=10.0, ge=3.0, le=30.0)
    glucose_decay_rate: float = Field(default=0.05, ge=0.0, le=0.2)
    initial_glucose: float = Field(default=120.0, ge=40.0, le=400.0)
    glucose_absorption_rate: float = Field(default=0.03, ge=0.0, le=0.2)
    insulin_action_duration: float = Field(default=300.0, ge=60.0, le=720.0)
    insulin_peak_time: float = Field(default=75.0, ge=15.0, le=240.0)
    meal_mismatch_epsilon: float = Field(default=1.0, ge=0.5, le=1.5)
    dawn_phenomenon_strength: float = Field(default=0.0, ge=0.0, le=50.0)
    dawn_start_hour: float = Field(default=4.0, ge=0.0, le=23.0)
    dawn_end_hour: float = Field(default=8.0, ge=0.0, le=24.0)

    @model_validator(mode="after")
    def _check_peak_vs_duration(self) -> "PatientConfigModel":
        if self.insulin_peak_time >= self.insulin_action_duration:
            raise ValueError("insulin_peak_time must be less than insulin_action_duration")
        if self.dawn_end_hour <= self.dawn_start_hour:
            raise ValueError("dawn_end_hour must be greater than dawn_start_hour")
        return self
