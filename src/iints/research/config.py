from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PredictorConfig:
    history_minutes: int = 240
    horizon_minutes: int = 60
    time_step_minutes: int = 5
    feature_columns: List[str] = field(default_factory=lambda: [
        "glucose_actual_mgdl",
        "patient_iob_units",
        "patient_cob_grams",
        "effective_isf",
        "effective_icr",
        "effective_basal_rate_u_per_hr",
        "glucose_trend_mgdl_min",
    ])
    target_column: str = "glucose_actual_mgdl"

    @property
    def history_steps(self) -> int:
        return int(self.history_minutes / self.time_step_minutes)

    @property
    def horizon_steps(self) -> int:
        return int(self.horizon_minutes / self.time_step_minutes)


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    validation_split: float = 0.1
    seed: int = 42
