from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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
    weight_decay: float = 0.0
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    # P0-2: Subject-level split fractions (must sum to <= 1.0).
    # The remaining fraction after val + test goes to training.
    # Set subject_level_split=True (default) to split by subject ID to
    # prevent data leakage between train/val/test sets.
    subject_level_split: bool = True
    validation_split: float = 0.15   # fraction of subjects for validation
    test_split: float = 0.15         # fraction of subjects for held-out test

    seed: int = 42

    # P3-10: Normalization strategy.  Options: "zscore", "robust", "none".
    normalization: str = "zscore"

    # P3-12: Loss function.  Options: "mse", "quantile", "safety_weighted", "band_weighted".
    # For quantile loss, also set `quantile` (0 < q < 1).
    loss: str = "mse"
    quantile: Optional[float] = None  # e.g. 0.9 for 90th-percentile upper bound

    # Fine-tuning controls
    freeze_lstm_layers: int = 0          # freeze first N LSTM layers
    early_stopping_patience: int = 0     # 0 disables early stopping
    early_stopping_min_delta: float = 0.0

    # Safety-weighted loss (penalize low-glucose errors more strongly)
    safety_weighted_low_threshold: float = 80.0
    safety_weighted_alpha: float = 2.0
    safety_weighted_max_weight: float = 4.0

    # Band-weighted loss (penalize low/high glucose errors)
    band_weighted_low_threshold: float = 70.0
    band_weighted_high_threshold: float = 180.0
    band_weighted_low_weight: float = 2.0
    band_weighted_high_weight: float = 1.5
    band_weighted_max_weight: float = 5.0

    # Optional: pre-announced meals (clinically common for AID systems).
    # If enabled, create `meal_announcement_feature` by shifting
    # `meal_announcement_column` earlier by `meal_announcement_minutes`.
    meal_announcement_minutes: Optional[float] = None
    meal_announcement_column: str = "carb_intake_grams"
    meal_announcement_feature: str = "meal_announcement_grams"
