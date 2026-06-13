from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from iints.research.config import PredictorConfig
from iints.research.jetson_hf_trainer import _normalize_downloaded_hf_model, _trial_config, model_score
from research.train_predictor import _fill_missing_training_features


def test_model_score_penalizes_physiology_and_hypo_risk() -> None:
    clean = {
        "mae": 20.0,
        "missed_hypo_rate_pct": 1.0,
        "any_physiology_violation_pct": 1.0,
    }
    risky = {
        "mae": 19.0,
        "missed_hypo_rate_pct": 8.0,
        "any_physiology_violation_pct": 25.0,
    }

    assert model_score(clean) < model_score(risky)


def test_trial_config_preserves_warm_start_architecture() -> None:
    base = {
        "predictor": {
            "history_minutes": 360,
            "horizon_minutes": 120,
            "time_step_minutes": 5,
            "feature_columns": ["glucose_actual_mgdl"],
            "target_column": "glucose_actual_mgdl",
        },
        "training": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "loss": "pinn",
        },
    }

    cfg = _trial_config(
        base,
        rng=random.Random(42),
        trial_index=1,
        seed=42,
        epochs=3,
        batch_size=32,
        min_lr=1e-5,
        max_lr=1e-4,
        min_pinn_lambda=0.1,
        max_pinn_lambda=0.2,
        weight_decay_choices=[0.0],
    )

    assert cfg["training"]["hidden_size"] == 128
    assert cfg["training"]["num_layers"] == 2
    assert cfg["training"]["epochs"] == 3
    assert cfg["training"]["batch_size"] == 32
    assert 1e-5 <= cfg["training"]["learning_rate"] <= 1e-4
    assert 0.1 <= cfg["training"]["pinn_lambda"] <= 0.2


def test_normalize_downloaded_hf_model_copies_nested_bundle(tmp_path: Path) -> None:
    nested = tmp_path / "huggingface"
    nested.mkdir()
    (nested / "predictor.pt").write_bytes(b"checkpoint")
    (nested / "glucose_model_config.yaml").write_text("predictor: {}\ntraining: {}\n")
    (nested / "training_report.json").write_text("{}\n")

    _normalize_downloaded_hf_model(tmp_path)

    assert (tmp_path / "predictor.pt").read_bytes() == b"checkpoint"
    assert (tmp_path / "glucose_model_config.yaml").is_file()
    assert (tmp_path / "training_report.json").is_file()


def test_training_fills_missing_legacy_optional_features() -> None:
    frame = pd.DataFrame({"glucose_actual_mgdl": [100.0, 101.0]})
    cfg = PredictorConfig(
        feature_columns=["glucose_actual_mgdl", "calories", "sleep_minutes"],
        target_column="glucose_actual_mgdl",
    )

    filled = _fill_missing_training_features(frame, cfg)

    assert filled["calories"].tolist() == [0.0, 0.0]
    assert filled["sleep_minutes"].tolist() == [0.0, 0.0]
