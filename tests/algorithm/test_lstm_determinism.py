from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from iints.api.base_algorithm import AlgorithmInput
from iints.core.algorithms.lstm_algorithm import LSTMInsulinAlgorithm, LSTMModel


def _write_constant_checkpoint(path: Path) -> None:
    model = LSTMModel(input_size=7, hidden_size=4, output_size=1, dropout_prob=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc.bias.fill_(0.4)
    torch.save(model.state_dict(), path)


def _input(**overrides: float) -> AlgorithmInput:
    values = {
        "current_glucose": 180.0,
        "time_step": 5.0,
        "insulin_on_board": 1.0,
        "carb_intake": 0.0,
        "glucose_trend_mgdl_min": 0.2,
        "predicted_glucose_30min": 175.0,
        "isf": 50.0,
        "icr": 10.0,
    }
    values.update(overrides)
    return AlgorithmInput(**values)


def test_missing_checkpoint_fails_closed_to_deterministic_fallback(tmp_path: Path) -> None:
    algorithm = LSTMInsulinAlgorithm(settings={"model_path": str(tmp_path / "missing.pt")})

    result = algorithm.predict_insulin(_input())

    assert result["fallback_triggered"] is True
    assert result["fallback_reason"] == "missing_model_checkpoint"
    assert result["model_checkpoint_loaded"] is False


def test_loaded_lstm_candidate_is_repeatable_and_hard_capped(tmp_path: Path) -> None:
    checkpoint = tmp_path / "constant.pt"
    _write_constant_checkpoint(checkpoint)
    algorithm = LSTMInsulinAlgorithm(
        settings={
            "model_path": str(checkpoint),
            "hidden_size": 4,
            "mc_samples": 8,
            "uncertainty_threshold": 1.0,
            "mc_seed": 123,
            "max_model_candidate_units": 0.3,
        }
    )

    first = algorithm.predict_insulin(_input())
    second = algorithm.predict_insulin(_input())

    assert first["total_insulin_delivered"] == pytest.approx(0.3)
    assert second["total_insulin_delivered"] == pytest.approx(0.3)
    assert first["uncertainty"] == second["uncertainty"]
    assert first["uncertainty_seed"] == 123


def test_loaded_lstm_fixed_low_guard_returns_zero(tmp_path: Path) -> None:
    checkpoint = tmp_path / "constant.pt"
    _write_constant_checkpoint(checkpoint)
    algorithm = LSTMInsulinAlgorithm(
        settings={"model_path": str(checkpoint), "hidden_size": 4}
    )

    result = algorithm.predict_insulin(
        _input(current_glucose=88.0, predicted_glucose_30min=75.0)
    )

    assert result["total_insulin_delivered"] == 0.0
    assert result["fallback_reason"] == "fixed_controller_safety_guard"
