from __future__ import annotations

import numpy as np

from iints.core.algorithms.mock_algorithms import ConstantDoseAlgorithm
from iints.core.patient.models import PatientModel
from iints.core.simulator import Simulator


FEATURES = [
    "glucose_actual_mgdl",
    "patient_iob_units",
    "patient_cob_grams",
    "effective_isf",
    "effective_icr",
    "effective_basal_rate_u_per_hr",
    "glucose_trend_mgdl_min",
]


def _simulator(predictor: object) -> Simulator:
    return Simulator(
        patient_model=PatientModel(initial_glucose=120.0),
        algorithm=ConstantDoseAlgorithm(dose=0.0),
        predictor=predictor,
    )


def _row() -> dict[str, float]:
    return {feature: 0.0 for feature in FEATURES}


def test_predictor_without_scaler_metadata_is_untrusted() -> None:
    class Predictor:
        config = {"feature_columns": FEATURES, "history_steps": 1, "horizon_steps": 6}
        scaler = None

        def predict(self, x):
            return np.array([[140.0]])

    prediction, fallback, meta = _simulator(Predictor())._predict_with_model(_row(), 123.0)

    assert prediction == fallback == 123.0
    assert meta["predictor_used"] is False
    assert meta["predictor_in_distribution"] is False
    assert meta["predictor_gate_reason"] == "missing_scaler_metadata"


def test_predictor_exception_is_recorded_before_fallback() -> None:
    class Scaler:
        _center = np.zeros(len(FEATURES))
        _scale = np.ones(len(FEATURES))

    class Predictor:
        config = {"feature_columns": FEATURES, "history_steps": 1, "horizon_steps": 6}
        scaler = Scaler()

        def predict(self, x):
            raise RuntimeError("broken checkpoint")

    prediction, fallback, meta = _simulator(Predictor())._predict_with_model(_row(), 123.0)

    assert prediction == fallback == 123.0
    assert meta["predictor_gate_reason"] == "predictor_exception"
    assert meta["predictor_error"] == "RuntimeError: broken checkpoint"
