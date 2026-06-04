from __future__ import annotations

import pytest

from iints.api.base_algorithm import AlgorithmInput, InsulinAlgorithm
from iints.core.devices.models import SensorModel
from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.safety.config import SafetyConfig
from iints.core.simulator import Simulator


class _GlucagonAlgorithm(InsulinAlgorithm):
    def __init__(self, glucagon_mg: float) -> None:
        super().__init__()
        self.glucagon_mg = glucagon_mg

    def predict_insulin(self, data: AlgorithmInput) -> dict[str, float]:
        self.why_log = []
        return {
            "total_insulin_delivered": 0.0,
            "total_glucagon_delivered_mg": self.glucagon_mg,
            "basal_insulin": 0.0,
            "bolus_insulin": 0.0,
        }


def test_simulator_blocks_glucagon_when_not_low_risk() -> None:
    patient = HovorkaPatientModel(initial_glucose=150.0)
    simulator = Simulator(
        patient_model=patient,
        algorithm=_GlucagonAlgorithm(glucagon_mg=5.0),
        time_step=5,
        sensor_model=SensorModel(),
    )

    results, safety_report = simulator.run_batch(duration_minutes=5)

    assert results["algo_recommended_glucagon_mg"].max() == pytest.approx(5.0)
    assert results["delivered_glucagon_mg"].max() == pytest.approx(0.0)
    assert results["glucagon_safety_triggered"].any()
    assert "GLUCAGON_BLOCKED_NOT_LOW_RISK" in " ".join(results["glucagon_safety_reason"].fillna(""))
    assert safety_report["glucagon_safety_interventions_count"] >= 1


def test_simulator_caps_rescue_glucagon_per_step() -> None:
    patient = HovorkaPatientModel(initial_glucose=62.0)
    simulator = Simulator(
        patient_model=patient,
        algorithm=_GlucagonAlgorithm(glucagon_mg=5.0),
        time_step=5,
        sensor_model=SensorModel(),
        safety_config=SafetyConfig(max_glucagon_per_step_mg=0.4, max_glucagon_per_hour_mg=1.0),
    )

    results, safety_report = simulator.run_batch(duration_minutes=5)

    assert results["algo_recommended_glucagon_mg"].max() == pytest.approx(5.0)
    assert results["delivered_glucagon_mg"].max() <= 0.4 + 1e-9
    assert results["delivered_glucagon_mg"].max() > 0.0
    assert "GLUCAGON_STEP_CAP" in " ".join(results["glucagon_safety_reason"].fillna(""))
    assert safety_report["glucagon_safety_interventions_count"] >= 1
