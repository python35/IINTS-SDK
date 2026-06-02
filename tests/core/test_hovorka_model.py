from __future__ import annotations

import pytest

from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.patient.patient_factory import PatientFactory


def test_hovorka_factory_creates_patient_model() -> None:
    patient = PatientFactory.create_patient(patient_type="hovorka", initial_glucose=118.0)

    assert isinstance(patient, HovorkaPatientModel)
    assert patient.get_current_glucose() == pytest.approx(118.0)


def test_hovorka_stress_increases_glucose_without_meal_or_insulin() -> None:
    baseline = HovorkaPatientModel(initial_glucose=112.0)
    stressed = HovorkaPatientModel(initial_glucose=112.0)
    stressed.start_stress(0.8)

    for minute in range(0, 60, 5):
        baseline_glucose = baseline.update(5.0, 0.0, 0.0, current_time=float(minute))
        stressed_glucose = stressed.update(5.0, 0.0, 0.0, current_time=float(minute))

    assert stressed_glucose > baseline_glucose + 15.0
    assert stressed_glucose < 220.0


def test_hovorka_state_round_trip_preserves_physiology_flags() -> None:
    patient = HovorkaPatientModel(initial_glucose=121.0)
    patient.start_stress(0.6)
    patient.start_exercise(0.3)
    patient.update(5.0, delivered_insulin=0.1, carb_intake=12.0, current_time=60.0)

    clone = HovorkaPatientModel(initial_glucose=95.0)
    clone.set_state(patient.get_state())

    assert clone.get_current_glucose() == pytest.approx(patient.get_current_glucose())
    assert clone.is_stressed is True
    assert clone.stress_intensity == pytest.approx(0.6)
    assert clone.is_exercising is True
    assert clone.exercise_intensity == pytest.approx(0.3)
    assert len(clone.get_state()["ode_state"]) == 10


def test_hovorka_loads_legacy_bergman_state_shape() -> None:
    patient = HovorkaPatientModel(initial_glucose=100.0)

    patient.set_state(
        {
            "ode_state": [145.0, 0.01, 12.0, 250.0, 500.0, 20.0, 30.0],
            "current_glucose": 145.0,
            "is_stressed": True,
            "stress_intensity": 0.4,
        }
    )

    state = patient.get_state()
    assert len(state["ode_state"]) == 10
    assert patient.get_current_glucose() == pytest.approx(145.0)
    assert state["is_stressed"] is True
