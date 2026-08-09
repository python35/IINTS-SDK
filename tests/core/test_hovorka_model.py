from __future__ import annotations

import pytest
import numpy as np

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
    assert len(clone.get_state()["ode_state"]) == 19
    physiology = clone.get_patient_state()
    assert physiology["delivered_insulin"] == pytest.approx(0.1)
    assert "active_insulin" in physiology
    assert "insulin_effect" in physiology
    assert "plasma_glucagon_pg_ml" in physiology
    assert "haaf_metric" in physiology
    assert "glut4_active" in physiology


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
    assert len(state["ode_state"]) == 19
    assert patient.get_current_glucose() == pytest.approx(145.0)
    assert state["is_stressed"] is True


def test_hovorka_hypoglycemia_builds_haaf_memory() -> None:
    patient = HovorkaPatientModel(initial_glucose=65.0)

    patient.update(5.0, delivered_insulin=0.0, carb_intake=0.0, current_time=0.0)

    physiology = patient.get_patient_state()
    assert physiology["haaf_metric"] > 0.0
    assert 20.0 <= patient.get_current_glucose() <= 120.0


def test_hovorka_tracks_exogenous_glucagon_state() -> None:
    patient = HovorkaPatientModel(initial_glucose=62.0)

    patient.update(5.0, delivered_insulin=0.0, carb_intake=0.0, delivered_glucagon_mg=0.2, current_time=0.0)
    for minute in range(5, 35, 5):
        patient.update(5.0, delivered_insulin=0.0, carb_intake=0.0, current_time=float(minute))

    physiology = patient.get_patient_state()
    assert patient.get_state()["last_delivered_glucagon_mg"] == pytest.approx(0.0)
    assert physiology["plasma_glucagon_pg_ml"] > 0.0


def test_hovorka_exercise_activates_glut4_without_insulin() -> None:
    patient = HovorkaPatientModel(initial_glucose=130.0)
    patient.start_exercise(0.8)

    for minute in range(0, 60, 5):
        glucose = patient.update(5.0, delivered_insulin=0.0, carb_intake=0.0, current_time=float(minute))

    physiology = patient.get_patient_state()
    assert physiology["glut4_active"] > 0.0
    assert glucose < 130.0


def test_hovorka_circadian_egp_is_gated_by_dawn_strength() -> None:
    no_dawn = HovorkaPatientModel(initial_glucose=120.0, dawn_phenomenon_strength=0.0)
    dawn = HovorkaPatientModel(initial_glucose=120.0, dawn_phenomenon_strength=8.0)

    no_dawn_state = np.array(no_dawn.get_state()["ode_state"], dtype=float)
    dawn_state = np.array(dawn.get_state()["ode_state"], dtype=float)

    no_dawn_morning = no_dawn._ode(0.0, no_dawn_state, 0.0, 0.0, current_time=360.0)[0]
    no_dawn_evening = no_dawn._ode(0.0, no_dawn_state, 0.0, 0.0, current_time=1080.0)[0]
    dawn_morning = dawn._ode(0.0, dawn_state, 0.0, 0.0, current_time=360.0)[0]
    dawn_evening = dawn._ode(0.0, dawn_state, 0.0, 0.0, current_time=1080.0)[0]

    assert no_dawn_morning == pytest.approx(no_dawn_evening)
    assert dawn_morning > dawn_evening


def test_hovorka_molecular_affinity_scalar_affects_insulin_action_curve() -> None:
    normal = HovorkaPatientModel(
        initial_glucose=120.0,
        molecular_affinity_scalar=1.0,
        max_glucose_rate_mgdl_per_min=4.0,
    )
    resistant = HovorkaPatientModel(
        initial_glucose=120.0,
        molecular_affinity_scalar=0.2,
        max_glucose_rate_mgdl_per_min=4.0,
    )
    normal_trace: list[float] = []
    resistant_trace: list[float] = []

    for minute in range(0, 241, 5):
        carbs = 60.0 if minute == 30 else 0.0
        basal_for_step = normal.basal_insulin_rate * 5.0 / 60.0
        insulin = basal_for_step + (6.0 if minute == 25 else 0.0)
        normal_trace.append(normal.update(5.0, insulin, carbs, current_time=float(minute)))
        resistant_trace.append(resistant.update(5.0, insulin, carbs, current_time=float(minute)))

    assert max(abs(a - b) for a, b in zip(normal_trace, resistant_trace)) > 5.0
    assert resistant_trace[-1] > normal_trace[-1]
