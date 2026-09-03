from __future__ import annotations

from iints.core.devices.models import SensorModel
from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.simulator import Simulator
from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm


def test_run_batch_step_callback_receives_time_and_glucose() -> None:
    """run_batch's step_callback pulls from the per-step record by key.

    The record uses time_minutes/glucose_actual_mgdl (tests/core/test_simulator_xai.py
    and results.py agree on this schema); a stale record["time"]/record["glucose"]
    lookup raised KeyError on every desktop-app run, since the record dict never
    had those keys.
    """
    calls: list[tuple[int, int, float]] = []
    patient = HovorkaPatientModel(initial_glucose=150.0)
    simulator = Simulator(
        patient_model=patient,
        algorithm=ClinicalBaselineAlgorithm(),
        time_step=5,
        sensor_model=SensorModel(),
    )

    simulator.run_batch(
        duration_minutes=10,
        step_callback=lambda step, duration, glucose: calls.append((step, duration, glucose)),
    )

    assert len(calls) >= 2
    steps = [call[0] for call in calls]
    assert steps == sorted(steps)
    assert all(duration == 10 for _, duration, _ in calls)
    assert all(50.0 <= glucose <= 400.0 for _, _, glucose in calls)
