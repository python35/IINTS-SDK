from __future__ import annotations

import pytest

from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.patient.physiology import renal_glucose_clearance_concentration, smooth_threshold_excess


def test_hovorka_iob_is_not_double_counted_across_multiple_doses() -> None:
    patient = HovorkaPatientModel(initial_glucose=120.0, insulin_action_duration=100.0)

    patient.update(time_step=10.0, delivered_insulin=1.0, carb_intake=0.0, current_time=0.0)
    patient.update(time_step=10.0, delivered_insulin=1.0, carb_intake=0.0, current_time=10.0)

    # Dose 1 age: 20 min -> 0.8 U remaining; dose 2 age: 10 min -> 0.9 U remaining.
    assert patient.insulin_on_board == pytest.approx(1.7, abs=1e-9)


def test_bolused_hovorka_meal_has_plausible_peak_timing() -> None:
    patient = HovorkaPatientModel(initial_glucose=120.0, max_glucose_rate_mgdl_per_min=4.0)
    trace: list[tuple[int, float]] = []

    for minute in range(0, 361, 5):
        basal_for_step = patient.basal_insulin_rate * 5.0 / 60.0
        glucose = patient.update(
            time_step=5.0,
            delivered_insulin=basal_for_step + (4.0 if minute == 0 else 0.0),
            carb_intake=60.0 if minute == 0 else 0.0,
            current_time=float(minute),
        )
        trace.append((minute, glucose))

    peak_minute, peak_glucose = max(trace, key=lambda row: row[1])
    assert 45 <= peak_minute <= 180
    assert peak_glucose > 150.0
    assert min(glucose for _, glucose in trace) >= 70.0
    assert trace[-1][1] < peak_glucose


def test_hovorka_exercise_lowers_glucose_without_impossible_crash() -> None:
    control = HovorkaPatientModel(initial_glucose=170.0, max_glucose_rate_mgdl_per_min=4.0)
    exercise = HovorkaPatientModel(initial_glucose=170.0, max_glucose_rate_mgdl_per_min=4.0)
    exercise.start_exercise(0.6)

    control_values = []
    exercise_values = []
    for minute in range(0, 121, 5):
        control_values.append(control.update(5.0, 0.0, 0.0, current_time=float(minute)))
        exercise_values.append(exercise.update(5.0, 0.0, 0.0, current_time=float(minute)))

    assert control_values[-1] - exercise_values[-1] > 20.0
    # Exercise can cause hypoglycemia when insulin exposure is not adjusted;
    # the model must remain finite rather than artificially clamping to 70.
    assert min(exercise_values) >= 20.0


def test_renal_clearance_curve_is_smooth_and_monotonic() -> None:
    low = renal_glucose_clearance_concentration(100.0)
    threshold = renal_glucose_clearance_concentration(180.0)
    high = renal_glucose_clearance_concentration(260.0)

    assert smooth_threshold_excess(180.0, threshold=180.0) > 0.0
    assert low < threshold < high
    assert low < 0.01
    assert high > 3.0


def test_hovorka_haaf_memory_is_bounded_during_repeated_lows() -> None:
    patient = HovorkaPatientModel(initial_glucose=60.0, max_glucose_rate_mgdl_per_min=4.0)
    values = []

    for minute in range(0, 121, 5):
        patient.update(5.0, 0.0, 0.0, current_time=float(minute))
        values.append(patient.get_patient_state()["haaf_metric"])

    assert max(values) <= 1.0
    assert values[0] > 0.0
    assert values[-1] > values[0]
