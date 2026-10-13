"""Regression tests for the software-contract defects found in the audit.

One test per defect, each small enough to read in a minute and each failing on
the pre-fix code. The identifiers D1-D4 match docs/DEFECT_REGISTER_NL.md.

    D1  a row mixed pre-step glucose with post-step compartments
    D2  a run of N minutes integrated the patient to N + time_step minutes
    D3  the PID produced a different hourly dose at every sample time
    D4  a delivery cap that is not a whole number of quantisation steps
        produced doses the pump cannot physically deliver
"""
from __future__ import annotations

import numpy as np
import pytest

from iints.api.base_algorithm import AlgorithmInput
from iints.core.algorithms.pid_controller import PIDController
from iints.core.devices.models import PumpModel
from iints.core.patient.bergman_model import BergmanPatientModel
from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel
from iints.core.simulator import Simulator
from iints.core.algorithms.mock_algorithms import ConstantDoseAlgorithm


def _simulator(patient, **kwargs):
    return Simulator(patient_model=patient,
                     algorithm=ConstantDoseAlgorithm(dose=0.05),
                     **kwargs)


# ------------------------------------------------------------------ D1
def test_d1_one_row_describes_one_instant_bergman() -> None:
    """glucose_actual_mgdl and the compartment columns share a timestamp."""
    simulator = _simulator(BergmanPatientModel(initial_glucose=140.0))
    frame, _ = simulator.run_batch(duration_minutes=120)

    difference = (frame["patient_state_G"] - frame["glucose_actual_mgdl"]).abs()
    assert difference.max() <= 0.1


def test_d1_one_row_describes_one_instant_hovorka() -> None:
    patient = HovorkaPatientModel(initial_glucose=140.0)
    volume_dl = patient.params.V_G_per_kg * patient.params.body_weight_kg * 10.0
    frame, _ = _simulator(patient).run_batch(duration_minutes=120)

    compartment_glucose = frame["patient_state_Q1"] / volume_dl
    assert (compartment_glucose - frame["glucose_actual_mgdl"]).abs().max() <= 0.1


def test_d1_interval_columns_are_explicit() -> None:
    simulator = _simulator(BergmanPatientModel(initial_glucose=120.0))
    frame, _ = simulator.run_batch(duration_minutes=60)

    assert (frame["interval_end_minutes"] - frame["time_minutes"]
            == simulator.time_step).all()
    # the state after the interval is available too, but in its own column
    assert "patient_iob_units_interval_end" in frame.columns


# ------------------------------------------------------------------ D2
@pytest.mark.parametrize("duration,time_step", [(120, 5), (60, 1), (240, 15)])
def test_d2_run_integrates_exactly_the_requested_horizon(duration, time_step) -> None:
    simulator = _simulator(BergmanPatientModel(initial_glucose=120.0),
                           time_step=time_step)
    frame, _ = simulator.run_batch(duration_minutes=duration)

    assert frame["interval_end_minutes"].max() == pytest.approx(duration)
    assert len(frame) == duration // time_step
    assert simulator.endpoint_observation()["time_minutes"] == pytest.approx(duration)


def test_d2_no_interval_starts_at_the_requested_end_time() -> None:
    """The old loop booked one extra interval of insulin past the horizon."""
    duration, step = 120, 5
    simulator = Simulator(patient_model=BergmanPatientModel(initial_glucose=120.0),
                          algorithm=ConstantDoseAlgorithm(dose=0.1))
    frame, _ = simulator.run_batch(duration_minutes=duration)

    assert frame["time_minutes"].max() == duration - step
    assert len(frame) == duration // step


# ------------------------------------------------------------------ D3
def _pid_hourly_units(time_step: float, error_mgdl: float = 1.0) -> float:
    """Total units a PID proposes in one hour at a fixed error, no feedback."""
    controller = PIDController()
    controller.reset()
    target = controller.target_glucose
    total = 0.0
    for step in range(int(60 / time_step)):
        data = AlgorithmInput(
            current_glucose=target + error_mgdl,
            time_step=time_step,
            insulin_on_board=0.0,
            current_time=step * time_step,
        )
        total += float(controller.predict_insulin(data)["total_insulin_delivered"])
    return total


def test_d3_pid_hourly_dose_is_portable_across_sample_times() -> None:
    reference = _pid_hourly_units(5.0)
    for time_step in (1.0, 2.0, 10.0, 15.0):
        hourly = _pid_hourly_units(time_step)
        relative_difference = abs(hourly - reference) / max(reference, 1e-9)
        assert relative_difference <= 0.05, (
            f"dt={time_step}: {hourly:.3f} U vs {reference:.3f} U at dt=5")


def test_d3_pid_keeps_its_tuned_behaviour_at_the_reference_interval() -> None:
    """The 5-minute behaviour is the tuning point and must not have moved."""
    controller = PIDController()
    controller.reset()
    data = AlgorithmInput(current_glucose=200.0, time_step=5.0, insulin_on_board=0.0)
    first = controller.predict_insulin(data)["total_insulin_delivered"]
    # kp * error = 0.1 * 80 = 8 -> saturated at max_insulin = 5 U
    assert first == pytest.approx(5.0)


def test_d3_pid_rejects_a_nonpositive_sample_time() -> None:
    controller = PIDController()
    with pytest.raises(ValueError):
        controller.predict_insulin(
            AlgorithmInput(current_glucose=150.0, time_step=0.0))


# ------------------------------------------------------------------ D4
def test_d4_capped_dose_stays_on_the_quantisation_grid() -> None:
    pump = PumpModel(max_units_per_step=0.975, quantization_units=0.05)
    delivery = pump.deliver(2.0, 5.0)

    assert delivery.delivered_units <= 0.975 + 1e-12
    assert delivery.delivered_units == pytest.approx(0.95)


def test_d4_every_delivery_is_a_whole_number_of_steps() -> None:
    pump = PumpModel(max_units_per_step=0.975, quantization_units=0.05, seed=7)
    rng = np.random.default_rng(11)

    for requested in rng.uniform(0.0, 2.0, 2000):
        delivered = pump.deliver(float(requested), 5.0).delivered_units
        steps = delivered / 0.05
        assert abs(steps - round(steps)) < 1e-9, f"{delivered} is not a whole step"
        assert delivered <= 0.975 + 1e-12


def test_d4_cap_without_quantisation_is_unchanged() -> None:
    pump = PumpModel(max_units_per_step=0.975)
    assert pump.deliver(2.0, 5.0).delivered_units == pytest.approx(0.975)


@pytest.mark.parametrize("model", [BergmanPatientModel, HovorkaPatientModel, AdvancedMetabolicModel])
@pytest.mark.parametrize("duration", [0, 3, 123])
def test_endpoint_tracks_the_actual_integrated_time(model, duration) -> None:
    simulator = _simulator(model(initial_glucose=140.0))
    assert simulator.endpoint_observation()["time_minutes"] == 0
    frame, _ = simulator.run_batch(duration_minutes=duration)
    assert simulator.endpoint_observation()["time_minutes"] == pytest.approx(duration)
    if duration:
        assert frame["interval_end_minutes"].iloc[-1] == pytest.approx(duration)
    saved = simulator.save_state()
    restored = _simulator(model(initial_glucose=140.0))
    restored.load_state(saved)
    expected = simulator.endpoint_observation()
    observed = restored.endpoint_observation()
    # State and its timestamp survive serialization; fluxes also depend on
    # transient infusion inputs and are not serialized patient state.
    for key in expected:
        if key.startswith("patient_flux_"):
            continue
        assert observed[key] == pytest.approx(expected[key])


def test_partial_interval_is_passed_to_the_controller() -> None:
    class RecordingAlgorithm(ConstantDoseAlgorithm):
        def predict_insulin(self, data):
            seen.append(data.time_step)
            return super().predict_insulin(data)

    seen = []
    simulator = Simulator(patient_model=BergmanPatientModel(initial_glucose=140.0),
                          algorithm=RecordingAlgorithm(dose=0.05), time_step=5)
    simulator.run_batch(duration_minutes=123)
    assert seen == [5] * 24 + [3]
