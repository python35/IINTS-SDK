"""Tests for physiological calibration of the virtual patients.

The point of these tests is identifiability: a configured clinical knob must
have a measurable physiological consequence. They therefore assert on
*measured* behaviour (paired simulations) rather than on stored attributes.
"""
from __future__ import annotations

import pytest

from iints.core.patient.bergman_model import BergmanPatientModel
from iints.core.patient.calibration import (
    calibrate_carb_factor,
    calibrate_hovorka,
    hovorka_euglycaemic_basal_rate,
    measure_fasting_drift,
    measure_insulin_sensitivity_factor,
    run_protocol,
)
from iints.core.patient.hovorka_model import HovorkaPatientModel


@pytest.fixture(scope="module")
def calibration_isf50():
    return calibrate_hovorka(insulin_sensitivity=50.0, target_glucose_mgdl=120.0)


@pytest.fixture(scope="module")
def calibration_isf30():
    return calibrate_hovorka(insulin_sensitivity=30.0, target_glucose_mgdl=120.0)


def test_legacy_constructor_keeps_fitting_sensitivity_to_the_basal_rate() -> None:
    patient = HovorkaPatientModel(basal_insulin_rate=0.8, insulin_sensitivity=50.0,
                                  initial_glucose=120.0)
    assert patient._basal_parameter_scale == pytest.approx(0.4707, abs=5e-4)


def test_calibrated_patient_realises_its_configured_isf(calibration_isf50) -> None:
    assert calibration_isf50.converged
    assert calibration_isf50.realised_isf_mgdl_per_u == pytest.approx(50.0, abs=1.0)
    assert calibration_isf50.basal_insulin_rate > 0.0


def test_calibrated_patient_holds_its_fasting_target(calibration_isf50) -> None:
    assert calibration_isf50.fasting_drift_mgdl < 1.0


def test_insulin_sensitivity_factor_is_identifiable_after_calibration(
        calibration_isf50, calibration_isf30) -> None:
    """Two patients that differ only in ISF must differ in insulin response."""
    assert calibration_isf30.realised_isf_mgdl_per_u == pytest.approx(30.0, abs=1.0)
    assert (calibration_isf50.realised_isf_mgdl_per_u
            > calibration_isf30.realised_isf_mgdl_per_u + 10.0)
    # A less sensitive patient needs more basal insulin to hold the same
    # fasting target. (The scale itself may land on the same value: the
    # configured ISF already enters the effective sensitivity as ISF/50.)
    assert (calibration_isf30.basal_insulin_rate
            > calibration_isf50.basal_insulin_rate + 0.1)


def test_derived_basal_rate_is_a_true_equilibrium(calibration_isf50) -> None:
    drift = measure_fasting_drift(calibration_isf50.build,
                                  calibration_isf50.basal_insulin_rate, hours=24.0)
    assert drift["max_abs_drift_mgdl"] < 1.0
    assert drift["final_glucose_mgdl"] == pytest.approx(120.0, abs=1.0)


def test_euglycaemic_basal_rate_matches_the_simulated_equilibrium() -> None:
    patient = HovorkaPatientModel(initial_glucose=130.0, insulin_sensitivity_scale=0.6)
    effective = 0.6 * patient._clinical_sensitivity_scale
    basal = hovorka_euglycaemic_basal_rate(
        patient.params, 130.0, transport_scale=effective,
        disposal_scale=effective, egp_scale=effective)

    def factory():
        return HovorkaPatientModel(initial_glucose=130.0, basal_insulin_rate=basal,
                                   insulin_sensitivity_scale=0.6)

    assert measure_fasting_drift(factory, basal, hours=12.0)["max_abs_drift_mgdl"] < 1.0


def test_measure_clinical_parameters_reports_configured_and_realised(
        calibration_isf50) -> None:
    report = calibration_isf50.build().measure_clinical_parameters()
    assert report["configured_isf_mgdl_per_u"] == pytest.approx(50.0)
    assert report["realised_isf_mgdl_per_u"] == pytest.approx(50.0, abs=1.5)
    assert report["euglycaemic_basal_u_per_h"] == pytest.approx(
        report["configured_basal_u_per_h"], abs=1e-3)
    assert report["fasting_max_abs_drift_mgdl"] < 1.0


def test_state_round_trip_preserves_the_calibration(calibration_isf50) -> None:
    patient = calibration_isf50.build()
    clone = HovorkaPatientModel(initial_glucose=95.0, basal_insulin_rate=0.3)
    clone.set_state(patient.get_state())
    assert clone._basal_parameter_scale == pytest.approx(
        patient._basal_parameter_scale)


def test_minimum_glucose_domain_is_enforced_when_requested() -> None:
    patient = HovorkaPatientModel(initial_glucose=120.0, insulin_sensitivity_scale=1.0,
                                  min_glucose_mgdl=20.0, max_glucose_rate_mgdl_per_min=0.0)
    run = run_protocol(patient, 24 * 60.0, 3.0, dt=5.0)
    assert run["aborted"]
    assert "20.0 mg/dL domain" in run["error"]


def test_calibrated_carb_factor_returns_the_patient_to_baseline(calibration_isf50) -> None:
    ratio = calibrate_carb_factor(calibration_isf50, meal_grams=60.0)
    assert 3.0 < ratio < 40.0
    model = calibration_isf50.build(carb_factor=ratio)
    run = run_protocol(model, 6 * 60.0, calibration_isf50.basal_insulin_rate, dt=5.0,
                       meals={0.0: 60.0}, boluses={0.0: 60.0 / ratio})
    assert abs(run["glucose"][-1] - run["glucose"][0]) < 5.0


def test_bergman_basal_rate_is_inert_in_legacy_mode() -> None:
    ends = []
    for basal in (0.4, 1.6):
        model = BergmanPatientModel(basal_insulin_rate=basal, initial_glucose=120.0)
        ends.append(run_protocol(model, 12 * 60.0, basal, dt=5.0)["glucose"][-1])
    assert abs(ends[0] - ends[1]) < 1.0


def test_bergman_physiological_reference_restores_a_basal_dose_response() -> None:
    ends = []
    for basal in (0.4, 1.6):
        model = BergmanPatientModel(basal_insulin_rate=basal, initial_glucose=120.0,
                                    insulin_action_reference="plasma_ib")
        ends.append(run_protocol(model, 12 * 60.0, basal, dt=5.0)["glucose"][-1])
    assert ends[1] < ends[0] - 5.0


def test_bergman_absorption_and_emptying_can_be_set_from_literature() -> None:
    model = BergmanPatientModel(initial_glucose=120.0,
                                insulin_absorption_tmax_minutes=55.0,
                                gastric_emptying_minutes=40.0)
    assert model.params.k_a == pytest.approx(1.0 / 55.0)
    assert model.params.tau_meal == pytest.approx(40.0)


def test_unknown_insulin_action_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="insulin_action_reference"):
        BergmanPatientModel(insulin_action_reference="magic")
