from __future__ import annotations

import copy

import numpy as np
import pytest

from iints.core.devices.models import PumpModel, SensorModel
from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel
from iints.core.patient.bergman_model import BergmanParameters, BergmanPatientModel
from iints.core.patient.hovorka_model import HovorkaParameters, HovorkaPatientModel
from iints.core.patient.models import CustomPatientModel, PatientModelDomainError
from iints.core.patient.physiology import (
    antecedent_hypoglycemia_memory_derivative,
    counterregulatory_rescue_multiplier,
    dawn_glucose_rate_mgdl_min,
    glucagon_mg_to_pg,
)
from iints.core.simulator import SimulationLimitError, Simulator


@pytest.mark.parametrize(
    "model_type",
    [BergmanPatientModel, HovorkaPatientModel, AdvancedMetabolicModel],
)
def test_pump_supported_fasting_state_is_numerically_stable(model_type) -> None:
    patient = model_type(
        initial_glucose=110.0,
        basal_insulin_rate=0.8,
        max_glucose_rate_mgdl_per_min=10.0,
    )

    for minute in range(0, 360, 5):
        basal_for_step = patient.basal_insulin_rate * 5.0 / 60.0
        patient.update(5.0, basal_for_step, current_time=float(minute))

    assert patient.get_current_glucose() == pytest.approx(110.0, abs=1.0)


def test_bergman_constructor_does_not_silently_rewrite_parameters() -> None:
    patient = BergmanPatientModel(
        initial_glucose=140.0,
        basal_glucose_target=135.0,
        glucose_absorption_rate=0.012,
    )

    assert patient.params.Gb == pytest.approx(135.0)
    assert patient.params.k_abs == pytest.approx(0.012)


@pytest.mark.parametrize("model_type", [BergmanPatientModel, HovorkaPatientModel])
def test_complete_basal_interruption_raises_glucose(model_type) -> None:
    patient = model_type(
        initial_glucose=110.0,
        basal_insulin_rate=0.8,
        max_glucose_rate_mgdl_per_min=10.0,
    )

    for minute in range(0, 360, 5):
        patient.update(5.0, 0.0, current_time=float(minute))

    assert patient.get_current_glucose() > 115.0


def test_glucagon_mass_conversion_and_pk_scale_are_consistent() -> None:
    assert glucagon_mg_to_pg(0.2) == pytest.approx(2.0e8)

    patient = HovorkaPatientModel(
        initial_glucose=62.0,
        max_glucose_rate_mgdl_per_min=10.0,
    )
    concentrations: list[float] = []
    glucose: list[float] = []
    for minute in range(0, 61, 5):
        basal_for_step = patient.basal_insulin_rate * 5.0 / 60.0
        glucose.append(
            patient.update(
                5.0,
                basal_for_step,
                delivered_glucagon_mg=0.2 if minute == 0 else 0.0,
                current_time=float(minute),
            )
        )
        concentrations.append(patient.get_patient_state()["plasma_glucagon_pg_ml"])

    # Low-dose human T1D studies report increments of hundreds of pg/mL,
    # not the million-scale values produced by an L-versus-mL unit error.
    assert 100.0 <= max(concentrations) <= 2_000.0
    assert 5 <= (concentrations.index(max(concentrations)) + 1) * 5 <= 25
    assert max(glucose) >= 75.0
    assert max(glucose) <= 130.0


@pytest.mark.parametrize("bad_dose", [-0.1, float("nan"), float("inf")])
def test_glucagon_mass_conversion_rejects_invalid_dose(bad_dose: float) -> None:
    with pytest.raises(ValueError):
        glucagon_mg_to_pg(bad_dose)


def test_dawn_profile_has_declared_units_and_smooth_boundaries() -> None:
    kwargs = {
        "peak_strength_mgdl_per_hour": 12.0,
        "start_hour": 4.0,
        "end_hour": 8.0,
    }

    assert dawn_glucose_rate_mgdl_min(4.0 * 60.0, **kwargs) == pytest.approx(0.0)
    assert dawn_glucose_rate_mgdl_min(6.0 * 60.0, **kwargs) == pytest.approx(0.2)
    assert dawn_glucose_rate_mgdl_min(8.0 * 60.0, **kwargs) == pytest.approx(0.0)
    assert dawn_glucose_rate_mgdl_min(12.0 * 60.0, **kwargs) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "model_type",
    [CustomPatientModel, BergmanPatientModel, HovorkaPatientModel],
)
def test_dawn_setting_raises_glucose_consistently_across_backends(model_type) -> None:
    common = {
        "initial_glucose": 110.0,
        "basal_insulin_rate": 0.8,
        "dawn_phenomenon_strength": 12.0,
        "dawn_start_hour": 4.0,
        "dawn_end_hour": 8.0,
        "max_glucose_rate_mgdl_per_min": 10.0,
    }
    dawn = model_type(**common)
    control = model_type(**{**common, "dawn_phenomenon_strength": 0.0})

    for minute in range(4 * 60, 8 * 60, 5):
        basal_for_step = dawn.basal_insulin_rate * 5.0 / 60.0
        dawn.update(5.0, basal_for_step, current_time=float(minute))
        control.update(5.0, basal_for_step, current_time=float(minute))

    assert dawn.get_current_glucose() > control.get_current_glucose()


def test_haaf_abstraction_builds_slowly_and_rescue_is_bounded() -> None:
    memory = 0.0
    for _ in range(6):
        memory += 5.0 * antecedent_hypoglycemia_memory_derivative(50.0, memory)

    assert 0.05 < memory < 0.15
    assert 1.0 <= counterregulatory_rescue_multiplier(50.0, 0.0) <= 2.0
    assert counterregulatory_rescue_multiplier(50.0, 1.0) == pytest.approx(1.0)


def test_patient_snapshot_rejects_conflicting_glucose_values() -> None:
    bergman = BergmanPatientModel(initial_glucose=110.0)
    bergman_state = bergman.get_state()
    bergman_state["current_glucose"] = 150.0
    with pytest.raises(ValueError, match="inconsistent"):
        bergman.set_state(bergman_state)

    hovorka = HovorkaPatientModel(initial_glucose=110.0)
    hovorka_state = hovorka.get_state()
    hovorka_state["current_glucose"] = 150.0
    with pytest.raises(ValueError, match="inconsistent"):
        hovorka.set_state(hovorka_state)


def test_sensor_requires_monotonic_time_and_reset_replays_rng() -> None:
    sensor = SensorModel(noise_std=8.0, noise_fbm_hurst=0.75, seed=17)
    first_run = [sensor.read(120.0, float(minute)).value for minute in range(0, 20, 5)]

    with pytest.raises(ValueError, match="strictly increasing"):
        sensor.read(120.0, 15.0)

    sensor.reset()
    second_run = [sensor.read(120.0, float(minute)).value for minute in range(0, 20, 5)]
    assert second_run == first_run


def test_pump_reset_replays_stochastic_delivery_sequence() -> None:
    pump = PumpModel(
        quantization_units=0.05,
        step_error_probability=0.25,
        dropout_prob=0.1,
        seed=9,
    )
    first = [pump.deliver(1.0, 5.0) for _ in range(12)]
    pump.reset()
    second = [pump.deliver(1.0, 5.0) for _ in range(12)]

    assert second == first


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), -1.0],
)
def test_patient_models_reject_invalid_delivered_insulin(bad_value: float) -> None:
    for patient in (BergmanPatientModel(), HovorkaPatientModel()):
        with pytest.raises(ValueError):
            patient.update(5.0, bad_value)


def test_hovorka_meal_curve_obeys_configured_rate_envelope() -> None:
    patient = HovorkaPatientModel(
        initial_glucose=120.0,
        max_glucose_rate_mgdl_per_min=3.0,
    )
    values = [patient.get_current_glucose()]
    for minute in range(0, 241, 5):
        basal_for_step = patient.basal_insulin_rate * 5.0 / 60.0
        values.append(
            patient.update(
                5.0,
                basal_for_step,
                carb_intake=60.0 if minute == 0 else 0.0,
                current_time=float(minute),
            )
        )

    rates = np.abs(np.diff(np.asarray(values))) / 5.0
    assert float(np.max(rates)) <= 3.0 + 1e-9


def test_custom_model_action_integrals_are_time_step_invariant() -> None:
    def insulin_trace(step: float) -> float:
        patient = CustomPatientModel(
            initial_glucose=200.0,
            glucose_decay_rate=0.0,
            max_glucose_rate_mgdl_per_min=10.0,
        )
        for index, _minute in enumerate(np.arange(0.0, 300.0, step)):
            patient.update(step, 1.0 if index == 0 else 0.0)
        return patient.get_current_glucose()

    def meal_trace(step: float) -> float:
        patient = CustomPatientModel(
            initial_glucose=100.0,
            glucose_decay_rate=0.0,
            max_glucose_rate_mgdl_per_min=10.0,
        )
        for index, _minute in enumerate(np.arange(0.0, 400.0, step)):
            patient.update(step, 0.0, carb_intake=10.0 if index == 0 else 0.0)
        return patient.get_current_glucose()

    assert insulin_trace(1.0) == pytest.approx(insulin_trace(5.0), abs=1e-8)
    assert meal_trace(1.0) == pytest.approx(meal_trace(5.0), abs=1e-8)


def test_custom_model_rejects_unsupported_glucagon_delivery() -> None:
    patient = CustomPatientModel()
    with pytest.raises(NotImplementedError, match="no glucagon PK/PD"):
        patient.update(5.0, 0.0, delivered_glucagon_mg=0.1)


def test_mechanistic_model_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="body_weight_kg"):
        BergmanParameters(body_weight_kg=0.0)
    with pytest.raises(ValueError, match="A_G"):
        HovorkaParameters(A_G=1.1)
    with pytest.raises(ValueError, match="insulin_type"):
        HovorkaParameters(insulin_type="unknown")


@pytest.mark.parametrize("model_type", [BergmanPatientModel, HovorkaPatientModel])
def test_mechanistic_models_reject_unknown_update_inputs(model_type) -> None:
    patient = model_type()
    with pytest.raises(TypeError, match="Unsupported"):
        patient.update(5.0, 0.0, unmodelled_hormone=1.0)


@pytest.mark.parametrize("model_type", [BergmanPatientModel, HovorkaPatientModel])
def test_ratio_updates_fail_closed_and_do_not_reinitialize_state(model_type) -> None:
    patient = model_type(initial_glucose=125.0)
    before = patient.get_state()["ode_state"]

    patient.set_ratio_state(basal_rate=0.9, isf=45.0, icr=11.0, dia_minutes=280.0)

    assert patient.get_state()["ode_state"] == before
    assert patient.get_ratio_state() == pytest.approx(
        {
            "basal_rate_u_per_hr": 0.9,
            "isf": 45.0,
            "icr": 11.0,
            "dia_minutes": 280.0,
        }
    )
    with pytest.raises(ValueError):
        patient.set_ratio_state(isf=float("nan"))
    with pytest.raises(ValueError):
        patient.set_ratio_state(icr=0.0)
    with pytest.raises(ValueError):
        patient.set_ratio_state(basal_rate=-0.1)


@pytest.mark.parametrize(
    ("model_type", "y2_index", "gamma_index"),
    [
        (BergmanPatientModel, 9, 10),
        (HovorkaPatientModel, 14, 15),
        (AdvancedMetabolicModel, 9, 10),
    ],
)
def test_snapshot_restore_recomputes_algebraic_glucagon_concentration(
    model_type,
    y2_index: int,
    gamma_index: int,
) -> None:
    patient = model_type(initial_glucose=110.0)
    state = patient.get_state()
    state["ode_state"][y2_index] = 1.0e7
    state["ode_state"][gamma_index] = 999_999.0
    patient.set_state(state)

    restored = patient.get_state()["ode_state"]
    expected = (
        patient.params.k_e_glucagon
        * restored[y2_index]
        / (
            patient.params.glucagon_clearance_ml_kg_min
            * patient.params.body_weight_kg
        )
    )
    assert restored[gamma_index] == pytest.approx(expected)


@pytest.mark.parametrize(
    "model_type", [CustomPatientModel, BergmanPatientModel, HovorkaPatientModel]
)
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("insulin_on_board", float("nan")),
        ("carbs_on_board", -0.1),
        ("exercise_intensity", 1.1),
        ("stress_intensity", float("inf")),
        ("is_exercising", "false"),
    ],
)
def test_patient_snapshots_fail_closed_on_invalid_metadata(
    model_type, field: str, bad_value: object
) -> None:
    patient = model_type()
    state = patient.get_state()
    state[field] = bad_value

    with pytest.raises(ValueError, match=field):
        patient.set_state(state)


@pytest.mark.parametrize(
    "model_type", [CustomPatientModel, BergmanPatientModel, HovorkaPatientModel]
)
def test_patient_snapshots_validate_activity_event_schema(model_type) -> None:
    patient = model_type()
    state = patient.get_state()
    state["active_insulin_doses"] = [{"amount": 1.0, "age": float("nan")}]

    with pytest.raises(ValueError, match="active_insulin_doses"):
        patient.set_state(state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_std": -1.0},
        {"noise_ar1_phi": 1.0},
        {"noise_fbm_hurst": 0.2},
        {"dropout_prob": 1.1},
        {"dropout_duration_steps": (0, 2)},
        {"compression_low_mgdl_range": (20.0, 10.0)},
    ],
)
def test_sensor_configuration_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SensorModel(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_units_per_step": 0.0},
        {"quantization_units": -0.01},
        {"dropout_prob": float("nan")},
        {"delivery_noise_std": -0.1},
        {"step_error_probability": 1.1},
    ],
)
def test_pump_configuration_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PumpModel(**kwargs)


@pytest.mark.parametrize(
    "model_type",
    [
        CustomPatientModel,
        BergmanPatientModel,
        HovorkaPatientModel,
        AdvancedMetabolicModel,
    ],
)
def test_failed_patient_step_rolls_back_all_dynamic_state(
    model_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    patient = model_type()
    before = copy.deepcopy(patient.get_state())

    def reject_transition(_value: float, _time_step: float) -> float:
        raise RuntimeError("forced validation failure")

    monkeypatch.setattr(patient, "_guard_glucose_transition", reject_transition)
    kwargs = {"fat_intake": 4.0} if model_type is AdvancedMetabolicModel else {}
    with pytest.raises(RuntimeError, match="forced validation failure"):
        patient.update(5.0, 1.0, carb_intake=20.0, **kwargs)

    assert patient.get_state() == before


@pytest.mark.parametrize(
    ("model_type", "index"),
    [
        (BergmanPatientModel, 12),
        (HovorkaPatientModel, 18),
        (AdvancedMetabolicModel, 15),
    ],
)
def test_fraction_states_above_one_are_rejected_on_restore(
    model_type, index: int
) -> None:
    patient = model_type()
    state = patient.get_state()
    state["ode_state"][index] = 1.01

    with pytest.raises(ValueError, match="fraction state"):
        patient.set_state(state)


def test_simulator_never_clips_or_replaces_hidden_patient_truth() -> None:
    simulator = Simulator(
        patient_model=CustomPatientModel(),
        algorithm=ClinicalBaselineAlgorithm(),
    )

    assert simulator._bound_simulation_glucose(120.0, 0.0) == 120.0
    for invalid in (10.0, 700.0, float("nan")):
        with pytest.raises(SimulationLimitError):
            simulator._bound_simulation_glucose(invalid, 0.0)


def test_simulator_reports_patient_model_domain_exit_without_fabricating_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patient = CustomPatientModel(initial_glucose=120.0)
    simulator = Simulator(
        patient_model=patient,
        algorithm=ClinicalBaselineAlgorithm(),
    )

    def leave_domain(*_args: object, **_kwargs: object) -> float:
        raise PatientModelDomainError(
            "forced model-domain exit",
            current_glucose=120.0,
            proposed_glucose=-5.0,
        )

    monkeypatch.setattr(patient, "update", leave_domain)
    results, safety_report = simulator.run_batch(duration_minutes=5)

    assert results.empty
    assert safety_report["terminated_early"] is True
    termination = safety_report["termination_reason"]
    assert termination["termination_class"] == "patient_model_domain"
    assert termination["last_supported_glucose_mgdl"] == pytest.approx(120.0)
    assert termination["glucose_value"] == pytest.approx(-5.0)
