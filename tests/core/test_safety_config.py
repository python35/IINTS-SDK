from iints.core.safety import SafetyConfig
from iints.core.safety.input_validator import InputValidator
from iints.core.supervisor import IndependentSupervisor
from iints.core.simulator import Simulator, StressEvent
from iints.core.patient.models import PatientModel
from iints.core.algorithms.mock_algorithms import ConstantDoseAlgorithm
from iints.core.devices.models import SensorModel


def test_input_validator_uses_safety_config():
    config = SafetyConfig(min_glucose=55.0, max_glucose=350.0, max_glucose_delta_per_5_min=15.0)
    validator = InputValidator(safety_config=config)

    assert validator.min_glucose == 55.0
    assert validator.max_glucose == 350.0
    assert validator.max_glucose_delta_per_5_min == 15.0


def test_supervisor_uses_safety_config():
    config = SafetyConfig(
        max_insulin_per_bolus=2.5,
        max_insulin_per_hour=2.0,
        hypo_cutoff=80.0,
    )
    supervisor = IndependentSupervisor(safety_config=config)

    assert supervisor.max_insulin_per_bolus == 2.5
    assert supervisor.max_60min == 2.0
    assert supervisor.hypo_cutoff == 80.0


def test_simulator_uses_critical_thresholds_from_config():
    config = SafetyConfig(critical_glucose_threshold=50.0, critical_glucose_duration_minutes=15)
    patient = PatientModel(initial_glucose=120.0)
    algo = ConstantDoseAlgorithm()
    simulator = Simulator(patient_model=patient, algorithm=algo, safety_config=config)

    assert simulator.critical_glucose_threshold == 50.0
    assert simulator.critical_glucose_duration_minutes == 15


def test_safety_config_has_stable_versioned_fingerprint():
    first = SafetyConfig()
    second = SafetyConfig()

    assert first.to_versioned_dict()["formula_version"] == "iints-safety-formulas-v1"
    assert first.to_versioned_dict()["units"]["glucose"] == "mg/dL"
    assert first.fingerprint_sha256() == second.fingerprint_sha256()
    assert first.fingerprint_sha256() != SafetyConfig(max_iob=3.5).fingerprint_sha256()


def test_simulator_keeps_raw_and_sensor_validation_history_separate():
    patient = PatientModel(initial_glucose=120.0)
    simulator = Simulator(
        patient_model=patient,
        algorithm=ConstantDoseAlgorithm(dose=0.0),
        sensor_model=SensorModel(noise_std=35.0, seed=2),
    )

    results, _ = simulator.run_batch(duration_minutes=20)

    assert results["glucose_actual_mgdl"].eq(120.0).all()


def test_simulator_does_not_rate_limit_hidden_patient_truth():
    patient = PatientModel(
        initial_glucose=120.0,
        glucose_absorption_rate=2.0,
        carb_absorption_duration_minutes=60.0,
    )
    simulator = Simulator(
        patient_model=patient,
        algorithm=ConstantDoseAlgorithm(dose=0.0),
        sensor_model=SensorModel(noise_std=0.0, seed=2),
    )
    simulator.add_stress_event(StressEvent(start_time=0, event_type="meal", value=100.0))

    results, _ = simulator.run_batch(duration_minutes=20)

    assert results["glucose_actual_mgdl"].equals(results["glucose_mechanistic_mgdl"])
    assert results["glucose_actual_mgdl"].max() > 160.0


def test_sensor_fail_soft_rate_limits_toward_incoming_value():
    patient = PatientModel(initial_glucose=120.0)
    simulator = Simulator(
        patient_model=patient,
        algorithm=ConstantDoseAlgorithm(dose=0.0),
        sensor_model=SensorModel(noise_std=0.0, seed=2),
    )
    simulator.add_stress_event(StressEvent(start_time=5, event_type="sensor_error", value=200.0))
    simulator.add_stress_event(StressEvent(start_time=10, event_type="sensor_error", value=200.0))

    results, _ = simulator.run_batch(duration_minutes=10)

    by_time = results.set_index("time_minutes")
    assert by_time.loc[5, "glucose_to_algo_mgdl"] == 140.0
    assert by_time.loc[10, "glucose_to_algo_mgdl"] == 160.0


def test_glucose_trend_uses_rolling_slope_instead_of_single_noisy_drop():
    simulator = Simulator(
        patient_model=PatientModel(initial_glucose=120.0),
        algorithm=ConstantDoseAlgorithm(dose=0.0),
        sensor_model=SensorModel(noise_std=0.0, seed=2),
    )

    assert simulator._update_glucose_trend(0.0, 120.0) == 0.0
    assert simulator._update_glucose_trend(5.0, 104.0) == 0.0
    trend = simulator._update_glucose_trend(10.0, 120.0)

    assert abs(trend) < 0.5
