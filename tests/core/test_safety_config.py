from iints.core.safety import SafetyConfig
from iints.core.safety.input_validator import InputValidator
from iints.core.supervisor import IndependentSupervisor
from iints.core.simulator import Simulator
from iints.core.patient.models import PatientModel
from iints.core.algorithms.mock_algorithms import ConstantDoseAlgorithm


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
