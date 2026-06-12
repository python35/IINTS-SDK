from __future__ import annotations

from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel
from iints.core.patient.patient_factory import PatientFactory


def test_advanced_metabolic_model_update_runs_with_18_states():
    patient = AdvancedMetabolicModel(initial_glucose=120.0, initial_beta_mass=0.0)

    glucose = patient.update(
        5.0,
        delivered_insulin=0.8,
        carb_intake=10.0,
        fat_intake=5.0,
        protein_intake=8.0,
        current_time=0.0,
    )

    assert len(patient._state) == 18
    assert glucose >= 20.0
    state = patient.get_patient_state()
    assert state["fat_pool_g"] > 0.0
    assert state["protein_pool_g"] > 0.0


def test_advanced_metabolic_model_backward_compatible_aliases():
    patient = AdvancedMetabolicModel(initial_glucose=120.0)

    glucose = patient.update(
        dt_minutes=5.0,
        delivered_insulin=0.8,
        delivered_glucagon=0.0,
        current_time_minutes=15.0,
    )

    assert glucose >= 20.0


def test_advanced_metabolic_model_state_roundtrip_includes_new_flags():
    patient = AdvancedMetabolicModel(initial_glucose=120.0)
    patient.start_illness(0.7)
    patient.start_menstrual_cycle(123.0)
    patient.update(5.0, delivered_insulin=0.8, fat_intake=3.0, protein_intake=4.0)

    restored = AdvancedMetabolicModel(initial_glucose=120.0)
    restored.set_state(patient.get_state())

    assert restored.is_ill is True
    assert restored.illness_severity == 0.7
    assert restored.menstrual_cycle_active is True
    assert restored.cycle_start_time_minutes == 123.0
    assert restored.pump_cannula_age_minutes == patient.pump_cannula_age_minutes
    assert len(restored._state) == 18


def test_patient_factory_can_create_advanced_metabolic_model():
    patient = PatientFactory.create_patient(patient_type="advanced", initial_glucose=130.0)

    assert isinstance(patient, AdvancedMetabolicModel)
    assert len(patient._state) == 18
