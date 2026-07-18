import pytest

from iints.core.patient.models import CustomPatientModel
from iints.core.patient.patient_factory import PatientFactory


def test_custom_patient_ignores_disabled_bergman_research_defaults() -> None:
    patient = PatientFactory.create_patient(
        patient_type="custom",
        stem_cell_engraftment_percent=0.0,
        stem_cell_subq_fraction=0.0,
        immune_rejection_rate=0.0,
    )

    assert isinstance(patient, CustomPatientModel)


def test_custom_patient_rejects_enabled_bergman_research_parameters() -> None:
    with pytest.raises(ValueError, match="Select patient_type='bergman'"):
        PatientFactory.create_patient(
            patient_type="custom",
            stem_cell_engraftment_percent=50.0,
        )
