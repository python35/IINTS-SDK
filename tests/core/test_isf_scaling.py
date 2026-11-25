"""Executable proofs for the equilibrium-consistent Hovorka ISF parameterisation.

Each test pins one mathematical claim about ``_basal_parameter_scale``, fasting
drift, realised ISF, and the single-knob repair in ``iints.core.patient.equilibrium``.
"""

from __future__ import annotations

import numpy as np
import pytest

from iints.core.control.linear_plant import linearise
from iints.core.patient import equilibrium as pf
from iints.core.patient.calibration import measure_insulin_sensitivity_factor
from iints.core.patient.hovorka_model import HovorkaPatientModel

REFERENCE_GLUCOSE = 146.0
BASAL_RATES = (0.5, 1.0, 1.5, 2.0, 3.0)


def _fasting_drift_mgdl_per_h(patient: HovorkaPatientModel) -> float:
    """dG/dt at the state the patient reports as its own operating point."""
    state = np.asarray(patient.get_state()["ode_state"], dtype=float)
    insulin_mu_per_min = float(patient.basal_insulin_rate) * 1000.0 / 60.0
    derivative = np.asarray(patient._ode(0.0, state, insulin_mu_per_min, 0.0, 0.0), dtype=float)
    return float(derivative[0]) / patient._glucose_volume_dl() * 60.0


@pytest.mark.parametrize("basal", BASAL_RATES)
def test_basal_scale_times_basal_is_invariant(basal: float) -> None:
    """sigma_b ~ 1/basal exactly, so sensitivity is fixed once basal is chosen."""
    product = pf.basal_scale(basal, REFERENCE_GLUCOSE) * basal
    assert product == pytest.approx(0.35739, rel=1e-3)


def test_declared_operating_point_is_an_equilibrium_only_at_the_reference_isf() -> None:
    at_reference = HovorkaPatientModel(
        initial_glucose=REFERENCE_GLUCOSE,
        basal_insulin_rate=2.0,
        insulin_sensitivity=50.0,
    )
    assert abs(_fasting_drift_mgdl_per_h(at_reference)) < 1e-9

    as_the_pipeline_ran = HovorkaPatientModel(
        initial_glucose=REFERENCE_GLUCOSE,
        basal_insulin_rate=2.0,
        insulin_sensitivity=16.35,
    )
    assert _fasting_drift_mgdl_per_h(as_the_pipeline_ran) == pytest.approx(31.83, abs=0.1)


@pytest.mark.parametrize("basal", BASAL_RATES)
def test_the_drift_does_not_depend_on_the_basal_rate(basal: float) -> None:
    """Confirms the clinical multiplier, not the operating point, breaks the balance."""
    patient = HovorkaPatientModel(
        initial_glucose=REFERENCE_GLUCOSE,
        basal_insulin_rate=basal,
        insulin_sensitivity=16.35,
    )
    assert _fasting_drift_mgdl_per_h(patient) == pytest.approx(31.83, abs=0.1)


def test_realised_isf_is_not_the_configured_isf() -> None:
    """insulin_sensitivity=50 realises ~15.8 mg/dL/U at a 2.0 U/h basal rate."""
    realised = measure_insulin_sensitivity_factor(
        lambda: HovorkaPatientModel(
            initial_glucose=REFERENCE_GLUCOSE,
            basal_insulin_rate=2.0,
            insulin_sensitivity=50.0,
        ),
        2.0,
        bolus_units=1.0,
        hours=8.0,
    )
    assert realised == pytest.approx(15.75, abs=0.2)
    assert realised < 0.4 * 50.0


def test_realised_isf_times_basal_is_approximately_constant() -> None:
    """ISF_realised ~ K / basal, the clinical reading of the invariant above."""
    products = [
        measure_insulin_sensitivity_factor(
            lambda b=basal: HovorkaPatientModel(
                initial_glucose=REFERENCE_GLUCOSE,
                basal_insulin_rate=b,
                insulin_sensitivity=50.0,
            ),
            basal,
            bolus_units=1.0,
            hours=8.0,
        )
        * basal
        for basal in (1.0, 1.5, 2.0, 2.5, 3.0)
    ]
    assert np.ptp(products) < 0.9
    assert np.mean(products) == pytest.approx(31.4, abs=0.5)


def test_the_configuration_the_pipeline_used_balances_at_330_not_146() -> None:
    """The anchor of the white-box forecast was 184 mg/dL away from the truth."""
    sigma = pf.basal_scale(2.0, REFERENCE_GLUCOSE) * (16.35 / 50.0)
    assert pf.equilibrium_glucose(2.0, sigma) == pytest.approx(329.9, abs=1.0)


@pytest.mark.parametrize("sigma", (0.05, 0.08, 0.12, 0.15))
def test_equilibrium_patient_has_no_fasting_drift(sigma: float) -> None:
    patient = pf.make_equilibrium_patient(2.0, sigma)
    assert abs(_fasting_drift_mgdl_per_h(patient)) < 1e-9


def test_equilibrium_glucose_inverts_the_basal_scale() -> None:
    """The two solvers are inverses, which is what makes sigma a single knob."""
    for basal in BASAL_RATES:
        sigma = pf.basal_scale(basal, REFERENCE_GLUCOSE)
        assert pf.equilibrium_glucose(basal, sigma) == pytest.approx(REFERENCE_GLUCOSE, abs=0.05)


def test_an_unreachable_operating_point_raises_instead_of_clipping() -> None:
    """Too much sensitivity for the basal rate raises ValueError instead of clipping."""
    too_sensitive = pf.basal_scale(2.0, REFERENCE_GLUCOSE) * 1.5
    with pytest.raises(ValueError, match="no fasting equilibrium"):
        pf.equilibrium_glucose(2.0, too_sensitive)


def test_dc_gain_stays_negative_and_finite_across_the_admissible_range() -> None:
    """More insulin must lower glucose at every admissible sensitivity."""
    for sigma in (0.05, 0.08, 0.12, 0.15, 0.178):
        plant = linearise(pf.make_equilibrium_patient(2.0, sigma))
        gain = plant.dc_gain_per_u_per_h()
        assert np.isfinite(gain) and gain < 0.0
        assert plant.is_stable()
