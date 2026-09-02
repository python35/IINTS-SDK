"""Tests for the compartment/flux schema and its export into run results.

The point of these tests is the coupling between three things that are easy to
let drift apart: the declared schema, the ODE terms that are actually recorded,
and the columns that reach results.csv. A schema entry with no numeric source,
or a recorded term nobody declared, is the failure mode being guarded here.
"""

import json

import numpy as np
import pytest

from iints.core.patient.bergman_model import BergmanPatientModel
from iints.core.patient.compartments import (
    BERGMAN_COMPARTMENTS,
    BERGMAN_FLUXES,
    HOVORKA_COMPARTMENTS,
    HOVORKA_FLUXES,
    compartment_schema,
    schema_for_model,
)
from iints.core.patient.hovorka_model import HovorkaPatientModel

MODELS = [
    pytest.param(HovorkaPatientModel, HOVORKA_COMPARTMENTS, HOVORKA_FLUXES, id="hovorka"),
    pytest.param(BergmanPatientModel, BERGMAN_COMPARTMENTS, BERGMAN_FLUXES, id="bergman"),
]


def _drive(model, steps=8, meal_step=2, glucagon_step=5):
    """Run a few steps with a meal and a glucagon dose so fluxes are non-trivial."""

    for i in range(steps):
        model.update(
            5.0,
            0.6,
            carb_intake=(60.0 if i == meal_step else 0.0),
            delivered_glucagon_mg=(0.15 if i == glucagon_step else 0.0),
            current_time=float(i * 5),
        )
    return model


@pytest.mark.parametrize("cls,compartments,fluxes", MODELS)
def test_state_indices_are_a_permutation_of_the_state_vector(cls, compartments, fluxes):
    model = cls()
    indices = sorted(item.state_index for item in compartments)
    assert indices == list(range(len(model._state))), (
        "every ODE state must be described exactly once, or the export mislabels columns"
    )


@pytest.mark.parametrize("cls,compartments,fluxes", MODELS)
def test_flux_endpoints_reference_declared_compartments(cls, compartments, fluxes):
    keys = {item.key for item in compartments}
    for flux in fluxes:
        for endpoint in (flux.source, flux.target):
            # None is the boundary: infusion from outside, elimination to outside.
            assert endpoint is None or endpoint in keys, f"{flux.key} points at {endpoint}"


@pytest.mark.parametrize("cls,compartments,fluxes", MODELS)
def test_recorded_fluxes_match_what_the_ode_reports(cls, compartments, fluxes):
    model = _drive(cls(initial_glucose=140.0))
    reported = set(model.flux_snapshot())
    declared = {flux.key for flux in fluxes if flux.recorded}
    assert reported == declared, (
        f"schema and ODE disagree; only in ODE: {sorted(reported - declared)}, "
        f"only in schema: {sorted(declared - reported)}"
    )


@pytest.mark.parametrize("cls,compartments,fluxes", MODELS)
def test_unrecorded_fluxes_are_discrete_events(cls, compartments, fluxes):
    for flux in fluxes:
        if not flux.recorded:
            # A term with no numeric value must say why, so consumers do not
            # read its absence as a zero rate.
            assert flux.description, f"{flux.key} is unrecorded without explanation"


@pytest.mark.parametrize("cls,compartments,fluxes", MODELS)
def test_snapshot_does_not_perturb_the_trajectory(cls, compartments, fluxes):
    plain = _drive(cls(initial_glucose=140.0))
    observed = cls(initial_glucose=140.0)
    for i in range(8):
        observed.update(
            5.0,
            0.6,
            carb_intake=(60.0 if i == 2 else 0.0),
            delivered_glucagon_mg=(0.15 if i == 5 else 0.0),
            current_time=float(i * 5),
        )
        observed.flux_snapshot()
    assert np.array_equal(plain._state, observed._state), (
        "recording fluxes must be a read-only observation of the integration"
    )


@pytest.mark.parametrize("cls,compartments,fluxes", MODELS)
def test_compartment_state_is_finite_and_complete(cls, compartments, fluxes):
    model = _drive(cls(initial_glucose=140.0))
    state = model.get_compartment_state()
    assert set(state) == {item.key for item in compartments}
    assert all(np.isfinite(value) for value in state.values())
    fluxes_now = model.flux_snapshot()
    assert all(np.isfinite(value) for value in fluxes_now.values())


def test_bergman_recorded_fluxes_reconstruct_the_glucose_derivative():
    """The recorded terms must add up to dG/dt, not merely be plausible numbers.

    Compared against a finite difference of the glucose trajectory, so the
    tolerance covers the difference between an end-of-step instantaneous rate
    and an average over the step -- it is not a tolerance on the equations.
    """

    model = BergmanPatientModel(initial_glucose=110.0)
    times, glucose, records = [], [], []
    for i in range(72):
        model.update(
            5.0,
            6.0 if i == 6 else 0.6,
            carb_intake=(75.0 if i == 6 else 0.0),
            current_time=float(i * 5),
        )
        times.append(i * 5.0)
        glucose.append(model.get_compartment_state()["G"])
        records.append(model.flux_snapshot())

    glucose_array = np.asarray(glucose)
    assert glucose_array.max() - glucose_array.min() > 30.0, "trace too flat to test anything"

    def term(key):
        return np.asarray([record[key] for record in records])

    rhs = (
        term("glucose_appearance")
        + term("basal_production")
        + term("dawn_flux")
        - term("glucose_uptake")
        - term("renal_clearance")
        - term("exercise_uptake")
    )
    derivative = np.gradient(glucose_array, np.asarray(times))
    residual = np.median(np.abs(derivative - rhs))
    assert residual < 0.05 * np.abs(derivative).max()


@pytest.mark.parametrize("cls,compartments,fluxes", MODELS)
def test_schema_for_model_round_trips_through_json(cls, compartments, fluxes):
    payload = schema_for_model(cls())
    assert payload is not None
    restored = json.loads(json.dumps(payload))
    assert len(restored["compartments"]) == len(compartments)
    assert len(restored["fluxes"]) == len(fluxes)


def test_schema_for_model_returns_none_without_a_published_schema():
    class UnknownBackend:
        pass

    assert schema_for_model(UnknownBackend()) is None


def test_compartment_schema_rejects_unknown_model():
    with pytest.raises(KeyError):
        compartment_schema("no_such_model")
