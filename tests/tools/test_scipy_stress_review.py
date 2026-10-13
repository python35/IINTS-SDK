"""Checks for evidence integrity in the offline stress-review runner."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from iints.core.simulator import Simulator
from iints.core.patient.bergman_model import BergmanPatientModel
from iints.core.algorithms.mock_algorithms import ConstantDoseAlgorithm


def load_runner():
    path = Path(__file__).resolve().parents[2] / "tools/analysis/scipy_stress_review.py"
    spec = importlib.util.spec_from_file_location("scipy_stress_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_endpoint_and_termination_are_not_confused():
    runner = load_runner()
    frame, report, elapsed = runner.simulate(runner.cohort(4, 3).iloc[0], "baseline", 3, minutes=60)
    summary = runner.summarize(frame, report, elapsed, 60, 5)
    assert summary["completed"]
    assert summary["coverage_pct"] == 100
    assert summary["steps"] == 12
    partial = runner.summarize(frame.iloc[:-1], {}, elapsed, 60, 5)
    assert not partial["completed"]
    assert partial["coverage_pct"] < 100
    stopped = runner.summarize(frame, {"terminated_early": True}, elapsed, 60, 5)
    assert not stopped["completed"]
    assert np.isclose(summary["tir_pct"] + summary["below70_pct"] + summary["above180_pct"], 100)


@pytest.mark.parametrize("minutes,dt", [(123, 5), (3, 5), (12, 2.5)])
def test_partial_last_interval_is_counted_as_completed(minutes, dt):
    simulator = Simulator(patient_model=BergmanPatientModel(initial_glucose=140),
                          algorithm=ConstantDoseAlgorithm(dose=0.05), time_step=dt)
    frame, report = simulator.run_batch(minutes)
    runner = load_runner()
    summary = runner.summarize(frame, report, 0, minutes, dt)
    assert summary["completed"]
    assert summary["coverage_pct"] == 100
    assert summary["expected_steps"] == len(frame)
    partial = runner.summarize(frame.iloc[:-1], {}, 0, minutes, dt)
    assert not partial["completed"]


def test_independent_oracle_converges_when_step_is_refined():
    runner = load_runner()
    _, errors, cdf_error = runner.numerical_oracle()
    assert errors.max_abs_error_mgdl.is_monotonic_increasing
    assert errors.max_abs_error_mgdl.iloc[0] < .05
    assert errors.max_abs_error_mgdl.iloc[-1] > .5
    assert cdf_error < 1e-12


def test_sampling_is_reproducible_and_covers_each_lhs_stratum():
    runner = load_runner()
    n = 16
    data = runner.cohort(n, 42)
    assert data.equals(runner.cohort(n, 42))
    assert not data.equals(runner.cohort(n, 43))
    for name, (low, high) in runner.RANGES.items():
        strata = np.floor((data[name].to_numpy() - low) / (high - low) * n).astype(int)
        assert sorted(strata) == list(range(n))


def test_safety_boundary_and_stateful_checks_pass():
    runner = load_runner()
    sweep, invalid = runner.safety_sweep(50, 9)
    assert sweep.passed.all()
    assert invalid.rejected.all()
    assert len(invalid) == 15
