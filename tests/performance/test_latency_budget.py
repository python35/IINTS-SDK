from __future__ import annotations

import time

import pytest

from iints.core.algorithms.pid_controller import PIDController
from iints.core.patient.models import PatientModel
from iints.core.simulator import Simulator
from iints.core.supervisor import IndependentSupervisor


@pytest.mark.performance
def test_supervisor_latency_budget_p95_p99() -> None:
    supervisor = IndependentSupervisor()
    latencies_ms = []
    for idx in range(1500):
        t0 = time.perf_counter()
        supervisor.evaluate_safety(
            current_glucose=120.0 + (idx % 7) - 3.0,
            proposed_insulin=0.6,
            current_time=float(idx * 5),
            current_iob=0.5,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_ms.sort()
    p95 = latencies_ms[int(0.95 * len(latencies_ms))]
    p99 = latencies_ms[int(0.99 * len(latencies_ms))]

    # Intentionally conservative thresholds to avoid flaky CI while still catching regressions.
    assert p95 < 2.5
    assert p99 < 4.0


@pytest.mark.performance
def test_simulator_performance_report_contains_budgeted_percentiles() -> None:
    simulator = Simulator(
        patient_model=PatientModel(initial_glucose=120.0),
        algorithm=PIDController(),
        time_step=5,
        seed=42,
        enable_profiling=True,
    )
    _, safety_report = simulator.run(duration_minutes=180)
    perf = safety_report.get("performance_report", {})
    supervisor = perf.get("supervisor_latency_ms", {})
    step = perf.get("step_latency_ms", {})

    assert supervisor
    assert step
    assert "p95_ms" in supervisor and "p99_ms" in supervisor
    assert "p95_ms" in step and "p99_ms" in step
    assert supervisor["p95_ms"] < 5.0
    assert step["p95_ms"] < 20.0
