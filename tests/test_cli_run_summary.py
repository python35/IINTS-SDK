from __future__ import annotations

from rich.console import Console

from iints.cli.cli import _print_run_summary


def test_run_summary_marks_early_termination(tmp_path) -> None:
    console = Console(record=True)

    _print_run_summary(
        console,
        algorithm_name="QuickstartAlgorithm",
        output_dir=tmp_path,
        metrics={
            "terminated_early": 1.0,
            "completed_duration_minutes": 560.0,
            "completion_ratio_pct": 5.56,
            "tir_70_180": 20.0,
            "tir_below_70": 80.0,
            "tir_above_180": 0.0,
            "mean_glucose": 70.0,
            "supervisor_interventions": 9.0,
        },
        duration_minutes=10080,
        seed=42,
        wall_seconds=1.2,
        termination_reason="glucose < 40 mg/dL for 30 minutes",
    )

    output = console.export_text()
    assert "Run Terminated Early" in output
    assert "Requested duration: 10080 min" in output
    assert "Completed duration: 560 min" in output
    assert "Completion ratio: 5.6%" in output
    assert "glucose < 40 mg/dL for 30 minutes" in output
