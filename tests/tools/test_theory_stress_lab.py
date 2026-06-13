from __future__ import annotations

import json

from iints.tools.theory_stress_lab import run_theory_stress_lab


def test_theory_stress_lab_writes_core_artifacts(tmp_path):
    output_dir = tmp_path / "theory_stress"

    report = run_theory_stress_lab(output_dir=output_dir, profile="ci", seed=3, repeats=1)

    assert report.checks
    assert {check.code for check in report.checks} >= {
        "no_negative_states",
        "hypo_blocks_insulin",
        "iob_limits_bolus",
        "pump_failure_raises_ffa_ketones",
        "sensor_lag_is_bounded",
        "exercise_does_not_create_impossible_crash",
        "meal_response_has_plausible_peak",
        "illness_increases_insulin_need_without_exploding",
    }
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "checks.json").is_file()
    assert (output_dir / "weakness_rankings.csv").is_file()

    payload = json.loads((output_dir / "checks.json").read_text())
    assert payload["profile"] == "ci"
    assert len(payload["checks"]) == len(report.checks)
    assert "Weakness Ranking" in (output_dir / "summary.md").read_text()


def test_theory_stress_lab_repeats_are_ranked(tmp_path):
    output_dir = tmp_path / "theory_stress_repeats"

    report = run_theory_stress_lab(output_dir=output_dir, profile="ci", seed=5, repeats=2)

    assert len(report.checks) == 16
    assert any("#r2" in check.code for check in report.checks)
    ranking = (output_dir / "weakness_rankings.csv").read_text()
    assert "rank,code,status,score" in ranking
