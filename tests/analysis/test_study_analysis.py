from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from iints.analysis.study_analysis import analyze_study_directory, compare_studies
from iints.cli.cli import app


runner = CliRunner()


def _write_run(
    run_dir: Path,
    *,
    run_id: str,
    algorithm: str,
    glucose: list[float],
    interventions: int,
    grade: str | None,
    nested_config: bool = False,
    algorithm_role: str | None = None,
    profile_id: str = "clinic_safe_baseline",
    study_arm: str = "clean_certified",
    condition_group: str = "clean_certified",
    scenario_slug: str = "baseline_day",
    supervisor_enabled: bool = True,
) -> None:
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "time_minutes": [idx * 5 for idx in range(len(glucose))],
            "glucose_actual_mgdl": glucose,
            "predicted_glucose_ai_30min": [value + 5.0 for value in glucose],
            "predictor_uncertainty_std_mgdl": [2.0 + idx for idx in range(len(glucose))],
            "safety_triggered": [False] * len(glucose),
        }
    )
    df.to_csv(run_dir / "results.csv", index=False)
    config_payload = (
        {
            "algorithm": {
                "class": f"algorithms.{algorithm}",
                "metadata": {"name": algorithm},
            },
            "duration_minutes": len(glucose) * 5,
            "seed": 42,
            "study_condition": study_arm,
            "condition_group": condition_group,
            "algorithm_role": algorithm_role,
            "profile_id": profile_id,
            "scenario_slug": scenario_slug,
            "supervisor_enabled": supervisor_enabled,
            "scenario": {
                "scenario_name": run_dir.name,
                "condition_group": condition_group,
                "study_arm": study_arm,
                "scenario_slug": scenario_slug,
                "supervisor_enabled": supervisor_enabled,
            },
        }
        if nested_config
        else {
            "algorithm": algorithm,
            "duration_minutes": len(glucose) * 5,
            "scenario_name": run_dir.name,
            "study_condition": study_arm,
            "condition_group": condition_group,
            "algorithm_role": algorithm_role,
            "profile_id": profile_id,
            "scenario_slug": scenario_slug,
            "supervisor_enabled": supervisor_enabled,
        }
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": 42,
                "algorithm_id": algorithm.lower().replace(" ", "_"),
                "algorithm_role": algorithm_role,
                "profile_id": profile_id,
                "config": config_payload,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "audit" / "audit_summary.json").write_text(
        json.dumps({"bolus_interventions_count": interventions, "terminated_early": False}),
        encoding="utf-8",
    )
    (run_dir / "baseline" / "baseline_comparison.json").write_text(
        json.dumps(
            {
                "reference": "Standard PID",
                "rows": [
                    {"algorithm": algorithm, "tir_70_180": 90.0, "bolus_interventions": interventions},
                    {"algorithm": "Standard PID", "tir_70_180": 85.0, "bolus_interventions": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    if grade is not None:
        (run_dir / "certification.json").write_text(
            json.dumps(
                {
                    "mdmp_grade": grade,
                    "certified_for_medical_research": grade in {"research_grade", "clinical_grade", "ai_ready"},
                }
            ),
            encoding="utf-8",
        )


def test_analyze_study_directory_aggregates_runs(tmp_path) -> None:
    study_dir = tmp_path / "study"
    _write_run(study_dir / "run_1", run_id="run-1", algorithm="AlgoA", glucose=[110.0, 120.0, 130.0], interventions=2, grade="research_grade")
    _write_run(study_dir / "run_2", run_id="run-2", algorithm="AlgoA", glucose=[190.0, 200.0, 210.0], interventions=4, grade=None)

    summary = analyze_study_directory(study_dir)
    payload = summary.to_dict()

    assert payload["run_count"] == 2
    assert payload["aggregate"]["mean_supervisor_interventions"] == 3.0
    assert payload["aggregate_stats"]["tir_70_180"]["count"] == 2
    assert payload["certification_comparison"]["certified_runs"] == 1
    assert payload["baseline_summary"]["mean_tir_70_180_by_algorithm"]["AlgoA"] == 90.0
    assert "failure_analysis" in payload
    assert "by_algorithm" in payload
    assert "safety_summary" in payload
    assert payload["calibration_summary"]["overall"]["run_count"] == 2
    assert payload["uncertainty_summary"]["overall"]["count"] == 2
    assert "uncertainty_vs_error" in payload["uncertainty_summary"]
    assert payload["uncertainty_summary"]["uncertainty_vs_error"]["overall"]["run_count"] == 2


def test_analyze_study_directory_parses_nested_run_metadata_shape(tmp_path) -> None:
    study_dir = tmp_path / "study"
    _write_run(
        study_dir / "run_1",
        run_id="run-1",
        algorithm="AlgoNested",
        glucose=[100.0, 110.0, 120.0],
        interventions=2,
        grade="research_grade",
        nested_config=True,
    )

    payload = analyze_study_directory(study_dir).to_dict()
    assert payload["runs"][0]["algorithm"] == "AlgoNested"
    assert payload["runs"][0]["condition_group"] == "clean_certified"


def test_analyze_study_directory_adds_subgroup_and_pairwise_summaries(tmp_path) -> None:
    study_dir = tmp_path / "study"
    _write_run(
        study_dir / "candidate",
        run_id="run-candidate",
        algorithm="AlgoA",
        glucose=[110.0, 118.0, 126.0],
        interventions=2,
        grade="research_grade",
        algorithm_role="candidate",
        profile_id="clinic_safe_baseline",
        study_arm="clean_certified",
        condition_group="clean_certified",
        scenario_slug="baseline_day",
    )
    _write_run(
        study_dir / "baseline_pid",
        run_id="run-pid",
        algorithm="PID Controller",
        glucose=[120.0, 128.0, 136.0],
        interventions=3,
        grade="research_grade",
        algorithm_role="baseline",
        profile_id="clinic_safe_baseline",
        study_arm="clean_certified",
        condition_group="clean_certified",
        scenario_slug="baseline_day",
    )
    _write_run(
        study_dir / "baseline_pump",
        run_id="run-pump",
        algorithm="Standard Pump",
        glucose=[130.0, 138.0, 146.0],
        interventions=4,
        grade=None,
        algorithm_role="baseline",
        profile_id="clinic_safe_baseline",
        study_arm="clean_certified",
        condition_group="clean_certified",
        scenario_slug="baseline_day",
    )

    payload = analyze_study_directory(study_dir).to_dict()

    assert payload["by_algorithm"]["AlgoA"]["run_count"] == 1
    assert payload["by_profile"]["clinic_safe_baseline"]["run_count"] == 3
    assert payload["by_arm"]["clean_certified"]["run_count"] == 3
    assert payload["by_scenario"]["baseline_day"]["run_count"] == 3
    assert payload["pairwise_baseline_deltas"]["candidate_algorithm"] == "AlgoA"
    assert "PID Controller" in payload["pairwise_baseline_deltas"]["baselines"]
    assert payload["safety_summary"]["mean_interventions_by_algorithm"]["Standard Pump"] == 4.0
    assert payload["safety_summary"]["supervisor_on_vs_off"]["supervisor_on_runs"] == 3
    assert "by_algorithm" in payload["uncertainty_summary"]["uncertainty_vs_error"]


def test_cli_analyze_writes_json_and_markdown(tmp_path) -> None:
    study_dir = tmp_path / "study"
    summary_json = tmp_path / "study_summary.json"
    summary_md = tmp_path / "study_summary.md"
    summary_csv = tmp_path / "evidence.csv"
    evidence_md = tmp_path / "evidence.md"
    _write_run(study_dir / "run_1", run_id="run-1", algorithm="AlgoA", glucose=[110.0, 115.0, 120.0], interventions=1, grade="clinical_grade")
    _write_run(study_dir / "run_2", run_id="run-2", algorithm="AlgoB", glucose=[150.0, 160.0, 170.0], interventions=3, grade=None)

    result = runner.invoke(
        app,
        [
            "analyze",
            str(study_dir),
            "--output-json",
            str(summary_json),
            "--output-markdown",
            str(summary_md),
            "--output-csv",
            str(summary_csv),
            "--output-evidence-markdown",
            str(evidence_md),
        ],
    )

    assert result.exit_code == 0
    assert summary_json.is_file()
    assert summary_md.is_file()
    assert summary_csv.is_file()
    assert evidence_md.is_file()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
    assert "IINTS Study Summary" in summary_md.read_text(encoding="utf-8")
    assert "quality_badges" in summary_csv.read_text(encoding="utf-8")
    assert "Failure Analysis" in summary_md.read_text(encoding="utf-8")


def test_compare_studies_reports_delta(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left / "run_1", run_id="left-1", algorithm="AlgoA", glucose=[110.0, 115.0, 120.0], interventions=1, grade="clinical_grade", algorithm_role="candidate")
    _write_run(right / "run_1", run_id="right-1", algorithm="AlgoA", glucose=[210.0, 215.0, 220.0], interventions=5, grade=None, algorithm_role="candidate")

    comparison = compare_studies(left, right).to_dict()
    assert comparison["delta"]["mean_supervisor_interventions"] == -4.0
    assert comparison["delta"]["certified_runs"] == 1.0
    assert "effect_estimates" in comparison
    assert "tir_70_180" in comparison["effect_estimates"]
    assert "AlgoA" in comparison["by_algorithm"]


def test_cli_compare_study_and_poster_study(tmp_path) -> None:
    study_dir = tmp_path / "study"
    summary_json = tmp_path / "study_summary.json"
    comparison_json = tmp_path / "comparison.json"
    poster_png = tmp_path / "study_poster.png"
    _write_run(study_dir / "run_1", run_id="run-1", algorithm="AlgoA", glucose=[110.0, 115.0, 120.0], interventions=1, grade="clinical_grade")
    _write_run(study_dir / "run_2", run_id="run-2", algorithm="AlgoA", glucose=[150.0, 155.0, 160.0], interventions=2, grade=None)
    runner.invoke(app, ["analyze", str(study_dir), "--output-json", str(summary_json)])

    compare_result = runner.invoke(app, ["compare-study", str(study_dir), str(study_dir), "--output-json", str(comparison_json)])
    poster_result = runner.invoke(app, ["poster-study", str(summary_json), "--output-path", str(poster_png)])

    assert compare_result.exit_code == 0
    assert poster_result.exit_code == 0
    assert comparison_json.is_file()
    assert poster_png.is_file()


def test_cli_analyze_supports_carelink_reference_metrics(tmp_path) -> None:
    study_dir = tmp_path / "study"
    metrics_path = tmp_path / "carelink_metrics.json"
    summary_json = tmp_path / "study_summary.json"
    _write_run(study_dir / "run_1", run_id="run-1", algorithm="AlgoA", glucose=[130.0, 140.0, 150.0], interventions=2, grade="clinical_grade")
    metrics_path.write_text(
        json.dumps(
            {
                "mean_glucose_mgdl": 140.0,
                "cv_pct": 25.0,
                "time_in_range_70_180_pct": 80.0,
                "time_below_70_pct": 2.0,
                "time_above_180_pct": 18.0,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "analyze",
            str(study_dir),
            "--output-json",
            str(summary_json),
            "--carelink-metrics",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["external_validation"]["reference_path"].endswith("carelink_metrics.json")
