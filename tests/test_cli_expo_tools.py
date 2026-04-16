from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_scenarios_export_study_pack_writes_manifest(tmp_path) -> None:
    output_dir = tmp_path / "study_pack"

    result = runner.invoke(
        app,
        ["scenarios", "export-study-pack", "--output-dir", str(output_dir), "--seeds", "1,2,3"],
    )

    assert result.exit_code == 0
    manifest = output_dir / "study_pack_manifest.json"
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["seeds"] == [1, 2, 3]
    assert (output_dir / "baseline_day.json").is_file()


def test_scenarios_export_study_pack_supports_eucys_preset(tmp_path) -> None:
    output_dir = tmp_path / "eucys_pack"

    result = runner.invoke(
        app,
        ["scenarios", "export-study-pack", "--output-dir", str(output_dir), "--preset", "eucys", "--seeds", "1,2"],
    )

    assert result.exit_code == 0
    manifest = json.loads((output_dir / "study_pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preset"] == "eucys"
    assert (output_dir / "eucys_study_matrix.csv").is_file()


def test_demo_expo_writes_bundle_summary(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "expo_demo"

    def _fake_booth_demo(output_dir, **kwargs):
        target = output_dir
        target.mkdir(parents=True, exist_ok=True)
        return {"poster_png": str(target / "booth_demo_poster.png")}

    class _FakeSummary:
        def to_dict(self):
            return {
                "study_dir": str(output_dir),
                "run_count": 2,
                "aggregate": {
                    "mean_tir_70_180": 82.0,
                    "mean_supervisor_interventions": 3.0,
                    "mean_glucose": 145.0,
                    "mean_cv": 28.0,
                    "run_count": 2,
                },
                "certification_comparison": {
                    "certified_runs": 1,
                    "uncertified_runs": 1,
                    "mean_tir_70_180_certified": 85.0,
                    "mean_tir_70_180_uncertified": 79.0,
                    "mean_supervisor_interventions_certified": 2.0,
                    "mean_supervisor_interventions_uncertified": 4.0,
                    "tir_delta_certified_minus_uncertified": 6.0,
                },
                "baseline_summary": {
                    "mean_tir_70_180_by_algorithm": {"AlgoA": 82.0},
                    "mean_bolus_interventions_by_algorithm": {"AlgoA": 3.0},
                    "run_quality_badge_counts": {"strong_tir": 2},
                },
                "evidence_rows": [
                    {
                        "scenario_name": "Run 1",
                        "algorithm": "AlgoA",
                        "seed": 1,
                        "tir_70_180": 82.0,
                        "tir_below_70": 1.0,
                        "tir_above_180": 17.0,
                        "mean_glucose": 145.0,
                        "cv": 28.0,
                        "gmi": 6.8,
                        "supervisor_interventions": 3.0,
                        "certification_grade": "research_grade",
                        "quality_badges": "strong_tir,stable_variability",
                    }
                ],
                "runs": [
                    {
                        "scenario_name": "Run 1",
                        "run_id": "run-1",
                        "algorithm": "AlgoA",
                        "certification_grade": "research_grade",
                        "quality_badges": ["strong_tir"],
                        "metrics": {
                            "tir_70_180": 82.0,
                            "supervisor_interventions": 3.0,
                        },
                    }
                ],
            }

    def _fake_analyze_study_directory(path):
        return _FakeSummary()

    def _fake_generate_study_poster(summary, **kwargs):
        output = output_dir / "study_poster.png"
        output.write_text("png", encoding="utf-8")
        summary_json = output_dir / "study_poster.json"
        summary_json.write_text("{}", encoding="utf-8")
        return {"poster_png": str(output), "poster_summary_json": str(summary_json)}

    def _fake_protocol_bundle(output_dir, **kwargs):
        target = output_dir
        target.mkdir(parents=True, exist_ok=True)
        markdown = target / "STUDY_PROTOCOL.md"
        markdown.write_text("# protocol", encoding="utf-8")
        design = target / "study_design.json"
        design.write_text("{}", encoding="utf-8")
        matrix = target / "study_matrix.csv"
        matrix.write_text("scenario,algorithm,seed\n", encoding="utf-8")
        algorithms = target / "algorithms.json"
        algorithms.write_text("[]", encoding="utf-8")
        return {
            "protocol_markdown": str(markdown),
            "study_design_json": str(design),
            "study_matrix_csv": str(matrix),
            "algorithms_json": str(algorithms),
        }

    monkeypatch.setattr("iints.cli.cli.build_booth_demo", _fake_booth_demo)
    monkeypatch.setattr("iints.cli.cli.analyze_study_directory", _fake_analyze_study_directory)
    monkeypatch.setattr("iints.cli.cli.generate_study_poster", _fake_generate_study_poster)
    monkeypatch.setattr("iints.cli.cli.write_study_protocol_bundle", _fake_protocol_bundle)

    result = runner.invoke(app, ["demo-expo", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "study_summary.json").is_file()
    assert (output_dir / "study_summary.md").is_file()
    assert (output_dir / "evidence_table.csv").is_file()
    assert (output_dir / "evidence_table.md").is_file()
    assert (output_dir / "study_poster.png").is_file()
    assert (output_dir / "protocol" / "STUDY_PROTOCOL.md").is_file()


def test_run_eucys_study_builds_scientific_bundle(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "eucys"
    algo_path = tmp_path / "algo.py"
    algo_path.write_text("class Dummy: pass\n", encoding="utf-8")

    def _fake_load_algorithm_instance_silent(_path):
        return object()

    def _fake_run_full(*, algorithm, scenario, patient_config, duration_minutes, time_step, seed, output_dir, safety_config=None, **kwargs):
        target = Path(output_dir)
        (target / "audit").mkdir(parents=True, exist_ok=True)
        (target / "baseline").mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "time_minutes": [0, 5, 10],
                "glucose_actual_mgdl": [110.0 + seed, 120.0 + seed, 130.0 + seed],
                "safety_triggered": [False, False, False],
            }
        )
        df.to_csv(target / "results.csv", index=False)
        (target / "run_metadata.json").write_text(
            json.dumps(
                {
                    "run_id": target.name,
                    "seed": seed,
                    "config": {
                        "algorithm": {
                            "class": "algorithms.DemoAlgorithm",
                            "metadata": {"name": "DemoAlgorithm"},
                        },
                        "duration_minutes": duration_minutes,
                        "scenario": scenario,
                    },
                }
            ),
            encoding="utf-8",
        )
        (target / "config.json").write_text(
            json.dumps(
                {
                    "algorithm": {
                        "class": "algorithms.DemoAlgorithm",
                        "metadata": {"name": "DemoAlgorithm"},
                    },
                    "duration_minutes": duration_minutes,
                    "scenario": scenario,
                }
            ),
            encoding="utf-8",
        )
        (target / "audit" / "audit_summary.json").write_text(
            json.dumps({"bolus_interventions_count": 1, "terminated_early": False}),
            encoding="utf-8",
        )
        (target / "baseline" / "baseline_comparison.json").write_text(
            json.dumps(
                {
                    "reference": "Standard PID",
                    "rows": [
                        {"algorithm": "DemoAlgorithm", "tir_70_180": 82.0, "bolus_interventions": 1},
                        {"algorithm": "Standard PID", "tir_70_180": 80.0, "bolus_interventions": 2},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"output_dir": str(target)}

    def _fake_prepare(_output_dir, _console):
        return None

    monkeypatch.setattr("iints.cli.cli._load_algorithm_instance_silent", _fake_load_algorithm_instance_silent)
    monkeypatch.setattr("iints.cli.cli._load_algorithm_plugin_instance", lambda _name: object())
    monkeypatch.setattr("iints.cli.cli._maybe_prepare_ai_artifacts", _fake_prepare)
    monkeypatch.setattr("iints.cli.cli.iints.run_full", _fake_run_full)

    result = runner.invoke(
        app,
        [
            "run-eucys-study",
            "--algo",
            str(algo_path),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "1,2",
            "--no-prepare-ai",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "protocol" / "STUDY_PROTOCOL.md").is_file()
    assert (output_dir / "scenarios" / "eucys_study_matrix.csv").is_file()
    assert (output_dir / "study_clean" / "study_summary.json").is_file()
    assert (output_dir / "study_corrupted" / "study_summary.json").is_file()
    assert (output_dir / "comparisons" / "clean_vs_corrupted.json").is_file()
    assert (output_dir / "EUCYS_SUMMARY.md").is_file()
    assert (output_dir / "EUCYS_RESULTS_TABLE.csv").is_file()
    assert (output_dir / "EUCYS_FIGURE_MANIFEST.json").is_file()
    assert (output_dir / "EUCYS_ABSTRACT_FILLED.md").is_file()
    assert (output_dir / "EUCYS_MAIN_FIGURE.png").is_file()
    assert (output_dir / "EUCYS_LIMITATIONS.md").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_REPRODUCIBILITY_BUNDLE.json").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_ABSTRACT_DRAFT.md").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_ABSTRACT_FILLED.md").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_POSTER_OUTLINE.md").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_JURY_QA.md").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_MAIN_FIGURE.png").is_file()


def test_run_study_builds_generic_scientific_bundle(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "study_bundle"
    algo_path = tmp_path / "algo.py"
    algo_path.write_text("class Dummy: pass\n", encoding="utf-8")

    def _fake_load_algorithm_instance_silent(_path):
        return object()

    def _fake_run_full(*, algorithm, scenario, patient_config, duration_minutes, time_step, seed, output_dir, safety_config=None, **kwargs):
        target = Path(output_dir)
        (target / "audit").mkdir(parents=True, exist_ok=True)
        (target / "baseline").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "time_minutes": [0, 5, 10],
                "glucose_actual_mgdl": [105.0 + seed, 115.0 + seed, 125.0 + seed],
                "predicted_glucose_ai_30min": [110.0 + seed, 120.0 + seed, 130.0 + seed],
                "predictor_uncertainty_std_mgdl": [2.0, 3.0, 4.0],
                "safety_triggered": [False, False, False],
            }
        ).to_csv(target / "results.csv", index=False)
        payload = {
            "algorithm": {
                "class": "algorithms.DemoAlgorithm",
                "metadata": {"name": "DemoAlgorithm"},
            },
            "duration_minutes": duration_minutes,
            "scenario": scenario,
        }
        (target / "run_metadata.json").write_text(
            json.dumps({"run_id": target.name, "seed": seed, "config": payload}),
            encoding="utf-8",
        )
        (target / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        (target / "audit" / "audit_summary.json").write_text(
            json.dumps({"bolus_interventions_count": 2, "terminated_early": False}),
            encoding="utf-8",
        )
        (target / "baseline" / "baseline_comparison.json").write_text(
            json.dumps(
                {
                    "reference": "PID Controller",
                    "rows": [
                        {"algorithm": "DemoAlgorithm", "tir_70_180": 82.0, "bolus_interventions": 2},
                        {"algorithm": "PID Controller", "tir_70_180": 79.0, "bolus_interventions": 3},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"output_dir": str(target)}

    monkeypatch.setattr("iints.cli.cli._load_algorithm_instance_silent", _fake_load_algorithm_instance_silent)
    monkeypatch.setattr("iints.cli.cli._load_algorithm_plugin_instance", lambda _name: object())
    monkeypatch.setattr("iints.cli.cli._maybe_prepare_ai_artifacts", lambda _output_dir, _console: None)
    monkeypatch.setattr("iints.cli.cli.iints.run_full", _fake_run_full)

    result = runner.invoke(
        app,
        [
            "run-study",
            "--algo",
            str(algo_path),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "1",
            "--no-prepare-ai",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "protocol" / "algorithms.json").is_file()
    assert (output_dir / "study_clean" / "study_summary.json").is_file()
    assert (output_dir / "study_corrupted" / "study_summary.json").is_file()
    assert (output_dir / "study_supervisor_off" / "study_summary.json").is_file()
    assert (output_dir / "comparisons" / "clean_vs_supervisor_off.json").is_file()


def test_cli_eucys_results_packages_existing_bundle(tmp_path) -> None:
    output_dir = tmp_path / "eucys_study"
    protocol_dir = output_dir / "protocol"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "STUDY_PROTOCOL.md").write_text("# protocol", encoding="utf-8")
    (protocol_dir / "study_matrix.csv").write_text("arm,seed\nclean,1\n", encoding="utf-8")
    (protocol_dir / "algorithms.json").write_text("[]", encoding="utf-8")
    (protocol_dir / "study_design.json").write_text(
        json.dumps(
            {
                "matrix_rows": [1],
                "profiles": [{"id": 1}],
                "scenarios": [{"id": 1}],
                "algorithms": [{"id": 1}, {"id": 2}],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "study_summary.json").write_text(json.dumps({"run_count": 3}), encoding="utf-8")

    summary_payload = {
        "run_count": 1,
        "aggregate": {
            "mean_tir_70_180": 82.0,
            "mean_tir_below_70": 2.0,
            "mean_tir_above_180": 18.0,
            "mean_glucose": 145.0,
            "mean_supervisor_interventions": 3.0,
        },
        "safety_summary": {
            "severe_hypo_run_count": 0,
            "terminated_early_run_count": 0,
        },
        "by_algorithm": {
            "CandidateAlgo": {
                "aggregate": {
                    "mean_tir_70_180": 82.0,
                    "mean_tir_below_70": 2.0,
                    "mean_tir_above_180": 18.0,
                    "mean_glucose": 145.0,
                    "mean_supervisor_interventions": 3.0,
                }
            },
            "PID Controller": {
                "aggregate": {
                    "mean_tir_70_180": 78.0,
                    "mean_tir_below_70": 2.5,
                    "mean_tir_above_180": 22.0,
                    "mean_glucose": 151.0,
                    "mean_supervisor_interventions": 4.0,
                }
            },
        },
        "pairwise_baseline_deltas": {
            "candidate_algorithm": "CandidateAlgo",
            "baselines": {
                "PID Controller": {
                    "mean_deltas": {
                        "tir_70_180": 4.0,
                    }
                }
            },
        },
    }
    for folder_name in ("study_clean", "study_corrupted", "study_supervisor_off"):
        arm_dir = output_dir / folder_name
        arm_dir.mkdir(parents=True)
        (arm_dir / "study_summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
        (arm_dir / "study_poster.png").write_text("png", encoding="utf-8")

    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True)
    (comparisons_dir / "clean_vs_corrupted.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["eucys-results", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_REPRODUCIBILITY_BUNDLE.json").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_LIMITATIONS.md").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_ABSTRACT_FILLED.md").is_file()
    assert (output_dir / "EUCYS_RESULTS" / "EUCYS_MAIN_FIGURE.png").is_file()
