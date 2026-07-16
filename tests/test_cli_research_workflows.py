from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from iints.analysis.run_quality import build_result_quality_summary, write_run_quality_artifacts
from iints.cli.cli import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def _disable_local_ai_review_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IINTS_LOCAL_AI_REVIEW", "off")


def _write_algorithm(path: Path) -> None:
    path.write_text(
        """
from iints import AlgorithmInput, InsulinAlgorithm


class DemoAlgorithm(InsulinAlgorithm):
    def predict_insulin(self, data: AlgorithmInput):
        return {"total_insulin_delivered": 0.0}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_run_doctor_blocks_aggressive_glucose_decay(tmp_path) -> None:
    algo = tmp_path / "algorithm.py"
    patient = tmp_path / "patient.yaml"
    scenario = tmp_path / "scenario.json"
    report_json = tmp_path / "doctor.json"
    _write_algorithm(algo)
    patient.write_text(
        "\n".join(
            [
                "basal_insulin_rate: 0.2",
                "insulin_sensitivity: 40.0",
                "carb_factor: 15.0",
                "initial_glucose: 140.0",
                "glucose_decay_rate: 0.05",
            ]
        ),
        encoding="utf-8",
    )
    scenario.write_text(
        json.dumps(
            {
                "scenario_name": "late meal",
                "scenario_version": "1.0",
                "stress_events": [{"start_time": 60, "event_type": "meal", "value": 45}],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "run-doctor",
            "--algo",
            str(algo),
            "--patient-config-path",
            str(patient),
            "--scenario-path",
            str(scenario),
            "--duration",
            "120",
            "--time-step",
            "5",
            "--output-json",
            str(report_json),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert any(check["name"] == "glucose_decay" and check["status"] == "fail" for check in payload["checks"])


def test_evidence_build_creates_public_bundle(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "time_minutes": [0, 5, 10, 15],
            "glucose_actual_mgdl": [110.0, 120.0, 135.0, 128.0],
            "delivered_insulin_units": [0.0, 0.0, 0.2, 0.0],
        }
    ).to_csv(run_dir / "results.csv", index=False)
    (run_dir / "run_manifest.json").write_text('{"seed": 42}\n', encoding="utf-8")
    (run_dir / "safety_report.json").write_text('{"critical_events": 0}\n', encoding="utf-8")

    output_dir = tmp_path / "evidence"
    result = runner.invoke(
        app,
        [
            "evidence",
            "build",
            "--run",
            f"normal={run_dir}",
            "--output-dir",
            str(output_dir),
            "--title",
            "Demo Evidence",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "MODEL_CARD.md").is_file()
    assert (output_dir / "evidence_summary.json").is_file()
    assert (output_dir / "run_index.csv").is_file()


def test_safety_visualize_writes_html_and_json(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "time_minutes": [0, 5, 10, 15],
            "glucose_actual_mgdl": [120.0, 118.0, 62.0, 65.0],
            "algo_recommended_insulin_units": [0.0, 0.5, 1.0, 0.0],
            "delivered_insulin_units": [0.0, 0.5, 0.0, 0.0],
            "safety_triggered": [False, False, True, False],
            "safety_reason": ["", "", "hypoglycemia risk", ""],
        }
    ).to_csv(results_csv, index=False)

    output_html = tmp_path / "safety.html"
    output_json = tmp_path / "safety.json"
    result = runner.invoke(
        app,
        [
            "safety-visualize",
            "--results-csv",
            str(results_csv),
            "--output-html",
            str(output_html),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0
    assert output_html.is_file()
    assert output_json.is_file()
    summary = json.loads(output_json.read_text())
    assert summary["intervention_count"] == 1
    assert "hypoglycemia risk" in summary["top_reasons"]


def test_run_quality_artifacts_write_realism_and_safety_outputs(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "time_minutes": list(range(0, 1440, 5)),
            "glucose_actual_mgdl": [120.0 + (idx % 48) * 0.8 for idx in range(288)],
            "carb_intake_grams": [45.0 if idx in {18, 84, 156} else 0.0 for idx in range(288)],
            "algo_recommended_insulin_units": [0.4 if idx in {18, 84, 156} else 0.0 for idx in range(288)],
            "delivered_insulin_units": [0.4 if idx in {18, 84, 156} else 0.0 for idx in range(288)],
            "safety_triggered": [False for _ in range(288)],
            "safety_reason": ["" for _ in range(288)],
        }
    )

    outputs = write_run_quality_artifacts(df, tmp_path, run_label="quality-test", safety_report={})

    assert Path(outputs["realism_report_json"]).is_file()
    assert Path(outputs["realism_dashboard_html"]).is_file()
    assert Path(outputs["run_quality_review_md"]).is_file()
    assert Path(outputs["run_quality_summary_json"]).is_file()
    assert Path(outputs["safety_visualizer_html"]).is_file()
    assert Path(outputs["safety_visualizer_json"]).is_file()
    assert "verdict" in outputs["realism_review"]
    assert outputs["realism_review"]["quality_grade"] in {"research_ready", "review_before_use", "do_not_use"}
    assert "grade" in outputs["run_quality"]
    review_text = Path(outputs["run_quality_review_md"]).read_text()
    assert "IINTS Run Quality Review" in review_text
    assert "Result quality grade" in review_text
    assert "Max glucose rate" in review_text


def test_run_quality_artifacts_can_write_local_ai_verification(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOllamaBackend:
        backend_name = "ollama"

        def __init__(self, *, model_name: str, **_: object) -> None:
            self.model_name = model_name
            self.base_url = "http://127.0.0.1:11434"

        def available(self) -> bool:
            return True

        def ensure_model_ready(self) -> str:
            return self.model_name

        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            assert "not a medical device" in system_prompt.lower()
            assert "deterministic_quality_gate" in user_prompt
            return "## Verdict\nResearch-ready with review notes.\n"

    monkeypatch.setattr("iints.analysis.run_quality.OllamaBackend", FakeOllamaBackend)
    df = pd.DataFrame(
        {
            "time_minutes": list(range(0, 240, 5)),
            "glucose_actual_mgdl": [125.0 + (idx % 12) * 1.4 for idx in range(48)],
            "carb_intake_grams": [35.0 if idx == 12 else 0.0 for idx in range(48)],
            "delivered_insulin_units": [1.2 if idx == 12 else 0.0 for idx in range(48)],
            "safety_triggered": [False for _ in range(48)],
            "safety_reason": ["" for _ in range(48)],
        }
    )

    outputs = write_run_quality_artifacts(
        df,
        tmp_path,
        run_label="ai-reviewed-run",
        safety_report={},
        local_ai_review="required",
        local_ai_model="fake-local-model",
    )

    assert outputs["local_ai_review_status"] == "completed"
    assert Path(outputs["local_ai_review_md"]).is_file()
    assert Path(outputs["local_ai_review_json"]).is_file()
    assert "Research-ready" in Path(outputs["local_ai_review_md"]).read_text()
    summary = json.loads(Path(outputs["run_quality_summary_json"]).read_text())
    assert summary["local_ai_review_status"] == "completed"


def test_run_quality_artifacts_skip_local_ai_when_ollama_unavailable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OfflineOllamaBackend:
        def __init__(self, **_: object) -> None:
            self.base_url = "http://127.0.0.1:11434"

        def available(self) -> bool:
            return False

    monkeypatch.setattr("iints.analysis.run_quality.OllamaBackend", OfflineOllamaBackend)
    df = pd.DataFrame(
        {
            "time_minutes": list(range(0, 120, 5)),
            "glucose_actual_mgdl": [118.0 + idx * 0.4 for idx in range(24)],
            "carb_intake_grams": [0.0 for _ in range(24)],
            "delivered_insulin_units": [0.0 for _ in range(24)],
        }
    )

    outputs = write_run_quality_artifacts(
        df,
        tmp_path,
        run_label="offline-ai-run",
        safety_report={},
        local_ai_review="auto",
    )

    assert outputs["local_ai_review_status"] == "skipped"
    metadata = json.loads(Path(outputs["local_ai_review_json"]).read_text())
    assert "not reachable" in metadata["reason"]


def test_run_quality_artifacts_do_not_force_daily_reference_on_short_demos(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "time_minutes": list(range(0, 180, 5)),
            "glucose_actual_mgdl": [125.0 + min(idx, 16) * 2.6 - max(idx - 18, 0) * 1.2 for idx in range(36)],
            "carb_intake_grams": [45.0 if idx == 6 else 0.0 for idx in range(36)],
            "delivered_insulin_units": [2.0 if idx == 6 else 0.0 for idx in range(36)],
            "safety_triggered": [False for _ in range(36)],
            "safety_reason": ["" for _ in range(36)],
        }
    )

    outputs = write_run_quality_artifacts(df, tmp_path, run_label="short-demo", safety_report={})

    assert outputs["realism_review"]["reference_selection"] == "auto"
    assert outputs["realism_review"]["reference"] is None
    assert outputs["realism_review"]["verdict"] == "likely_realistic"
    assert outputs["run_quality"]["grade"] == "research_ready"


def test_run_quality_artifacts_flag_unusable_runs(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "time_minutes": [0, 5, 5, 10],
            "glucose_actual_mgdl": [120.0, None, 520.0, 20.0],
            "carb_intake_grams": [0.0, 0.0, 0.0, 0.0],
            "delivered_insulin_units": [0.0, 0.0, 0.0, 0.0],
            "safety_triggered": [False, True, True, True],
            "safety_reason": ["", "sensor", "sensor", "sensor"],
        }
    )

    outputs = write_run_quality_artifacts(
        df,
        tmp_path,
        run_label="bad-run",
        safety_report={"terminated_early": True, "input_validator_fail_soft_count": 2},
    )

    assert outputs["run_quality"]["grade"] == "do_not_use"
    assert outputs["run_quality"]["terminated_early"] is True
    assert outputs["run_quality"]["duplicate_timestamp_rows"] == 1
    assert outputs["run_quality"]["nan_glucose_rows"] == 1


def test_run_quality_treats_pd_stacking_as_routine_dose_shaping(tmp_path) -> None:
    class FakeRealismReport:
        verdict = "likely_realistic"
        realism_score = 0.92

    df = pd.DataFrame(
        {
            "time_minutes": list(range(0, 240, 5)),
            "glucose_actual_mgdl": [125.0 + (idx % 12) * 1.2 for idx in range(48)],
            "carb_intake_grams": [35.0 if idx == 12 else 0.0 for idx in range(48)],
            "delivered_insulin_units": [0.1 for _ in range(48)],
            "safety_triggered": [True for _ in range(48)],
            "safety_reason": [
                "PD_STACKING_PREVENTION: Dose reduced by 35.0% due to unabsorbed IOB (0.80U)"
                for _ in range(48)
            ],
        }
    )

    quality = build_result_quality_summary(df, realism_report=FakeRealismReport(), safety_report={})

    assert quality["safety_intervention_count"] == 48
    assert quality["routine_dose_shaping_count"] == 48
    assert quality["material_safety_intervention_count"] == 0
    assert quality["grade"] == "research_ready"


def test_run_quality_treats_pd_clearance_limit_as_routine_dose_shaping() -> None:
    class FakeRealismReport:
        verdict = "likely_realistic"
        realism_score = 0.92

    df = pd.DataFrame(
        {
            "time_minutes": [0, 5, 10],
            "glucose_actual_mgdl": [150.0, 148.0, 146.0],
            "delivered_insulin_units": [0.2, 0.15, 0.12],
            "safety_triggered": [True, True, False],
            "safety_reason": [
                "PD_CLEARANCE_LIMIT / High IOB: Active IOB 1.00U restricts max safe bolus to 3.00U",
                "PD_CLEARANCE_LIMIT / High IOB: Active IOB 1.20U restricts max safe bolus to 2.80U; PD_STACKING_PREVENTION: Dose reduced by 35.0% due to unabsorbed IOB (1.20U)",
                "APPROVED",
            ],
        }
    )

    quality = build_result_quality_summary(df, realism_report=FakeRealismReport(), safety_report={})

    assert quality["routine_dose_shaping_count"] == 2
    assert quality["material_safety_intervention_count"] == 0
    assert quality["grade"] == "research_ready"


def test_top_level_pump_compile_and_bench_test(tmp_path) -> None:
    algo = tmp_path / "algorithm.py"
    bundle = tmp_path / "bundle"
    report_json = tmp_path / "bench.json"
    _write_algorithm(algo)

    compile_result = runner.invoke(
        app,
        [
            "pump",
            "compile",
            "--algorithm",
            str(algo),
            "--output-dir",
            str(bundle),
        ],
    )
    assert compile_result.exit_code == 0

    bench_result = runner.invoke(
        app,
        [
            "pump",
            "bench-test",
            "--bundle-dir",
            str(bundle),
            "--output-json",
            str(report_json),
        ],
    )
    assert bench_result.exit_code == 0
    assert json.loads(report_json.read_text(encoding="utf-8"))["passed"] is True


def test_research_train_local_ai_alias_is_available() -> None:
    result = runner.invoke(app, ["research", "train-local-ai", "--help"])

    assert result.exit_code == 0
    assert "--run" in result.stdout


def test_report_command_generates_agp_style_bundle(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    bundle_dir = tmp_path / "report_bundle"
    pd.DataFrame(
        {
            "time_minutes": list(range(0, 24 * 60, 5)),
            "glucose_actual_mgdl": [110.0 + (idx % 48) * 1.8 for idx in range(288)],
        }
    ).to_csv(results_csv, index=False)

    result = runner.invoke(
        app,
        [
            "report",
            "--results-csv",
            str(results_csv),
            "--bundle-dir",
            str(bundle_dir),
            "--style",
            "agp",
            "--subject-name",
            "CLI demo",
        ],
    )

    assert result.exit_code == 0
    assert (bundle_dir / "agp_report.pdf").is_file()
    assert (bundle_dir / "agp_summary.json").is_file()


def test_research_glucose_model_commands_are_available() -> None:
    result = runner.invoke(app, ["research", "glucose-model", "--help"])

    assert result.exit_code == 0
    assert "build-dataset" in result.stdout
    assert "compare" in result.stdout
    assert "export-hf" in result.stdout
    assert "jetson-train-hf" in result.stdout


def test_research_glucose_model_init_writes_config(tmp_path) -> None:
    output_dir = tmp_path / "glucose_model"
    result = runner.invoke(
        app,
        [
            "research",
            "glucose-model",
            "init",
            "--output-dir",
            str(output_dir),
            "--profile",
            "smoke",
            "--history-minutes",
            "60",
            "--horizon-minutes",
            "30",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "glucose_model_config.yaml").is_file()
    assert (output_dir / "MODEL_INTENT.md").is_file()


def test_research_glucose_model_compare_writes_research_gate_outputs(tmp_path) -> None:
    rows = 130
    dataset = tmp_path / "glucose.csv"
    pd.DataFrame(
        {
            "time_minutes": [index * 5 for index in range(rows)],
            "glucose": [120 + ((index % 24) - 12) * 1.5 for index in range(rows)],
            "carbs": [35.0 if index in {20, 70} else 0.0 for index in range(rows)],
            "insulin": [2.0 if index in {22, 72} else 0.0 for index in range(rows)],
            "subject_id": ["demo"] * rows,
        }
    ).to_csv(dataset, index=False)
    output_dir = tmp_path / "comparison"

    result = runner.invoke(
        app,
        [
            "research",
            "glucose-model",
            "compare",
            "--data",
            str(dataset),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "comparison_report.json").is_file()
    assert (output_dir / "physiological_violation_metrics.csv").is_file()
    report = json.loads((output_dir / "comparison_report.json").read_text())
    assert report["model_count"] == 3
