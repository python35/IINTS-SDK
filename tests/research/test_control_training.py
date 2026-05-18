from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.research.control import (
    CONTROL_FEATURE_COLUMNS,
    CONTROL_TARGET_COLUMN,
    build_control_dataset_from_runs,
    load_linear_controller,
    predict_linear_controller,
    save_linear_controller,
    train_linear_imitation_controller,
)
from iints.research.control_eval import evaluate_controller_factories
from iints.research.data_blend import blend_predictor_datasets
from iints.research.neural_control import (
    NeuralControllerConfig,
    load_neural_controller,
    predict_neural_controller,
    save_neural_controller,
    train_neural_imitation_controller,
)


runner = CliRunner()


def _control_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_minutes": [0, 5, 10, 15],
            "glucose_actual_mgdl": [120.0, 145.0, 170.0, 90.0],
            "glucose_trend_mgdl_min": [0.0, 1.0, 1.5, -0.5],
            "patient_iob_units": [0.0, 0.1, 0.2, 0.3],
            "patient_cob_grams": [0.0, 20.0, 10.0, 0.0],
            "effective_isf": [50.0, 50.0, 50.0, 50.0],
            "effective_icr": [10.0, 10.0, 10.0, 10.0],
            "effective_basal_rate_u_per_hr": [0.5, 0.5, 0.5, 0.5],
            "carb_intake_grams": [0.0, 30.0, 0.0, 0.0],
            "delivered_insulin_units": [0.05, 2.0, 0.3, 0.0],
            "algo_recommended_insulin_units": [0.05, 2.0, 0.3, 0.0],
            "safety_triggered": [False, False, False, True],
        }
    )


def _controller_dataset_rows() -> pd.DataFrame:
    rows = _control_rows().copy()
    rows[CONTROL_TARGET_COLUMN] = rows["delivered_insulin_units"]
    return rows


def test_build_and_train_local_controller(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    (run_dir / "raw").mkdir(parents=True)
    _control_rows().to_csv(run_dir / "raw" / "steps.csv", index=False)
    dataset_path = tmp_path / "controller.csv"
    manifest_path = tmp_path / "controller_manifest.json"

    manifest = build_control_dataset_from_runs(
        [("run_a", run_dir)],
        output_path=dataset_path,
        manifest_path=manifest_path,
    )
    dataset = pd.read_csv(dataset_path)
    model = train_linear_imitation_controller(dataset)
    model_path = tmp_path / "controller.json"
    save_linear_controller(model, model_path)
    loaded = load_linear_controller(model_path)
    predictions = predict_linear_controller(loaded, dataset)

    assert manifest["rows"] == 4
    assert manifest["target_column"] == CONTROL_TARGET_COLUMN
    assert set(CONTROL_FEATURE_COLUMNS).issubset(dataset.columns)
    assert len(predictions) == 4
    assert manifest_path.is_file()


def test_blend_predictor_datasets_prefixes_subjects(tmp_path: Path) -> None:
    base = pd.DataFrame(
        {
            "subject_id": ["001", "001"],
            "time_minutes": [0, 5],
            "glucose_actual_mgdl": [120.0, 121.0],
            "glucose_trend_mgdl_min": [0.0, 0.2],
            "patient_iob_units": [0.0, 0.0],
            "patient_cob_grams": [0.0, 0.0],
            "effective_isf": [50.0, 50.0],
            "effective_icr": [10.0, 10.0],
            "effective_basal_rate_u_per_hr": [0.5, 0.5],
        }
    )
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    base.to_csv(path_a, index=False)
    base.to_csv(path_b, index=False)

    report = blend_predictor_datasets(
        [("azt1d", path_a), ("hupa", path_b)],
        output_path=tmp_path / "blend.csv",
        manifest_path=tmp_path / "blend_manifest.json",
    )
    blended = pd.read_csv(tmp_path / "blend.csv")

    assert report["rows"] == 4
    assert sorted(blended["subject_id"].unique().tolist()) == ["azt1d:1", "hupa:1"]


def test_research_cli_builds_and_trains_controller(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    (run_dir / "raw").mkdir(parents=True)
    _control_rows().to_csv(run_dir / "raw" / "steps.csv", index=False)
    dataset_path = tmp_path / "controller.csv"
    manifest_path = tmp_path / "controller_manifest.json"
    model_path = tmp_path / "controller.json"
    metrics_path = tmp_path / "controller_metrics.json"

    build_result = runner.invoke(
        app,
        [
            "research",
            "build-control-dataset",
            "--run",
            f"run_a={run_dir}",
            "--output",
            str(dataset_path),
            "--manifest",
            str(manifest_path),
        ],
    )
    train_result = runner.invoke(
        app,
        [
            "research",
            "train-controller",
            "--data",
            str(dataset_path),
            "--output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
        ],
    )

    assert build_result.exit_code == 0
    assert train_result.exit_code == 0
    assert model_path.is_file()
    assert json.loads(metrics_path.read_text())["rows"] == 4


def test_train_neural_controller_and_run_held_out_evaluation(tmp_path: Path) -> None:
    __import__("pytest").importorskip("torch")
    dataset = pd.concat([_controller_dataset_rows(), _controller_dataset_rows()], ignore_index=True)
    checkpoint = train_neural_imitation_controller(
        dataset,
        config=NeuralControllerConfig(epochs=3, batch_size=4),
    )
    model_path = tmp_path / "controller_neural.pt"
    save_neural_controller(checkpoint, model_path)
    loaded = load_neural_controller(model_path)
    predictions = predict_neural_controller(loaded, dataset)

    from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
    from iints.core.algorithms.neural_controller import ExperimentalNeuralController

    evaluation = evaluate_controller_factories(
        {
            "clinical_baseline": ClinicalBaselineAlgorithm,
            "neural_controller": lambda: ExperimentalNeuralController(settings={"model_path": str(model_path)}),
        },
        output_dir=tmp_path / "evaluation",
        presets=["hypo_prone_night"],
        seeds=[7],
        duration_minutes=60,
    )

    assert len(predictions) == len(dataset)
    assert checkpoint["validation_metrics"] is not None
    assert (tmp_path / "evaluation" / "CONTROL_EVALUATION_REPORT.md").is_file()
    assert "neural_controller" in evaluation["algorithms"]


def test_research_cli_trains_neural_controller_and_evaluates(tmp_path: Path) -> None:
    __import__("pytest").importorskip("torch")
    dataset_path = tmp_path / "controller.csv"
    pd.concat([_controller_dataset_rows(), _controller_dataset_rows()], ignore_index=True).to_csv(
        dataset_path,
        index=False,
    )
    model_path = tmp_path / "controller_neural.pt"
    metrics_path = tmp_path / "controller_neural_metrics.json"
    evaluation_dir = tmp_path / "evaluation"

    train_result = runner.invoke(
        app,
        [
            "research",
            "train-neural-controller",
            "--data",
            str(dataset_path),
            "--output",
            str(model_path),
            "--metrics-output",
            str(metrics_path),
            "--epochs",
            "3",
        ],
    )
    eval_result = runner.invoke(
        app,
        [
            "research",
            "evaluate-controller",
            "--model",
            str(model_path),
            "--model-kind",
            "neural",
            "--output-dir",
            str(evaluation_dir),
            "--preset",
            "hypo_prone_night",
            "--seed",
            "7",
            "--duration-minutes",
            "60",
        ],
    )

    assert train_result.exit_code == 0
    assert eval_result.exit_code == 0
    assert model_path.is_file()
    assert (evaluation_dir / "closed_loop_summary.json").is_file()
