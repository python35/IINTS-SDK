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
from iints.research.data_blend import blend_predictor_datasets


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
