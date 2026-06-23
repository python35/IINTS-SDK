from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from iints.research.glucose_model import (
    GLUCOSE_MODEL_FEATURE_COLUMNS,
    build_glucose_training_pack,
    compare_glucose_models,
    glucose_model_config_payload,
    physiological_violation_report,
    public_manifest_from_private,
    standardize_glucose_forecast_frame,
    write_huggingface_export_bundle,
)


def _demo_frame(rows: int = 80, subject: str = "s1") -> pd.DataFrame:
    t = np.arange(rows) * 5
    return pd.DataFrame(
        {
            "time_minutes": t,
            "subject_id": subject,
            "glucose": 120 + np.sin(np.arange(rows) / 8) * 18,
            "carbs": np.where((np.arange(rows) % 36) == 4, 35.0, 0.0),
            "insulin": np.where((np.arange(rows) % 36) == 5, 2.5, 0.0),
            "heart_rate": 80 + np.sin(np.arange(rows) / 12) * 5,
        }
    )


def test_standardize_glucose_forecast_frame_derives_contract_columns() -> None:
    frame = standardize_glucose_forecast_frame(
        _demo_frame(),
        source_label="demo",
        time_step_minutes=5,
    )

    for column in ["time_minutes", "subject_id", "segment", "source_dataset", *GLUCOSE_MODEL_FEATURE_COLUMNS]:
        assert column in frame.columns
    assert frame["subject_id"].iloc[0].startswith("demo:")
    assert frame["glucose_actual_mgdl"].between(35, 450).all()
    assert frame["glucose_trend_mgdl_min"].abs().max() > 0
    assert frame["time_of_day_sin"].between(-1, 1).all()
    assert frame["time_of_day_cos"].between(-1, 1).all()


def test_build_glucose_training_pack_writes_dataset_config_and_manifest(tmp_path: Path) -> None:
    path_a = tmp_path / "source_a.csv"
    path_b = tmp_path / "source_b.csv"
    _demo_frame(subject="a").to_csv(path_a, index=False)
    _demo_frame(subject="b").to_csv(path_b, index=False)

    pack = build_glucose_training_pack(
        [path_a, path_b],
        tmp_path / "pack",
        labels=["ohio_like", "sim_like"],
        profile="smoke",
        history_minutes=60,
        horizon_minutes=30,
    )

    assert pack.row_count == 160
    assert pack.subject_count == 2
    assert pack.source_count == 2
    assert pack.dataset_path.is_file()
    assert pack.config_path.is_file()
    assert pack.manifest_path.is_file()
    config = yaml.safe_load(pack.config_path.read_text())
    assert config["iints_glucose_model"]["model_id"] == "iints-glucose-forecast-v0"
    assert config["training"]["epochs"] == 2
    manifest = json.loads(pack.manifest_path.read_text())
    assert manifest["privacy"]["raw_private_data_included"] is False
    assert manifest["sources"][0]["label"] == "ohio_like"


def test_public_manifest_redacts_local_paths_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "private_ohio.csv"
    _demo_frame().to_csv(source, index=False)
    pack = build_glucose_training_pack([source], tmp_path / "pack", labels=["ohio_full"], profile="smoke")
    manifest = json.loads(pack.manifest_path.read_text())

    public = public_manifest_from_private(manifest)

    assert public["privacy"]["local_paths_redacted"] is True
    assert "path" not in public["sources"][0]
    assert "sha256" not in public["sources"][0]
    assert public["sources"][0]["label"] == "ohio_full"


def test_write_huggingface_export_bundle_creates_research_safe_model_card(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _demo_frame().to_csv(source, index=False)
    pack = build_glucose_training_pack([source], tmp_path / "pack", labels=["demo"], profile="smoke")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "predictor.pt").write_bytes(b"fake torch checkpoint")
    (model_dir / "glucose_model_config.resolved.yaml").write_text(pack.config_path.read_text())
    (model_dir / "training_report.json").write_text(
        json.dumps({"test_mae": 12.3, "test_rmse": 18.4, "test_bias": -1.2})
    )
    comparison_dir = tmp_path / "comparison"
    comparison_dir.mkdir()
    (comparison_dir / "comparison_report.md").write_text("# Comparison\n")
    (comparison_dir / "comparison_report.json").write_text(json.dumps({"schema_version": "demo"}))
    (comparison_dir / "horizon_metrics.csv").write_text("model,mae\npinn,12.3\n")
    (comparison_dir / "physiological_violation_metrics.csv").write_text("model,any_physiology_violation_pct\npinn,0\n")
    (comparison_dir / "hypo_detection_metrics.csv").write_text("model,missed_hypo_rate_pct\npinn,0\n")
    (comparison_dir / "model_card_metrics.json").write_text(
        json.dumps(
            {
                "best_by_mae": {"model": "pinn", "mae": 12.3},
                "models": [
                    {
                        "model": "pinn",
                        "kind": "checkpoint",
                        "mae": 12.3,
                        "rmse": 18.4,
                        "missed_hypo_rate_pct": 0.0,
                        "any_physiology_violation_pct": 0.0,
                    }
                ],
            }
        )
    )

    outputs = write_huggingface_export_bundle(
        model_dir=model_dir,
        output_dir=tmp_path / "hf",
        repo_id="rune/iints-glucose-forecast-v0",
        dataset_manifest=pack.manifest_path,
        comparison_dir=comparison_dir,
    )

    readme = Path(outputs["readme"]).read_text()
    assert "IINTS-AF Glucose Forecast v0" in readme
    assert "not a medical device" in readme.lower()
    assert "Test MAE: 12.300" in readme
    assert "Physiology-Aware Evaluation" in readme
    assert "comparison_interpretation.md" in readme
    assert (tmp_path / "hf" / "dataset_manifest.public.json").is_file()
    assert (tmp_path / "hf" / "predictor.pt").read_bytes() == b"fake torch checkpoint"
    assert (tmp_path / "hf" / "glucose_model_config.yaml").is_file()
    assert (tmp_path / "hf" / "comparison_report.md").is_file()
    assert (tmp_path / "hf" / "comparison_interpretation.md").is_file()
    assert (tmp_path / "hf" / "model_card_metrics.json").is_file()
    assert (tmp_path / "hf" / "privacy.md").is_file()
    assert (tmp_path / "hf" / "limitations.md").is_file()
    assert (tmp_path / "hf" / "examples" / "inference_example.py").is_file()
    assert (tmp_path / "hf" / "examples" / "sample_glucose_trace.csv").is_file()
    interpretation = (tmp_path / "hf" / "comparison_interpretation.md").read_text()
    assert "Why MSE Can Look Best" in interpretation
    assert "Why PINN Is Different" in interpretation


def test_physiological_violation_report_flags_impossible_roc_and_iob_logic() -> None:
    feature_columns = ["glucose_actual_mgdl", "patient_iob_units", "patient_cob_grams"]
    X = np.zeros((2, 4, 3), dtype=np.float32)
    X[:, :, 0] = 100.0
    X[0, -1, 1] = 2.0  # IOB high
    X[0, -1, 2] = 0.0  # no COB
    X[1, -1, 1] = 0.0
    X[1, -1, 2] = 30.0
    predicted = np.array([[100.0, 700.0], [100.0, -10.0]], dtype=np.float32)

    report = physiological_violation_report(
        X,
        predicted,
        feature_columns=feature_columns,
        time_step_minutes=5,
        max_roc_mgdl_min=10,
    )

    assert report["impossible_high_count"] == 1
    assert report["impossible_low_count"] == 1
    assert report["roc_violation_count"] >= 2
    assert report["any_physiology_violation_pct"] == 100.0


def test_compare_glucose_models_writes_baseline_comparison_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _demo_frame(rows=120).to_csv(source, index=False)
    pack = build_glucose_training_pack(
        [source],
        tmp_path / "pack",
        labels=["demo"],
        profile="smoke",
        history_minutes=30,
        horizon_minutes=15,
    )

    bundle = compare_glucose_models(
        data_path=pack.dataset_path,
        config_path=pack.config_path,
        output_dir=tmp_path / "comparison",
        include_baselines=True,
    )

    assert bundle.model_count == 3
    assert bundle.report_json.is_file()
    assert bundle.report_md.is_file()
    assert bundle.horizon_metrics_csv.is_file()
    assert bundle.physiological_violations_csv.is_file()
    report = json.loads(bundle.report_json.read_text())
    assert report["schema_version"] == "iints_glucose_model_comparison_v1"
    assert report["best_by_mae"]["model"] in {"LastValue", "LinearTrend", "PhysiologyAware"}
    model_card_metrics = json.loads(bundle.model_card_metrics_json.read_text())
    assert model_card_metrics["privacy"]["raw_data_included"] is False


def test_compare_glucose_models_derives_meal_announcement_feature(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _demo_frame(rows=120).to_csv(source, index=False)
    pack = build_glucose_training_pack(
        [source],
        tmp_path / "pack",
        labels=["demo"],
        profile="smoke",
        history_minutes=30,
        horizon_minutes=15,
    )
    config = glucose_model_config_payload(
        profile="smoke",
        history_minutes=30,
        horizon_minutes=15,
        feature_columns=[
            "glucose_actual_mgdl",
            "glucose_trend_mgdl_min",
            "patient_iob_units",
            "patient_cob_grams",
            "meal_announcement_grams",
        ],
    )
    config["training"]["meal_announcement_minutes"] = 15
    config["training"]["meal_announcement_column"] = "carb_intake_grams"
    config["training"]["meal_announcement_feature"] = "meal_announcement_grams"
    config_path = tmp_path / "meal_announcement_config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    bundle = compare_glucose_models(
        data_path=pack.dataset_path,
        config_path=config_path,
        output_dir=tmp_path / "comparison_meal_announcement",
        include_baselines=True,
    )

    assert bundle.report_json.is_file()
    report = json.loads(bundle.report_json.read_text())
    assert "meal_announcement_grams" in report["feature_columns"]


def test_glucose_model_config_is_band_pinn_first_for_dedicated_training() -> None:
    pack = glucose_model_config_payload(profile="smoke")

    assert pack["training"]["loss"] == "band_pinn"
    assert pack["training"]["pinn_lambda"] > 0
    assert pack["training"]["band_weighted_low_weight"] > 0
