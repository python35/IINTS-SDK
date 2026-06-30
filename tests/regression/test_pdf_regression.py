import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import pandas as pd
from typer.testing import CliRunner


os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from iints.cli.cli import app
from iints.analysis.reporting import ClinicalReportGenerator
from iints.data.importer import load_demo_dataframe, import_cgm_dataframe

runner = CliRunner()


def test_demo_pdf_generation(tmp_path: Path) -> None:
    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")

    sim_df = standard_df.copy()
    sim_df["time_minutes"] = sim_df["timestamp"]
    sim_df["glucose_actual_mgdl"] = sim_df["glucose"]
    sim_df["delivered_insulin_units"] = 0.0

    safety_report = {"total_violations": 0, "bolus_interventions_count": 0}

    output_path = tmp_path / "demo_report.pdf"
    generator = ClinicalReportGenerator()
    generator.generate_pdf(sim_df, safety_report, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_clinical_validation_plot_survives_tight_layout_mathtext_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sim_df = pd.DataFrame(
        {
            "time_minutes": [0, 5, 10, 15, 20, 25],
            "glucose_actual_mgdl": [110, 118, 140, 170, 155, 135],
            "carb_intake_grams": [0, 15, 0, 0, 0, 0],
            "delivered_insulin_units": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        }
    )

    def broken_tight_layout(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise ValueError("simulated Matplotlib mathtext parse failure")

    monkeypatch.setattr(Figure, "tight_layout", broken_tight_layout)

    output_path = tmp_path / "clinical_validation_pattern.png"
    ClinicalReportGenerator()._plot_clinical_validation_pattern(sim_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_agp_pdf_generation(tmp_path: Path) -> None:
    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")

    sim_df = standard_df.copy()
    sim_df["time_minutes"] = sim_df["timestamp"]
    sim_df["glucose_actual_mgdl"] = sim_df["glucose"]

    output_path = tmp_path / "agp_report.pdf"
    summary_path = tmp_path / "agp_summary.json"
    generator = ClinicalReportGenerator()
    generator.generate_agp_pdf(
        sim_df,
        str(output_path),
        subject_name="Regression demo",
        summary_json_path=str(summary_path),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 1000
    summary = json.loads(summary_path.read_text())
    assert summary["reading_count"] == len(sim_df)
    assert "target_70_180" in summary["time_ranges_pct"]


def test_agp_png_asset_export(tmp_path: Path) -> None:
    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")

    sim_df = standard_df.copy()
    sim_df["time_minutes"] = sim_df["timestamp"]
    sim_df["glucose_actual_mgdl"] = sim_df["glucose"]

    generator = ClinicalReportGenerator()
    outputs = generator.export_agp_assets(sim_df, str(tmp_path / "agp_assets"), subject_name="PNG demo")

    assert Path(outputs["agp_profile_png"]).is_file()
    assert Path(outputs["daily_profiles_png"]).is_file()
    assert Path(outputs["agp_profile_svg"]).is_file()
    assert Path(outputs["daily_profiles_svg"]).is_file()
    summary = json.loads(Path(outputs["summary_json"]).read_text())
    assert summary["subject_name"] == "PNG demo"
    assert summary["reading_count"] == len(sim_df)
    assert abs(sum(summary["time_ranges_pct"].values()) - 100.0) < 1e-6


def test_agp_time_in_range_uses_all_standard_zones(tmp_path: Path) -> None:
    sim_df = pd.DataFrame(
        {
            "time_minutes": [0, 5, 10, 15, 20],
            "glucose_actual_mgdl": [50, 60, 100, 200, 260],
        }
    )

    generator = ClinicalReportGenerator()
    outputs = generator.export_agp_assets(sim_df, str(tmp_path / "agp_zones"), subject_name="Five-zone demo")

    summary = json.loads(Path(outputs["summary_json"]).read_text())
    assert summary["time_ranges_pct"] == {
        "very_high_gt_250": 20.0,
        "high_181_250": 20.0,
        "target_70_180": 20.0,
        "low_54_69": 20.0,
        "very_low_lt_54": 20.0,
    }


def test_agp_asset_export_writes_structured_xai_events(tmp_path: Path) -> None:
    sim_df = pd.DataFrame(
        {
            "time_minutes": [0, 5],
            "glucose_actual_mgdl": [110, 150],
            "explainable_events": [
                "",
                "At 00:05 glucose started rising after meal/breakfast.; "
                "At 00:05 the model detected faster-than-expected absorption.",
            ],
        }
    )

    generator = ClinicalReportGenerator()
    outputs = generator.export_agp_assets(sim_df, str(tmp_path / "agp_xai"), subject_name="XAI demo")

    text = Path(outputs["xai_events_txt"]).read_text(encoding="utf-8")
    events = json.loads(Path(outputs["xai_events_json"]).read_text(encoding="utf-8"))

    assert "glucose started rising after meal/breakfast" in text
    assert "faster-than-expected absorption" in text
    assert [entry["event"] for entry in events] == [
        "At 00:05 glucose started rising after meal/breakfast.",
        "At 00:05 the model detected faster-than-expected absorption.",
    ]
    assert events[0]["time_minutes"] == 5.0


def test_report_cli_exports_agp_png_assets(tmp_path: Path) -> None:
    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")

    sim_df = standard_df.copy()
    sim_df["time_minutes"] = sim_df["timestamp"]
    sim_df["glucose_actual_mgdl"] = sim_df["glucose"]
    results_csv = tmp_path / "results.csv"
    sim_df.to_csv(results_csv, index=False)

    bundle_dir = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "report",
            "--results-csv",
            str(results_csv),
            "--style",
            "agp",
            "--png",
            "--bundle-dir",
            str(bundle_dir),
        ],
    )

    assert result.exit_code == 0
    assert (bundle_dir / "agp_report.pdf").is_file()
    assert (bundle_dir / "agp_summary.json").is_file()
    assert (bundle_dir / "agp_assets" / "agp_profile.png").is_file()
    assert (bundle_dir / "agp_assets" / "daily_profiles.png").is_file()
    assert (bundle_dir / "agp_assets" / "agp_profile.svg").is_file()
    assert (bundle_dir / "agp_assets" / "daily_profiles.svg").is_file()
