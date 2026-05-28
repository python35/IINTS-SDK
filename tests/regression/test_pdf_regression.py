import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")


os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from iints.analysis.reporting import ClinicalReportGenerator
from iints.data.importer import load_demo_dataframe, import_cgm_dataframe


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
