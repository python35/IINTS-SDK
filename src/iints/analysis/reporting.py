import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.utils.plotting import apply_plot_style


class ClinicalReportGenerator:
    """Generate a clean, publication-ready PDF report."""

    def __init__(self) -> None:
        self.metrics_calculator = ClinicalMetricsCalculator()

    def _plot_glucose(self, df: pd.DataFrame, output_path: Path) -> None:
        apply_plot_style()
        plt.figure(figsize=(10, 4))
        plt.plot(df["time_minutes"], df["glucose_actual_mgdl"], color="#2e7d32", linewidth=1.8)
        plt.axhspan(70, 180, alpha=0.12, color="#4caf50", label="Target 70-180")
        plt.axhline(70, color="#d32f2f", linestyle="--", linewidth=1)
        plt.axhline(180, color="#f57c00", linestyle="--", linewidth=1)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Glucose (mg/dL)")
        plt.title("Glucose Trace")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

    def _plot_insulin(self, df: pd.DataFrame, output_path: Path) -> None:
        apply_plot_style()
        plt.figure(figsize=(10, 3))
        plt.bar(df["time_minutes"], df["delivered_insulin_units"], width=4, color="#1976d2", alpha=0.7)
        plt.ylim(bottom=0)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Insulin (U)")
        plt.title("Delivered Insulin")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

    def export_plots(self, simulation_data: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        glucose_plot = output_path / "glucose.png"
        insulin_plot = output_path / "insulin.png"
        self._plot_glucose(simulation_data, glucose_plot)
        self._plot_insulin(simulation_data, insulin_plot)
        return {
            "glucose_plot": str(glucose_plot),
            "insulin_plot": str(insulin_plot),
        }

    def _top_safety_reasons(self, df: pd.DataFrame, limit: int = 3) -> Dict[str, int]:
        if "safety_reason" not in df.columns:
            return {}
        if "safety_triggered" in df.columns:
            filtered = df[df["safety_triggered"] == True]
        else:
            filtered = df

        reasons: Dict[str, int] = {}
        for reason in filtered["safety_reason"].dropna():
            if not reason:
                continue
            for entry in str(reason).split(";"):
                label = entry.strip().split(":")[0].strip()
                if not label:
                    continue
                reasons[label] = reasons.get(label, 0) + 1

        if not reasons:
            return {}

        sorted_reasons = sorted(reasons.items(), key=lambda item: item[1], reverse=True)
        return dict(sorted_reasons[:limit])

    def generate_pdf(
        self,
        simulation_data: pd.DataFrame,
        safety_report: Dict[str, Any],
        output_path: str,
        title: str = "IINTS-AF Clinical Report",
    ) -> str:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.metrics_calculator.calculate(
            glucose=simulation_data["glucose_actual_mgdl"],
            duration_hours=(simulation_data["time_minutes"].max() / 60.0),
        ).to_dict()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            glucose_plot = tmp_dir_path / "glucose.png"
            insulin_plot = tmp_dir_path / "insulin.png"
            self._plot_glucose(simulation_data, glucose_plot)
            self._plot_insulin(simulation_data, insulin_plot)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()

            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, title, ln=1)

            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 7, f"Duration: {simulation_data['time_minutes'].max()/60:.1f} hours", ln=1)
            pdf.cell(0, 7, f"Data points: {len(simulation_data)}", ln=1)

            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Clinical Metrics", ln=1)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"TIR (70-180): {metrics.get('tir_70_180', 0):.1f}%", ln=1)
            pdf.cell(0, 6, f"Time <70: {metrics.get('tir_below_70', 0):.1f}%", ln=1)
            pdf.cell(0, 6, f"Time >180: {metrics.get('tir_above_180', 0):.1f}%", ln=1)
            pdf.cell(0, 6, f"CV: {metrics.get('cv', 0):.1f}%", ln=1)
            pdf.cell(0, 6, f"GMI: {metrics.get('gmi', 0):.1f}%", ln=1)

            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Safety Summary", ln=1)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Total violations: {safety_report.get('total_violations', 0)}", ln=1)
            pdf.cell(0, 6, f"Bolus interventions: {safety_report.get('bolus_interventions_count', 0)}", ln=1)
            top_reasons = self._top_safety_reasons(simulation_data)
            if top_reasons:
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6, "Top intervention reasons:", ln=1)
                pdf.set_font("Helvetica", "", 10)
                for reason, count in top_reasons.items():
                    pdf.cell(0, 5, f"- {reason}: {count}", ln=1)

            baseline = safety_report.get("baseline_comparison")
            if baseline and baseline.get("rows"):
                pdf.ln(3)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 7, "Head-to-Head Comparison", ln=1)
                pdf.set_font("Helvetica", "B", 9)
                col_widths = [52, 26, 26, 26, 30]
                headers = ["Algorithm", "TIR 70-180", "Time <70", "Time >180", "Safety Overrides"]
                for idx, header in enumerate(headers):
                    pdf.cell(col_widths[idx], 6, header, 1, 0, "C")
                pdf.ln()

                pdf.set_font("Helvetica", "", 9)
                for row in baseline["rows"]:
                    pdf.cell(col_widths[0], 6, str(row.get("algorithm", ""))[:24], 1, 0)
                    pdf.cell(col_widths[1], 6, f"{row.get('tir_70_180', 0):.1f}%", 1, 0, "C")
                    pdf.cell(col_widths[2], 6, f"{row.get('tir_below_70', 0):.1f}%", 1, 0, "C")
                    pdf.cell(col_widths[3], 6, f"{row.get('tir_above_180', 0):.1f}%", 1, 0, "C")
                    pdf.cell(col_widths[4], 6, str(row.get("bolus_interventions", 0)), 1, 0, "C")
                    pdf.ln()

            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Glucose Trace", ln=1)
            pdf.image(str(glucose_plot), w=180)

            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Insulin Delivery", ln=1)
            pdf.image(str(insulin_plot), w=180)

            pdf.output(str(output_file))

        return str(output_file)
