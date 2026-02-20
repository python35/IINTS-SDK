"""
Population Report Generator — IINTS-AF
========================================
Generates a PDF report with aggregate statistics and visualisations
for a Monte Carlo population evaluation run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

from iints.utils.plotting import apply_plot_style, IINTS_BLUE, IINTS_RED, IINTS_TEAL


class PopulationReportGenerator:
    """Generate a PDF report for population simulation results."""

    def generate_pdf(
        self,
        summary_df: pd.DataFrame,
        aggregate_metrics: Dict[str, Any],
        aggregate_safety: Dict[str, Any],
        output_path: str,
        title: str = "IINTS-AF Population Evaluation Report",
    ) -> str:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = self._generate_plots(summary_df, output_dir)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Title page ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Population size: {len(summary_df)}", new_x="LMARGIN", new_y="NEXT")

        # --- Aggregate clinical metrics ---
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Aggregate Clinical Metrics (95% CI)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)

        _METRIC_LABELS = {
            "tir_70_180": "TIR 70-180 mg/dL (%)",
            "tir_below_70": "Time <70 mg/dL (%)",
            "tir_below_54": "Time <54 mg/dL (%)",
            "tir_above_180": "Time >180 mg/dL (%)",
            "mean_glucose": "Mean Glucose (mg/dL)",
            "cv": "Coefficient of Variation (%)",
            "gmi": "Glucose Management Indicator (%)",
        }

        for key, stats in aggregate_metrics.items():
            label = _METRIC_LABELS.get(key, key)
            line = f"  {label}: {stats['mean']:.1f}  [{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]"
            pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")

        # --- Safety summary ---
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Population Safety Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)

        si = aggregate_safety.get("safety_index", {})
        if si:
            pdf.cell(
                0, 6,
                f"  Safety Index: {si['mean']:.1f}  [{si['ci_lower']:.1f}, {si['ci_upper']:.1f}]",
                new_x="LMARGIN", new_y="NEXT",
            )

        grade_dist = aggregate_safety.get("grade_distribution", {})
        if grade_dist:
            n = len(summary_df)
            for grade in sorted(grade_dist):
                count = grade_dist[grade]
                pct = count / n * 100 if n else 0
                pdf.cell(0, 6, f"  Grade {grade}: {count} ({pct:.1f}%)", new_x="LMARGIN", new_y="NEXT")

        etr = aggregate_safety.get("early_termination_rate")
        if etr is not None:
            pdf.cell(0, 6, f"  Early termination rate: {etr * 100:.1f}%", new_x="LMARGIN", new_y="NEXT")

        # --- Plots ---
        for plot_label, plot_path in plots.items():
            if Path(plot_path).exists():
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 9, plot_label, new_x="LMARGIN", new_y="NEXT")
                pdf.image(plot_path, x=10, w=190)

        pdf.output(output_path)
        return output_path

    # ------------------------------------------------------------------
    def _generate_plots(self, df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
        try:
            apply_plot_style()
        except ImportError:
            pass

        plots: Dict[str, str] = {}

        # 1. TIR distribution
        if "tir_70_180" in df.columns:
            path = str(output_dir / "_plot_tir_distribution.png")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df["tir_70_180"].dropna(), bins=30, color=IINTS_BLUE, edgecolor="white", alpha=0.85)
            ax.axvline(70, color=IINTS_RED, linestyle="--", linewidth=1.4, label="Target: 70 %")
            mean_tir = df["tir_70_180"].mean()
            ax.axvline(mean_tir, color=IINTS_TEAL, linestyle="-", linewidth=1.4, label=f"Mean: {mean_tir:.1f} %")
            ax.set_xlabel("TIR 70-180 (%)")
            ax.set_ylabel("Patient count")
            ax.set_title("Time-in-Range Distribution Across Population")
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plots["TIR Distribution"] = path

        # 2. Safety index distribution
        if "safety_index_score" in df.columns:
            path = str(output_dir / "_plot_safety_index.png")
            scores = df["safety_index_score"].dropna()
            if len(scores) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(scores, bins=30, color=IINTS_BLUE, edgecolor="white", alpha=0.85)
                ax.axvline(75, color="#f4a261", linestyle="--", linewidth=1.4, label="Grade B (75)")
                ax.axvline(90, color=IINTS_TEAL, linestyle="--", linewidth=1.4, label="Grade A (90)")
                ax.set_xlabel("Safety Index Score")
                ax.set_ylabel("Patient count")
                ax.set_title("Safety Index Distribution Across Population")
                ax.legend()
                fig.tight_layout()
                fig.savefig(path, dpi=150)
                plt.close(fig)
                plots["Safety Index Distribution"] = path

        # 3. ISF vs TIR scatter
        if "insulin_sensitivity" in df.columns and "tir_70_180" in df.columns:
            path = str(output_dir / "_plot_isf_vs_tir.png")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(
                df["insulin_sensitivity"], df["tir_70_180"],
                alpha=0.45, s=18, color=IINTS_BLUE, edgecolors="none",
            )
            ax.set_xlabel("ISF (mg/dL per Unit)")
            ax.set_ylabel("TIR 70-180 (%)")
            ax.set_title("Insulin Sensitivity vs Time-in-Range")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plots["ISF vs TIR"] = path

        # 4. Hypo-risk box plot
        hypo_cols = [c for c in ["tir_below_70", "tir_below_54"] if c in df.columns]
        if hypo_cols:
            path = str(output_dir / "_plot_hypo_risk.png")
            fig, ax = plt.subplots(figsize=(6, 4))
            data = [df[c].dropna().values for c in hypo_cols]
            labels = ["TBR <70 mg/dL (%)", "TBR <54 mg/dL (%)"][:len(data)]
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], [IINTS_BLUE, IINTS_RED]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel("Percentage of time")
            ax.set_title("Hypoglycaemia Risk Distribution")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plots["Hypoglycaemia Risk"] = path

        return plots
