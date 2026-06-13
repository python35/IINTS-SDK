import os
import tempfile
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iints-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from matplotlib.patches import Patch

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.utils.academic_artifacts import (
    add_academic_footer,
    add_academic_header,
    add_academic_section,
    add_key_value_table,
    add_metric_cards,
    setup_academic_pdf,
)
from iints.utils.plotting import apply_plot_style

logger = logging.getLogger("iints")

AGP_TIR_STACK = [
    ("very_low_lt_54", "Very Low", "<54 mg/dL", "#8B0000", "white"),
    ("low_54_69", "Low", "54-69 mg/dL", "#D32F2F", "white"),
    ("target_70_180", "Target", "70-180 mg/dL", "#4CAF50", "black"),
    ("high_181_250", "High", "181-250 mg/dL", "#FFD54F", "black"),
    ("very_high_gt_250", "Very High", ">250 mg/dL", "#F57C00", "black"),
]


class ClinicalReportGenerator:
    """Generate a clean, publication-ready PDF report."""

    def __init__(self) -> None:
        self.metrics_calculator = ClinicalMetricsCalculator()

    def _resolve_logo_path(self) -> Optional[Path]:
        candidates = []
        # Package asset (installed)
        candidates.append(Path(__file__).resolve().parent.parent / "assets" / "iints_logo.png")
        # Repo root img/ (dev)
        candidates.append(Path(__file__).resolve().parents[3] / "img" / "iints_logo.png")
        for path in candidates:
            if path.exists():
                return path
        return None

    def _render_logo(self, pdf: FPDF) -> None:
        logo_path = self._resolve_logo_path()
        if not logo_path:
            return
        try:
            logo_width = 36
            x_pos = pdf.w - pdf.r_margin - logo_width
            y_pos = 6
            pdf.image(str(logo_path), x=x_pos, y=y_pos, w=logo_width)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug("Skipping report logo render: %s", exc)
            return

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

    @staticmethod
    def _infer_step_minutes(df: pd.DataFrame) -> float:
        if "time_minutes" not in df.columns or len(df) < 2:
            return 5.0
        diffs = pd.to_numeric(df["time_minutes"], errors="coerce").diff().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            return 5.0
        return float(diffs.median())

    @staticmethod
    def _format_duration(minutes: float) -> str:
        minutes = max(float(minutes), 0.0)
        if minutes >= 60:
            hours = int(minutes // 60)
            mins = int(round(minutes % 60))
            if mins:
                return f"{hours}h {mins}m"
            return f"{hours}h"
        return f"{int(round(minutes))}m"

    def _prepare_agp_frame(self, simulation_data: pd.DataFrame) -> pd.DataFrame:
        if "glucose_actual_mgdl" not in simulation_data.columns:
            raise ValueError("AGP report requires a glucose_actual_mgdl column.")

        df = simulation_data.copy()
        if "time_minutes" not in df.columns:
            if "timestamp" in df.columns:
                df["time_minutes"] = pd.to_numeric(df["timestamp"], errors="coerce")
            else:
                step = 5.0
                df["time_minutes"] = np.arange(len(df), dtype=float) * step

        df["time_minutes"] = pd.to_numeric(df["time_minutes"], errors="coerce")
        df["glucose_actual_mgdl"] = pd.to_numeric(df["glucose_actual_mgdl"], errors="coerce")
        df = df.dropna(subset=["time_minutes", "glucose_actual_mgdl"]).sort_values("time_minutes")
        if df.empty:
            raise ValueError("AGP report requires at least one valid glucose reading.")

        start_minute = float(df["time_minutes"].min())
        df["relative_minutes"] = df["time_minutes"] - start_minute
        df["day_index"] = np.floor(df["relative_minutes"] / 1440.0).astype(int)
        df["minute_of_day"] = df["relative_minutes"] % 1440.0
        return df.reset_index(drop=True)

    def _agp_summary(self, df: pd.DataFrame, *, target_low: float, target_high: float) -> Dict[str, Any]:
        glucose = df["glucose_actual_mgdl"].astype(float)
        step_minutes = self._infer_step_minutes(df)
        active_minutes = float(len(df)) * step_minutes
        span_minutes = max(float(df["relative_minutes"].max()) + step_minutes, step_minutes)
        expected_points = max(int(round(span_minutes / step_minutes)), len(df))
        data_active_pct = min(100.0, (float(len(df)) / float(expected_points)) * 100.0)
        report_days = max(1.0, span_minutes / 1440.0)

        very_low = float((glucose < 54).mean() * 100.0)
        low = float(((glucose >= 54) & (glucose < target_low)).mean() * 100.0)
        target = float(((glucose >= target_low) & (glucose <= target_high)).mean() * 100.0)
        high = float(((glucose > target_high) & (glucose <= 250)).mean() * 100.0)
        very_high = float((glucose > 250).mean() * 100.0)

        metrics = self.metrics_calculator.calculate(glucose=glucose, duration_hours=active_minutes / 60.0).to_dict()
        return {
            "report_days": int(np.ceil(report_days)),
            "data_active_pct": data_active_pct,
            "active_minutes": active_minutes,
            "step_minutes": step_minutes,
            "average_glucose_mgdl": float(glucose.mean()),
            "gmi_pct": float(metrics.get("gmi", 0.0)),
            "glucose_variability_cv_pct": float(metrics.get("cv", 0.0)),
            "time_ranges_pct": {
                "very_high_gt_250": very_high,
                "high_181_250": high,
                "target_70_180": target,
                "low_54_69": low,
                "very_low_lt_54": very_low,
            },
            "time_ranges_minutes": {
                "very_high_gt_250": very_high / 100.0 * active_minutes,
                "high_181_250": high / 100.0 * active_minutes,
                "target_70_180": target / 100.0 * active_minutes,
                "low_54_69": low / 100.0 * active_minutes,
                "very_low_lt_54": very_low / 100.0 * active_minutes,
            },
            "target_low_mgdl": target_low,
            "target_high_mgdl": target_high,
            "reading_count": int(len(df)),
        }

    def _plot_agp_profile(
        self,
        df: pd.DataFrame,
        output_path: Path,
        *,
        target_low: float,
        target_high: float,
        svg_path: Optional[Path] = None,
    ) -> None:
        apply_plot_style()
        step = max(1, int(round(self._infer_step_minutes(df))))
        working = df.copy()
        working["minute_bin"] = ((working["minute_of_day"] / step).round().astype(int) * step) % 1440
        grouped = working.groupby("minute_bin")["glucose_actual_mgdl"]
        q = grouped.quantile(np.array([0.05, 0.25, 0.5, 0.75, 0.95])).unstack()
        bins = np.arange(0, 1440, step)
        q = q.reindex(bins).interpolate(limit_direction="both")

        x = q.index.to_numpy(dtype=float) / 60.0
        p05 = q[0.05].to_numpy(dtype=float)
        p25 = q[0.25].to_numpy(dtype=float)
        p50 = q[0.5].to_numpy(dtype=float)
        p75 = q[0.75].to_numpy(dtype=float)
        p95 = q[0.95].to_numpy(dtype=float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [4.5, 1.4]})

        # Panel 1: AGP Curve
        ax1.fill_between(x, p05, p95, color="#D0D0D0", label="5-95%", alpha=0.5, linewidth=0)
        ax1.fill_between(x, p25, p75, color="#A0A0A0", label="25-75%", alpha=0.8, linewidth=0)
        ax1.plot(x, p50, color="#103b5c", linewidth=2.5, label="Median")

        ax1.axhspan(target_low, target_high, color="#2ca02c", alpha=0.1)
        ax1.axhline(target_low, color="#2ca02c", linestyle=":", linewidth=1)
        ax1.axhline(target_high, color="#2ca02c", linestyle=":", linewidth=1)

        ax1.set_xlim(0, 24)
        ymax = max(350.0, float(np.nanmax(p95)) + 20.0)
        ax1.set_ylim(40, ymax)
        ax1.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
        ax1.set_xticklabels(["12a", "3a", "6a", "9a", "12p", "3p", "6p", "9p", "12a"])
        ax1.set_ylabel("Glucose (mg/dL)", fontweight="bold")

        report_day_count = int(max(1, df["day_index"].nunique()))
        title = "Single-Day Glucose Profile" if report_day_count < 2 else "Ambulatory Glucose Profile (AGP)"
        ax1.set_title(title, fontweight="bold")

        # Panel 2: TIR Bar
        summary = self._agp_summary(df, target_low=target_low, target_high=target_high)
        pct_map = summary["time_ranges_pct"]
        bottom = 0.0
        legend_handles = []
        for key, label, range_text, color, text_color in AGP_TIR_STACK:
            val = float(pct_map.get(key, 0.0))
            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="none",
                    label=f"{label} ({range_text}) {val:.0f}%",
                )
            )
            if val > 0:
                ax2.bar([0], [val], bottom=[bottom], color=color, width=0.48)
                if val >= 6:
                    ax2.text(
                        0,
                        bottom + val / 2.0,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                        fontsize=9,
                    )
            bottom += val

        ax2.set_ylim(0, 100)
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xticks([0])
        ax2.set_xticklabels(["TIR"])
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_ylabel("% Time", fontweight="bold")
        ax2.set_title("Time in Range", fontweight="bold")
        ax2.legend(
            handles=list(reversed(legend_handles)),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=7,
            borderaxespad=0.0,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        if svg_path is not None:
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_daily_profiles(
        self,
        df: pd.DataFrame,
        output_path: Path,
        *,
        target_low: float,
        target_high: float,
        max_days: int = 14,
        svg_path: Optional[Path] = None,
    ) -> None:
        apply_plot_style()
        day_ids = sorted(df["day_index"].unique().tolist())[:max_days]
        if not day_ids:
            day_ids = [0]
        # A single-day AGP should not render as one tiny panel in a 7-day grid.
        # Keep the 7-column layout for longer reports, but let short reports breathe.
        if len(day_ids) <= 3:
            cols = len(day_ids)
        else:
            cols = 7
        rows = int(np.ceil(len(day_ids) / cols))
        fig_height = max(1.9, rows * 1.45 if cols >= 7 else 2.35)
        fig, axes = plt.subplots(rows, cols, figsize=(10, fig_height), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")

        for idx, day_id in enumerate(day_ids):
            ax = axes[idx // cols][idx % cols]
            ax.axis("on")
            daily = df[df["day_index"] == day_id].copy()
            x = daily["minute_of_day"].to_numpy(dtype=float) / 60.0
            y = daily["glucose_actual_mgdl"].to_numpy(dtype=float)
            ax.axhspan(target_low, target_high, color="#F5F5F5", alpha=1.0)
            ax.fill_between(x, target_high, y, where=y > target_high, color="#FFB300", alpha=0.7)
            ax.fill_between(x, y, target_low, where=y < target_low, color="#D32F2F", alpha=0.7)
            ax.plot(x, y, color="#212121", linewidth=1.2)
            ax.set_xlim(0, 24)
            ax.set_ylim(40, max(260, float(np.nanmax(y)) + 20 if len(y) else 260))
            ax.set_xticks([0, 12, 24])
            ax.set_xticklabels(["12a", "12p", "12a"], fontsize=6)
            ax.set_yticks([])
            ax.set_title(f"Day {int(day_id) + 1}", fontsize=8, fontweight="bold")

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#888888")
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_color("#888888")
            ax.spines['bottom'].set_linewidth(0.5)

        fig.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        if svg_path is not None:
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _extract_xai_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract explainable-event strings into a structured export list."""
        if "explainable_events" not in df.columns:
            return []

        events: List[Dict[str, Any]] = []
        for row_index, row in df.dropna(subset=["explainable_events"]).iterrows():
            raw_events = row.get("explainable_events")
            if isinstance(raw_events, str):
                parts = [part.strip() for part in raw_events.split(";") if part.strip()]
            elif isinstance(raw_events, (list, tuple)):
                parts = [str(part).strip() for part in raw_events if str(part).strip()]
            else:
                continue

            time_minutes = row.get("time_minutes")
            try:
                normalized_time = None if pd.isna(time_minutes) else float(time_minutes)
            except (TypeError, ValueError):
                normalized_time = None

            for event in parts:
                events.append(
                    {
                        "row_index": str(row_index),
                        "time_minutes": normalized_time,
                        "event": event,
                    }
                )
        return events

    @staticmethod
    def _section_header(pdf: FPDF, title: str, width: float = 0) -> None:
        pdf.set_fill_color(35, 55, 67)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(width, 5.5, title, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(25, 35, 43)

    @staticmethod
    def _range_bar(pdf: FPDF, x: float, y: float, width: float, height: float, summary: Dict[str, Any]) -> None:
        ranges = [
            ("very_high_gt_250", (230, 126, 34), "Very High", ">250 mg/dL"),
            ("high_181_250", (248, 200, 48), "High", "181-250 mg/dL"),
            ("target_70_180", (86, 160, 88), "Target Range", "70-180 mg/dL"),
            ("low_54_69", (204, 57, 57), "Low", "54-69 mg/dL"),
            ("very_low_lt_54", (126, 29, 33), "Very Low", "<54 mg/dL"),
        ]
        total_h = height
        pct_map = summary["time_ranges_pct"]

        current_y = y
        for key, color, _label, _range_text in ranges:
            pct = float(pct_map.get(key, 0.0))
            part_h = total_h * pct / 100.0
            pdf.set_fill_color(*color)
            if part_h > 0:
                pdf.rect(x, current_y, width, part_h, style="F")
            current_y += part_h
        pdf.set_draw_color(85, 95, 103)
        pdf.rect(x, y, width, height)

        label_x = x + width + 7
        row_y = y - 1
        pdf.set_font("Helvetica", "", 7.3)
        for idx, (key, color, label, range_text) in enumerate(ranges):
            pct = float(pct_map.get(key, 0.0))
            minutes = float(summary["time_ranges_minutes"].get(key, 0.0))
            line_y = row_y + idx * 10.8
            pdf.set_fill_color(*color)
            pdf.rect(label_x, line_y + 1.4, 3.4, 3.4, style="F")
            pdf.set_xy(label_x + 5, line_y)
            pdf.set_font("Helvetica", "B", 7.3)
            pdf.cell(24, 3.6, label)
            pdf.set_font("Helvetica", "", 7.0)
            pdf.cell(22, 3.6, range_text)
            pdf.set_font("Helvetica", "B", 7.3)
            pdf.cell(12, 3.6, f"{pct:.0f}%", align="R")
            pdf.set_font("Helvetica", "", 6.8)
            pdf.cell(0, 3.6, f" ({ClinicalReportGenerator._format_duration(minutes)})")

    def generate_agp_pdf(
        self,
        simulation_data: pd.DataFrame,
        output_path: str,
        *,
        title: str = "IINTS Research AGP-Style Report",
        subject_name: str = "Research simulation",
        safety_report: Optional[Dict[str, Any]] = None,
        target_low: float = 70.0,
        target_high: float = 180.0,
        summary_json_path: Optional[str] = None,
    ) -> str:
        """Generate a one-page AGP-style research report from dense CGM/simulation data."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df = self._prepare_agp_frame(simulation_data)
        summary = self._agp_summary(df, target_low=target_low, target_high=target_high)
        safety_report = safety_report or {}

        if summary_json_path:
            summary_file = Path(summary_json_path)
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            agp_plot = tmp_dir_path / "agp_profile.png"
            daily_plot = tmp_dir_path / "daily_profiles.png"
            self._plot_agp_profile(df, agp_plot, target_low=target_low, target_high=target_high)
            self._plot_daily_profiles(df, daily_plot, target_low=target_low, target_high=target_high)

            pdf = FPDF(format="A4")
            pdf.set_auto_page_break(auto=False)
            pdf.add_page()
            pdf.set_margins(8, 8, 8)

            header_y = 8
            title_w = 121
            meta_x = 134
            pdf.set_xy(8, header_y)
            title_size = 15 if len(title) <= 72 else 13
            pdf.set_text_color(25, 35, 43)
            pdf.set_font("Helvetica", "B", title_size)
            pdf.multi_cell(title_w, 6.2, title, align="L")
            title_bottom = pdf.get_y()

            pdf.set_xy(meta_x, header_y)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(25, 35, 43)
            pdf.cell(0, 4, "Name", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_xy(meta_x, header_y + 4.5)
            pdf.set_font("Helvetica", "", 7.2)
            pdf.multi_cell(64, 3.5, subject_name)
            meta_bottom = pdf.get_y()
            pdf.set_xy(meta_x, max(meta_bottom + 1.5, header_y + 12))
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(0, 4, "Report type", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_xy(meta_x, pdf.get_y())
            pdf.set_font("Helvetica", "", 7.2)
            pdf.multi_cell(64, 3.5, "Research simulation / educational")
            meta_bottom = pdf.get_y()

            header_bottom = max(title_bottom, meta_bottom, header_y + 23)
            pdf.set_draw_color(170, 181, 188)
            pdf.line(8, header_bottom + 1.5, 202, header_bottom + 1.5)
            pdf.set_y(header_bottom + 5)

            left_x = 8
            right_x = 112
            left_w = 98
            right_w = 90
            top_y = pdf.get_y()
            pdf.set_xy(left_x, top_y)
            self._section_header(pdf, "GLUCOSE STATISTICS AND TARGETS", left_w)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(0, 5, f"Report period: {summary['report_days']} day(s)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 5, f"Time CGM/simulation active: {summary['data_active_pct']:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(48, 5, "Glucose ranges")
            pdf.cell(36, 5, "Targets", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 7.5)
            target_rows = [
                (f"Target range {target_low:.0f}-{target_high:.0f} mg/dL", "Greater than 70%"),
                ("Below 70 mg/dL", "Less than 4%"),
                ("Below 54 mg/dL", "Less than 1%"),
                ("Above 180 mg/dL", "Less than 25%"),
                ("Above 250 mg/dL", "Less than 5%"),
            ]
            for label, target in target_rows:
                pdf.cell(52, 4, label)
                pdf.cell(42, 4, target, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(55, 5, "Average glucose")
            pdf.cell(0, 5, f"{summary['average_glucose_mgdl']:.0f} mg/dL", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(55, 5, "Glucose Management Indicator (GMI)")
            pdf.cell(0, 5, f"{summary['gmi_pct']:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(55, 5, "Glucose variability (%CV)")
            pdf.cell(0, 5, f"{summary['glucose_variability_cv_pct']:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            if safety_report:
                pdf.cell(55, 5, "Safety interventions")
                pdf.cell(0, 5, str(safety_report.get("bolus_interventions_count", 0)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_xy(right_x, top_y)
            self._section_header(pdf, "TIME IN RANGES", right_w)
            self._range_bar(pdf, right_x + 3, top_y + 12, 13, 55, summary)

            agp_y = max(100, top_y + 74)
            pdf.set_xy(8, agp_y)
            self._section_header(pdf, "AMBULATORY GLUCOSE PROFILE (AGP-STYLE)")
            pdf.set_font("Helvetica", "", 7)
            if summary["report_days"] < 2:
                agp_note = (
                    "Single-day AGP-style view: percentile bands collapse toward the visible trace. "
                    "Use multi-day CGM/simulation data for a full AGP percentile profile."
                )
            else:
                agp_note = (
                    "AGP-style summary of glucose values over the report period, with median (50%) "
                    "and percentile bands shown as a single modal day."
                )
            pdf.multi_cell(
                0,
                4,
                agp_note,
            )
            agp_image_y = pdf.get_y() + 2
            pdf.image(str(agp_plot), x=12, y=agp_image_y, w=186)

            daily_y = agp_image_y + 78
            pdf.set_xy(8, daily_y)
            self._section_header(pdf, "DAILY GLUCOSE PROFILES")
            daily_image_y = pdf.get_y() + 4
            pdf.image(str(daily_plot), x=10, y=daily_image_y, w=190)

            pdf.set_xy(8, 285)
            pdf.set_font("Helvetica", "I", 7)
            pdf.multi_cell(
                0,
                3.5,
                "IINTS-AF research report. Not a medical device report, diagnosis, or treatment recommendation. "
                "Targets follow common CGM consensus framing for research interpretation.",
            )
            pdf.output(str(output_file))

        return str(output_file)

    def export_agp_assets(
        self,
        simulation_data: pd.DataFrame,
        output_dir: str,
        *,
        subject_name: str = "Research simulation",
        target_low: float = 70.0,
        target_high: float = 180.0,
        summary_json_path: Optional[str] = None,
        export_svg: bool = True,
    ) -> Dict[str, str]:
        """Export AGP-style PNG/SVG assets and summary JSON without creating a PDF."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        df = self._prepare_agp_frame(simulation_data)
        summary = self._agp_summary(df, target_low=target_low, target_high=target_high)
        summary["subject_name"] = subject_name

        agp_plot = output_path / "agp_profile.png"
        daily_plot = output_path / "daily_profiles.png"
        agp_svg = output_path / "agp_profile.svg" if export_svg else None
        daily_svg = output_path / "daily_profiles.svg" if export_svg else None
        summary_file = Path(summary_json_path) if summary_json_path else output_path / "agp_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        self._plot_agp_profile(
            df,
            agp_plot,
            target_low=target_low,
            target_high=target_high,
            svg_path=agp_svg,
        )
        self._plot_daily_profiles(
            df,
            daily_plot,
            target_low=target_low,
            target_high=target_high,
            svg_path=daily_svg,
        )
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        xai_file = output_path / "xai_events.txt"
        xai_json_file = output_path / "xai_events.json"
        xai_events = self._extract_xai_events(df)

        with open(xai_file, "w", encoding="utf-8") as f:
            f.write("=== IINTS Explainable AI (XAI) Events Log ===\n")
            if xai_events:
                for entry in xai_events:
                    f.write(f"- {entry['event']}\n")
            else:
                f.write("No significant XAI events detected.\n")
        xai_json_file.write_text(json.dumps(xai_events, indent=2), encoding="utf-8")

        outputs = {
            "agp_profile_png": str(agp_plot),
            "daily_profiles_png": str(daily_plot),
            "summary_json": str(summary_file),
            "xai_events_txt": str(xai_file),
            "xai_events_json": str(xai_json_file),
        }
        if agp_svg is not None and daily_svg is not None:
            outputs["agp_profile_svg"] = str(agp_svg)
            outputs["daily_profiles_svg"] = str(daily_svg)
        return outputs

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
            setup_academic_pdf(pdf, title=title)
            pdf.add_page()
            self._render_logo(pdf)

            duration_hours = float(simulation_data["time_minutes"].max() / 60.0)
            add_academic_header(
                pdf,
                title,
                metadata={
                    "Duration": f"{duration_hours:.1f} h",
                    "Data points": len(simulation_data),
                },
            )

            add_academic_section(pdf, "Clinical metrics")
            add_metric_cards(
                pdf,
                [
                    ("TIR 70-180", f"{metrics.get('tir_70_180', 0):.1f}%"),
                    ("Time below 70", f"{metrics.get('tir_below_70', 0):.1f}%"),
                    ("Time above 180", f"{metrics.get('tir_above_180', 0):.1f}%"),
                    ("CV", f"{metrics.get('cv', 0):.1f}%"),
                    ("GMI", f"{metrics.get('gmi', 0):.1f}%"),
                    ("Mean glucose", f"{metrics.get('mean_glucose', 0):.1f} mg/dL"),
                ],
            )

            add_academic_section(pdf, "Safety summary")
            add_key_value_table(
                pdf,
                [
                    ("Total violations", str(safety_report.get("total_violations", 0))),
                    ("Bolus interventions", str(safety_report.get("bolus_interventions_count", 0))),
                    ("Terminated early", str(bool(safety_report.get("terminated_early", False)))),
                ],
            )
            top_reasons = self._top_safety_reasons(simulation_data)
            if top_reasons:
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6, "Top intervention reasons:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", "", 10)
                for reason, count in top_reasons.items():
                    pdf.cell(0, 5, f"- {reason}: {count}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            baseline = safety_report.get("baseline_comparison")
            if baseline and baseline.get("rows"):
                pdf.ln(3)
                add_academic_section(pdf, "Head-to-head comparison")
                pdf.set_font("Helvetica", "B", 9)
                col_widths = [52, 26, 26, 26, 30]
                headers = ["Algorithm", "TIR 70-180", "Time <70", "Time >180", "Safety Overrides"]
                for idx, header in enumerate(headers):
                    pdf.cell(col_widths[idx], 6, header, border=1, align="C")
                pdf.ln()

                pdf.set_font("Helvetica", "", 9)
                for row in baseline["rows"]:
                    pdf.cell(col_widths[0], 6, str(row.get("algorithm", ""))[:24], border=1)
                    pdf.cell(col_widths[1], 6, f"{row.get('tir_70_180', 0):.1f}%", border=1, align="C")
                    pdf.cell(col_widths[2], 6, f"{row.get('tir_below_70', 0):.1f}%", border=1, align="C")
                    pdf.cell(col_widths[3], 6, f"{row.get('tir_above_180', 0):.1f}%", border=1, align="C")
                    pdf.cell(col_widths[4], 6, str(row.get("bolus_interventions", 0)), border=1, align="C")
                    pdf.ln()

            pdf.ln(3)
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(
                0,
                4,
                "Research-use framing follows ADA 2026 glycemic goals and ATTD Time-in-Range consensus. "
                "See docs/EVIDENCE_BASE.md and `iints sources` for full citations.",
            )

            pdf.ln(4)
            add_academic_section(pdf, "Glucose trace")
            pdf.image(str(glucose_plot), w=180)

            pdf.ln(4)
            add_academic_section(pdf, "Insulin delivery")
            pdf.image(str(insulin_plot), w=180)

            add_academic_footer(pdf)

            pdf.output(str(output_file))

        return str(output_file)

    def generate_demo_pdf(
        self,
        simulation_data: pd.DataFrame,
        safety_report: Dict[str, Any],
        output_path: str,
        title: str = "IINTS-AF Demo Report",
    ) -> str:
        """
        Generate a Maker Faire / demo-friendly PDF with bold visuals and minimal text.
        """
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
            setup_academic_pdf(pdf, title=title)
            pdf.add_page()
            self._render_logo(pdf)

            add_academic_header(
                pdf,
                title,
                subtitle="Demo-facing research summary - pre-clinical simulation only",
                metadata={"Duration": f"{simulation_data['time_minutes'].max()/60:.1f} h"},
            )

            # Metric tiles
            tiles = [
                ("TIR 70-180", f"{metrics.get('tir_70_180', 0):.1f}%"),
                ("Time <70", f"{metrics.get('tir_below_70', 0):.1f}%"),
                ("GMI", f"{metrics.get('gmi', 0):.1f}%"),
                ("CV", f"{metrics.get('cv', 0):.1f}%"),
                ("Overrides", str(safety_report.get("bolus_interventions_count", 0))),
                ("Violations", str(safety_report.get("total_violations", 0))),
            ]

            add_metric_cards(pdf, tiles)

            pdf.ln(4)
            add_academic_section(pdf, "Glucose trace")
            pdf.image(str(glucose_plot), w=180)

            pdf.ln(4)
            add_academic_section(pdf, "Insulin delivery")
            pdf.image(str(insulin_plot), w=180)

            top_reasons = self._top_safety_reasons(simulation_data)
            if top_reasons:
                pdf.ln(4)
                add_academic_section(pdf, "Top safety interventions")
                pdf.set_font("Helvetica", "", 10)
                for reason, count in top_reasons.items():
                    pdf.cell(0, 5, f"- {reason}: {count}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            add_academic_footer(pdf, note="Method references: docs/EVIDENCE_BASE.md | CLI: `iints sources`. Not for treatment decisions.")

            pdf.output(str(output_file))

        return str(output_file)
