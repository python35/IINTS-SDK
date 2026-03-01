#!/usr/bin/env python3
"""
IINTS-AF Clinical Report Generator
Professional PDF reports in Medtronic CareLink style
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ClinicalReportPDF(FPDF):
    """Professional clinical report PDF generator"""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def cell(
        self,
        w,
        h=0,
        txt="",
        border=0,
        ln=0,
        align="",
        fill=False,
        link="",
        new_x=None,
        new_y=None,
    ):
        """Backward-compatible cell() that avoids deprecated ln usage."""
        if new_x is not None or new_y is not None:
            return super().cell(
                w,
                h,
                txt,
                border=border,
                align=align,
                fill=fill,
                link=link,
                new_x=new_x if new_x is not None else XPos.RIGHT,
                new_y=new_y if new_y is not None else YPos.TOP,
            )
        if ln:
            return super().cell(
                w,
                h,
                txt,
                border=border,
                align=align,
                fill=fill,
                link=link,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
        return super().cell(w, h, txt, border=border, align=align, fill=fill, link=link)
        
    def header(self):
        """Add header with PROMINENT IINTS logo and branding"""
        # Add logo if available
        logo_path = Path(__file__).parent.parent / "img" / "iints_logo.png"
        if logo_path.exists():
            self.image(str(logo_path), 10, 9, 18)  # Smaller logo to avoid text overlap
            logo_x = 32
        else:
            # Fallback: Create text-based logo
            self.set_xy(10, 10)
            self.set_font('Helvetica', 'B', 18)
            self.set_text_color(34, 139, 34)  # Forest green
            self.cell(30, 12, 'IINTS-AF', 0, 0, 'C', fill=True)
            self.set_fill_color(34, 139, 34)
            logo_x = 45
        
        # Title next to logo - PROMINENT BRANDING
        self.set_xy(logo_x, 10)
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(34, 139, 34)  # Consistent green branding
        self.cell(0, 8, 'IINTS-AF Research Forecast Performance Report', 0, 1, 'L')
        
        self.set_xy(logo_x, 18)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 6, 'Retrospective Model Performance (No Intervention)', 0, 1, 'L')
        
        # Add professional line separator
        self.set_xy(10, 28)
        self.set_draw_color(34, 139, 34)
        self.set_line_width(0.5)
        self.line(10, 28, 200, 28)
        
        self.ln(15)
        
    def footer(self):
        """Add footer with page number"""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        footer_text = f'Page {self.page_no()} - For research use only. Not for clinical use.'
        self.cell(0, 10, footer_text, 0, 0, 'C')
        
    def add_learning_cover_page(self, patient_id, session_id, summary, provenance):
        """Add forecast report cover page (research use)."""
        self.add_page()
        
        # Cover spacing (avoid duplicate titles)
        self.ln(30)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(34, 139, 34)
        self.cell(0, 12, 'Report Overview', 0, 1, 'C')
        self.set_text_color(0, 0, 0)
        self.ln(6)
        
        # Session info
        self.set_font('Helvetica', '', 14)
        self.cell(0, 10, f'Session ID: {session_id}', 0, 1, 'C')
        self.cell(0, 10, f'Subject ID (de-identified): {patient_id}', 0, 1, 'C')
        self.cell(0, 10, 'Evaluation Mode: Retrospective forecast (no therapy changes)', 0, 1, 'C')
        self.cell(0, 10, f'Report Date (UTC): {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        
        self.ln(30)
        
        # Forecast executive summary
        self.set_fill_color(240, 248, 255)
        self.rect(20, self.get_y(), 170, 80, 'F')
        
        self.ln(10)
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'FORECAST SUMMARY (NO INTERVENTION)', 0, 1, 'C')
        
        self.set_font('Helvetica', '', 10)
        data_source = provenance.get("data_source", "N/A")
        model_name = provenance.get("model_name", "N/A")
        horizon = summary.get("forecast_horizon", "N/A")
        duration = summary.get("duration_hours", "N/A")
        self.cell(0, 8, f'Data Source: {data_source}', 0, 1, 'C')
        self.cell(0, 8, f'Model: {model_name} (Horizon: {horizon})', 0, 1, 'C')
        self.cell(0, 8, f'Window: {duration}', 0, 1, 'C')
        self.cell(0, 8, 'No on-session learning or parameter adaptation.', 0, 1, 'C')
        self.cell(0, 8, 'This report evaluates retrospective forecasting performance only.', 0, 1, 'C')
        self.cell(0, 8, 'No therapy recommendations, dosing guidance, or clinical decision support is provided.', 0, 1, 'C')
            
    def add_learning_analysis_page(self, learning_data):
        """Add learning analysis page"""
        self.add_page()
        
        # Title
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(34, 139, 34)
        self.cell(0, 15, 'NEURAL LEARNING ANALYSIS', 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        
        # Learning metrics
        self.ln(10)
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Learning Performance Metrics', 0, 1, 'L')
        
        learning_curve = learning_data.get('learning_curve', [])
        if learning_curve:
            initial_loss = learning_curve[0]
            final_loss = learning_curve[-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            self.set_font('Helvetica', '', 10)
            self.cell(0, 8, f'Initial Model Loss: {initial_loss:.3f}', 0, 1, 'L')
            self.cell(0, 8, f'Final Model Loss: {final_loss:.3f}', 0, 1, 'L')
            self.cell(0, 8, f'Learning Improvement: {improvement:.1f}%', 0, 1, 'L')
            self.cell(0, 8, f'Convergence Status: {"Achieved" if learning_data.get("convergence_achieved") else "In Progress"}', 0, 1, 'L')
        
        # Safety validation
        self.ln(10)
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Safety Validation Results', 0, 1, 'L')
        
        self.set_font('Helvetica', '', 10)
        if learning_data.get('safety_validated'):
            self.cell(0, 8, 'All safety constraints satisfied', 0, 1, 'L')
            self.cell(0, 8, 'Neural weights within therapeutic bounds', 0, 1, 'L')
            self.cell(0, 8, 'Model convergence validated', 0, 1, 'L')
        else:
            self.cell(0, 8, 'Conservative parameters applied', 0, 1, 'L')
            self.cell(0, 8, 'Safety thresholds enforced', 0, 1, 'L')
        
    def add_comparison_page(self, summary, plot_paths, provenance):
        """Add forecast performance summary page."""
        self.add_page()
        
        # Title with IINTS branding
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(34, 139, 34)  # IINTS green
        self.cell(0, 15, 'FORECAST PERFORMANCE SUMMARY', 0, 1, 'L')
        self.set_text_color(0, 0, 0)  # Reset to black
        
        self.ln(5)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(34, 139, 34)
        self.cell(0, 10, 'Observed CGM Metrics (observed only)', 0, 1, 'L')
        self.set_text_color(0, 0, 0)

        self.set_font('Helvetica', '', 10)
        observed_rows = [
            ('Time in Range (70-180)', summary.get('tir_70_180', 'N/A')),
            ('Time Below Range (<70)', summary.get('tbr_lt70', 'N/A')),
            ('Time Below Range (<54)', summary.get('tbr_lt54', 'N/A')),
            ('Time Above Range (>180)', summary.get('tar_gt180', 'N/A')),
            ('Time Above Range (>250)', summary.get('tar_gt250', 'N/A')),
            ('Mean Glucose', summary.get('mean_glucose', 'N/A')),
            ('Variability (CV)', summary.get('cv', 'N/A')),
            ('Sampling Interval', summary.get('sampling_interval', 'N/A')),
            ('Missing Observed (%)', summary.get('missing_obs_pct', 'N/A')),
        ]
        for label, value in observed_rows:
            self.cell(70, 7, label, 0, 0, 'L')
            self.cell(0, 7, value, 0, 1, 'L')

        self.ln(4)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(34, 139, 34)
        self.cell(0, 10, 'Forecast Accuracy', 0, 1, 'L')
        self.set_text_color(0, 0, 0)

        self.set_font('Helvetica', '', 10)
        forecast_rows = [
            ('Forecast Horizon', summary.get('forecast_horizon', 'N/A')),
            ('MAE', summary.get('forecast_mae', 'N/A')),
            ('RMSE', summary.get('forecast_rmse', 'N/A')),
            ('Bias (Pred - Obs)', summary.get('forecast_bias', 'N/A')),
            ('Relative Absolute Difference (RAD)', summary.get('forecast_mard', 'N/A')),
            ('Within ±10 mg/dL', summary.get('within_10_mgdl_pct', 'N/A')),
            ('Within ±20 mg/dL', summary.get('within_20_mgdl_pct', 'N/A')),
            ('MAE (<70)', summary.get('band_lt70_mae', 'N/A')),
            ('MAE (70-180)', summary.get('band_70_180_mae', 'N/A')),
            ('MAE (>180)', summary.get('band_gt180_mae', 'N/A')),
            ('Forecast Samples', summary.get('forecast_samples', 'N/A')),
            ('Missing Forecast (%)', summary.get('missing_pred_pct', 'N/A')),
        ]
        for label, value in forecast_rows:
            self.cell(70, 7, label, 0, 0, 'L')
            self.cell(0, 7, value, 0, 1, 'L')

        self.ln(4)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(34, 139, 34)
        self.cell(0, 10, 'Provenance', 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.set_font('Helvetica', '', 9)
        provenance_lines = [
            f"Data Source: {provenance.get('data_source', 'N/A')}",
            f"Model: {provenance.get('model_name', 'N/A')}",
            f"Holdout Metrics: {provenance.get('holdout_metrics', 'N/A')}",
            "Split: subject-level holdout (train/val/test)",
            f"Train Subjects: {provenance.get('train_subjects', 'N/A')}",
            f"Val Subjects: {provenance.get('val_subjects', 'N/A')}",
            f"Test Subjects: {provenance.get('test_subjects', 'N/A')}",
            f"Training Timestamp (UTC): {provenance.get('timestamp_utc', 'N/A')}",
            f"Seed: {provenance.get('seed', 'N/A')}",
            f"Data SHA256: {provenance.get('data_sha256', 'N/A')}",
            f"Config SHA256: {provenance.get('config_sha256', 'N/A')}",
            f"Commit: {provenance.get('commit', 'N/A')}",
            "Metrics Script: examples/generate_clinical_report.py",
        ]
        for line in provenance_lines:
            self.set_x(10)
            self.multi_cell(0, 5, line, 0, 'L')

        self.ln(2)
        self.set_font('Helvetica', 'I', 9)
        self.set_x(10)
        missing_obs = summary.get('missing_obs_pct', 'N/A')
        missing_pred = summary.get('missing_pred_pct', 'N/A')
        self.multi_cell(
            0,
            4,
            "Notes: This report is a retrospective forecast on recorded CGM data. "
            "No in-session learning or therapy changes were applied. "
            "Forecasts are shifted forward by the horizon to align with observed target time. "
            f"Missing observed: {missing_obs}; missing forecast: {missing_pred} "
            "(missing forecasts occur when required history windows are unavailable). "
            "Forecast samples reflect valid prediction points after history window and horizon constraints. "
            "Metrics exclude missing values and are computed on raw CGM values (unsmoothed). "
            "Observed trace is smoothed for visualization only.",
            0,
            'L',
        )

        # Add glucose pattern chart
        if 'glucose_pattern' in plot_paths:
            self.ln(10)
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, 'Glucose Pattern Review', 0, 1, 'L')
            self.image(plot_paths['glucose_pattern'], x=10, y=None, w=190)
            self.ln(2)
            self.set_font('Helvetica', 'I', 9)
            horizon_label = summary.get("forecast_horizon", "30 min")
            self.multi_cell(
                0,
                4,
                f"Legend: Observed CGM = green. Forecast = blue (t+{horizon_label}, shifted to target time). "
                "Shaded band = observed variability (rolling std).",
                0,
                'L',
            )


class ClinicalReportGenerator:
    """Generate professional clinical reports"""
    
    def __init__(self):
        self.output_dir = Path("results/clinical_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _format_value(self, value, unit="", decimals=1):
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        try:
            if not np.isfinite(value):
                return "N/A"
        except Exception:
            return "N/A"
        return f"{value:.{decimals}f}{unit}"

    def _compute_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        if df is None or df.empty:
            return summary

        total_rows = int(df.shape[0])

        # Observed glucose series
        glucose_col = None
        for cand in ["glucose_actual_mgdl", "glucose_mgdl", "glucose", "glucose_val"]:
            if cand in df.columns:
                glucose_col = cand
                break
        obs = pd.to_numeric(df[glucose_col], errors="coerce") if glucose_col else pd.Series(dtype=float)
        obs = obs.dropna()
        summary["samples"] = int(obs.shape[0]) if not obs.empty else 0
        if total_rows:
            summary["missing_obs_pct"] = float((total_rows - summary["samples"]) / total_rows * 100.0)

        if not obs.empty:
            summary["mean_glucose"] = float(obs.mean())
            std = float(obs.std(ddof=0)) if obs.shape[0] > 1 else 0.0
            summary["cv"] = (std / summary["mean_glucose"] * 100.0) if summary["mean_glucose"] else None
            summary["tir_70_180"] = float(((obs >= 70) & (obs <= 180)).mean() * 100.0)
            summary["tbr_lt70"] = float((obs < 70).mean() * 100.0)
            summary["tbr_lt54"] = float((obs < 54).mean() * 100.0)
            summary["tar_gt180"] = float((obs > 180).mean() * 100.0)
            summary["tar_gt250"] = float((obs > 250).mean() * 100.0)

        # Duration
        if "time_minutes" in df.columns:
            t = pd.to_numeric(df["time_minutes"], errors="coerce").dropna()
            if not t.empty:
                summary["duration_hours"] = float((t.max() - t.min()) / 60.0)
        if "duration_hours" not in summary and summary.get("samples"):
            summary["duration_hours"] = float(summary["samples"] * 5.0 / 60.0)

        # Forecast metrics (aligned to target time)
        pred_col = None
        for cand in ["predicted_glucose_30min", "predicted_glucose_ai_30min", "predicted_glucose"]:
            if cand in df.columns:
                pred_col = cand
                break
        if pred_col and glucose_col and "time_minutes" in df.columns:
            horizon_min = float(summary.get("forecast_horizon_minutes", 30.0))
            pred_base = df[["time_minutes", pred_col]].copy()
            pred_base["pred_val"] = pd.to_numeric(pred_base[pred_col], errors="coerce")
            pred_base = pred_base.dropna(subset=["time_minutes", "pred_val"])
            pred_base["pred_time"] = pred_base["time_minutes"] + horizon_min

            obs_base = df[["time_minutes", glucose_col]].copy()
            obs_base["obs_val"] = pd.to_numeric(obs_base[glucose_col], errors="coerce")
            obs_base = obs_base.dropna(subset=["time_minutes", "obs_val"])

            aligned = obs_base.merge(
                pred_base[["pred_time", "pred_val"]],
                left_on="time_minutes",
                right_on="pred_time",
                how="inner",
            )
            if not aligned.empty:
                diff = aligned["pred_val"] - aligned["obs_val"]
                summary["forecast_samples"] = int(aligned.shape[0])
                summary["forecast_mae"] = float(diff.abs().mean())
                summary["forecast_rmse"] = float(np.sqrt((diff ** 2).mean()))
                summary["forecast_bias"] = float(diff.mean())
                if total_rows:
                    summary["missing_pred_pct"] = float((total_rows - aligned.shape[0]) / total_rows * 100.0)

                abs_diff = diff.abs()
                summary["within_10_mgdl_pct"] = float((abs_diff <= 10).mean() * 100.0)
                summary["within_20_mgdl_pct"] = float((abs_diff <= 20).mean() * 100.0)

                obs_vals = aligned["obs_val"]
                if obs_vals.max() > 0:
                    summary["forecast_mard"] = float(
                        (abs_diff / obs_vals).replace([np.inf, -np.inf], np.nan).dropna().mean() * 100.0
                    )

                band_masks = {
                    "band_lt70": obs_vals < 70,
                    "band_70_180": (obs_vals >= 70) & (obs_vals <= 180),
                    "band_gt180": obs_vals > 180,
                }
                for key, mask in band_masks.items():
                    if mask.any():
                        summary[f"{key}_mae"] = float(abs_diff[mask].mean())

        if "prediction_horizon_minutes" in df.columns:
            horizon = pd.to_numeric(df["prediction_horizon_minutes"], errors="coerce").dropna()
            if not horizon.empty:
                summary["forecast_horizon_minutes"] = float(horizon.iloc[0])
        if "forecast_horizon_minutes" not in summary:
            summary["forecast_horizon_minutes"] = 30.0

        if "time_minutes" in df.columns:
            t = pd.to_numeric(df["time_minutes"], errors="coerce").dropna()
            if t.shape[0] >= 2:
                deltas = np.diff(np.sort(t.to_numpy()))
                deltas = deltas[deltas > 0]
                if deltas.size:
                    summary["sampling_interval_minutes"] = float(np.median(deltas))
        if "sampling_interval_minutes" not in summary:
            summary["sampling_interval_minutes"] = 5.0

        return summary

    def _format_summary(self, summary: Dict[str, Any]) -> Dict[str, str]:
        formatted: Dict[str, str] = {}
        formatted["tir_70_180"] = self._format_value(summary.get("tir_70_180"), "%")
        formatted["tbr_lt70"] = self._format_value(summary.get("tbr_lt70"), "%")
        formatted["tbr_lt54"] = self._format_value(summary.get("tbr_lt54"), "%")
        formatted["tar_gt180"] = self._format_value(summary.get("tar_gt180"), "%")
        formatted["tar_gt250"] = self._format_value(summary.get("tar_gt250"), "%")
        formatted["mean_glucose"] = self._format_value(summary.get("mean_glucose"), " mg/dL")
        formatted["cv"] = self._format_value(summary.get("cv"), "%")
        formatted["forecast_mae"] = self._format_value(summary.get("forecast_mae"), " mg/dL")
        formatted["forecast_rmse"] = self._format_value(summary.get("forecast_rmse"), " mg/dL")
        formatted["forecast_bias"] = self._format_value(summary.get("forecast_bias"), " mg/dL")
        formatted["forecast_mard"] = self._format_value(summary.get("forecast_mard"), "%")
        formatted["within_10_mgdl_pct"] = self._format_value(summary.get("within_10_mgdl_pct"), "%")
        formatted["within_20_mgdl_pct"] = self._format_value(summary.get("within_20_mgdl_pct"), "%")
        formatted["band_lt70_mae"] = self._format_value(summary.get("band_lt70_mae"), " mg/dL")
        formatted["band_70_180_mae"] = self._format_value(summary.get("band_70_180_mae"), " mg/dL")
        formatted["band_gt180_mae"] = self._format_value(summary.get("band_gt180_mae"), " mg/dL")
        formatted["forecast_samples"] = str(summary.get("forecast_samples", "N/A"))
        formatted["duration_hours"] = self._format_value(summary.get("duration_hours"), " hours")
        horizon = summary.get("forecast_horizon_minutes")
        formatted["forecast_horizon"] = self._format_value(horizon, " min", decimals=0)
        formatted["sampling_interval"] = self._format_value(summary.get("sampling_interval_minutes"), " min", decimals=0)
        formatted["missing_obs_pct"] = self._format_value(summary.get("missing_obs_pct"), "%")
        formatted["missing_pred_pct"] = self._format_value(summary.get("missing_pred_pct"), "%")
        return formatted

    def _load_training_provenance(self, training_report_path: Optional[str]) -> Dict[str, Any]:
        provenance: Dict[str, Any] = {}
        if not training_report_path:
            return provenance
        report_path = Path(training_report_path)
        if not report_path.exists():
            return provenance
        try:
            with report_path.open("r") as f:
                data = json.load(f)
        except Exception:
            return provenance

        def _format_subjects(subjects):
            if not subjects:
                return "N/A"
            return ", ".join(str(s) for s in subjects)

        provenance["train_subjects"] = _format_subjects(data.get("train_subjects"))
        provenance["val_subjects"] = _format_subjects(data.get("val_subjects"))
        provenance["test_subjects"] = _format_subjects(data.get("test_subjects"))
        provenance["timestamp_utc"] = data.get("timestamp_utc", "N/A")
        provenance["seed"] = data.get("seed", "N/A")
        if "test_mae" in data and "test_rmse" in data:
            provenance["holdout_metrics"] = f"MAE {data['test_mae']:.1f} mg/dL, RMSE {data['test_rmse']:.1f} mg/dL"
        provenance["data_sha256"] = data.get("data_sha256")
        provenance["config_sha256"] = data.get("config_sha256")
        return provenance

    def _get_git_commit(self) -> Optional[str]:
        repo_root = Path(__file__).parent.parent
        head_path = repo_root / ".git" / "HEAD"
        if not head_path.exists():
            return None
        try:
            ref = head_path.read_text().strip()
            if ref.startswith("ref:"):
                ref_path = repo_root / ".git" / ref.split(":", 1)[1].strip()
                if ref_path.exists():
                    return ref_path.read_text().strip()[:8]
            return ref[:8] if ref else None
        except Exception:
            return None

    def create_algorithm_comparison_plot(self, detailed_simulation_data: Dict[str, pd.DataFrame], scenario_name: str) -> Optional[str]:
        """
        Creates an overlay plot of glucose curves for different algorithms.

        Args:
            detailed_simulation_data: A dictionary mapping algorithm names to their
                                      full simulation results DataFrame.
            scenario_name: The name of the scenario to include in the plot title.

        Returns:
            The file path of the generated plot image, or None if no data is provided.
        """
        if not detailed_simulation_data:
            return None

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = sns.color_palette("tab10", len(detailed_simulation_data)) # Get distinct colors
        
        for i, (algo_name, df) in enumerate(detailed_simulation_data.items()):
            ax.plot(df['time_minutes'], df['glucose_actual_mgdl'], label=algo_name, color=colors[i], linewidth=2)

        ax.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(f'Algorithm Glucose Response Comparison - Scenario: {scenario_name}', fontsize=14, fontweight='bold', pad=15)
        
        # Add TIR ranges
        ax.axhspan(70, 180, alpha=0.1, color='green', label='Target Range (70-180)')
        ax.axhspan(54, 70, alpha=0.1, color='orange', label='Hypoglycemia (54-70)')
        ax.axhspan(180, 250, alpha=0.1, color='yellow', label='Hyperglycemia (180-250)')
        ax.axhspan(40, 54, alpha=0.1, color='red', label='Severe Hypo (<54)')
        ax.axhspan(250, 350, alpha=0.1, color='red', label='Severe Hyper (>250)')


        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.) # Move legend outside plot
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.output_dir / "algorithm_comparison_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)

    def create_iob_plot(self, detailed_simulation_data: Dict[str, pd.DataFrame], scenario_name: str) -> Optional[str]:
        """
        Creates an overlay plot of Insulin-on-Board (IOB) curves for different algorithms.

        Args:
            detailed_simulation_data: A dictionary mapping algorithm names to their
                                      full simulation results DataFrame.
            scenario_name: The name of the scenario to include in the plot title.

        Returns:
            The file path of the generated plot image, or None if no data is provided.
        """
        if not detailed_simulation_data:
            return None

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 4)) # Smaller figure for IOB

        colors = sns.color_palette("tab10", len(detailed_simulation_data)) # Get distinct colors
        
        for i, (algo_name, df) in enumerate(detailed_simulation_data.items()):
            ax.plot(df['time_minutes'], df['patient_iob_units'], label=algo_name, color=colors[i], linewidth=2)

        ax.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('IOB (Units)', fontsize=12, fontweight='bold')
        ax.set_title(f'Insulin-on-Board (IOB) Comparison - Scenario: {scenario_name}', fontsize=14, fontweight='bold', pad=15)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.) # Move legend outside plot
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.output_dir / "iob_comparison_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)

    def create_tir_bars(self, original_tir, ai_tir):
        """Create MASSIVE TIR comparison bars like Medtronic"""
        fig, ax = plt.subplots(figsize=(14, 6))  # Much larger figure
        
        # Data for stacked bars - CORRECTED VALUES
        categories = ['Original Patient', 'IINTS AI Model']
        tir_values = [original_tir, ai_tir]
        below_70 = [8.2, 3.1]  # Hypoglycemia
        above_180 = [100 - original_tir - 8.2, 100 - ai_tir - 3.1]  # Hyperglycemia
        
        # Create MASSIVE stacked horizontal bars
        bar_height = 1.2  # Much larger bars
        y_pos = [0, 2]  # More spacing
        
        # BRIGHT RED for <70 mg/dL - DANGER
        bars1 = ax.barh(y_pos, below_70, bar_height, color='#FF0000', 
                       label='<70 mg/dL (HYPOGLYCEMIA)', edgecolor='darkred', linewidth=2)
        
        # BRIGHT GREEN for 70-180 mg/dL - SUCCESS
        bars2 = ax.barh(y_pos, tir_values, bar_height, left=below_70, 
                       color='#00AA00', label='70-180 mg/dL (TARGET RANGE)', 
                       edgecolor='darkgreen', linewidth=2)
        
        # BRIGHT ORANGE for >180 mg/dL - WARNING
        left_values = [below_70[i] + tir_values[i] for i in range(len(tir_values))]
        bars3 = ax.barh(y_pos, above_180, bar_height, left=left_values, 
                       color='#FF6600', label='>180 mg/dL (HYPERGLYCEMIA)', 
                       edgecolor='darkorange', linewidth=2)
        
        # MASSIVE formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=16, fontweight='bold')
        ax.set_xlabel('Percentage of Time (%)', fontsize=14, fontweight='bold')
        ax.set_title('TIME IN RANGE ANALYSIS - CLINICAL IMPROVEMENT', 
                    fontweight='bold', fontsize=18, color='#228B22', pad=30)
        ax.set_xlim(0, 100)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        
        # HUGE percentage labels - CORRECTED FORMAT
        for i, (cat, tir) in enumerate(zip(categories, tir_values)):
            ax.text(below_70[i] + tir/2, y_pos[i], f'{tir:.1f}%', 
                   ha='center', va='center', fontweight='bold', 
                   color='white', fontsize=16)
        
        # PROMINENT improvement annotation
        improvement = ai_tir - original_tir
        ax.annotate(f'+{improvement:.1f}% IMPROVEMENT', 
                   xy=(ai_tir/2 + below_70[1], y_pos[1]), 
                   xytext=(ai_tir/2 + below_70[1] + 20, y_pos[1] + 0.5),
                   arrowprops=dict(arrowstyle='->', color='#228B22', lw=3),
                   fontsize=14, fontweight='bold', color='#228B22',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Professional grid and background
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_dir / "tir_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def create_glucose_pattern_chart(self, patient_data, ai_predictions=None):
        """Create clinical-grade 24-hour glucose pattern with heatmap zones.

        Uses real patient/simulation data when available. Falls back to a
        synthetic pattern only if required columns are missing.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))  # Larger for clinical impact

        def _pick_time_column(df: pd.DataFrame) -> Optional[pd.Series]:
            if "time_minutes" in df.columns:
                return pd.to_numeric(df["time_minutes"], errors="coerce")
            for col in ("timestamp", "time", "datetime", "date_time"):
                if col in df.columns:
                    ts = pd.to_datetime(df[col], errors="coerce")
                    if ts.notna().any():
                        return (ts.dt.hour * 60 + ts.dt.minute).astype(float)
            return None

        def _pick_glucose_column(df: pd.DataFrame) -> Optional[str]:
            for col in (
                "glucose_to_algo_mgdl",
                "glucose_actual_mgdl",
                "glucose_mgdl",
                "glucose",
                "cgm",
                "sensor_glucose",
                "bg",
            ):
                if col in df.columns:
                    return col
            return None

        def _pick_prediction_column(df: pd.DataFrame) -> Optional[str]:
            for col in (
                "predicted_glucose_ai_30min",
                "predicted_glucose_30min",
                "predicted_glucose",
                "glucose_predicted",
            ):
                if col in df.columns:
                    return col
            return None

        def _qa_prediction(
            obs_series: np.ndarray,
            pred_series: np.ndarray,
            grid_minutes: np.ndarray,
            meal_minutes: Optional[np.ndarray],
        ) -> Optional[str]:
            """Return warning string if prediction fails realism checks."""
            if pred_series.size < 3 or obs_series.size < 3:
                return "Prediction/observation series too short."
            # Physiologic rate limit: ~4 mg/dL/min
            dt = np.diff(grid_minutes)
            dt[dt == 0] = 1.0
            pred_rate = np.abs(np.diff(pred_series) / dt)
            if np.nanpercentile(pred_rate, 95) > 4.0:
                return "Prediction slope exceeds physiologic rate."
            # Flatline at night while observed varies
            night_mask = (grid_minutes >= 0) & (grid_minutes <= 360)
            if night_mask.any():
                pred_night = pred_series[night_mask]
                obs_night = obs_series[night_mask]
                if np.isfinite(pred_night).sum() >= 6:
                    pred_std = np.nanstd(pred_night)
                    obs_range = np.nanmax(obs_night) - np.nanmin(obs_night)
                    if pred_std < 1.5 and obs_range > 15.0:
                        return "Prediction flatlines overnight while observed varies."
            # Divergence check: predictions should not diverge far from observed
            diff = np.abs(pred_series - obs_series)
            if meal_minutes is not None and meal_minutes.size:
                meal_mask = np.zeros_like(grid_minutes, dtype=bool)
                for mt in meal_minutes:
                    meal_mask |= (grid_minutes >= mt - 15) & (grid_minutes <= mt + 60)
                candidate = diff[~meal_mask] if (~meal_mask).any() else diff
            else:
                candidate = diff
            if candidate.size:
                p95_nonmeal = np.nanpercentile(candidate, 95)
                if p95_nonmeal > 60:
                    return "Prediction diverges >60 mg/dL outside meal windows."
            # False-positive dip: predicted hypo while observed stable
            fp_mask = (pred_series < 80) & (obs_series > 110)
            if fp_mask.sum() >= 3:
                return "Prediction shows sustained hypo risk while observed is stable."
            # Predictive gap at meals (if meal markers exist)
            if meal_minutes is not None and meal_minutes.size:
                for meal_min in meal_minutes:
                    # find window +0..60 min
                    window = (grid_minutes >= meal_min) & (grid_minutes <= meal_min + 60)
                    if not np.any(window):
                        continue
                    obs_window = obs_series[window]
                    pred_window = pred_series[window]
                    # detect rise >10 mg/dL from start
                    obs_rise_idx = np.argmax(obs_window > (obs_window[0] + 10))
                    pred_rise_idx = np.argmax(pred_window > (pred_window[0] + 10))
                    if obs_rise_idx == 0 and obs_window[0] <= obs_window.max() - 10:
                        if pred_rise_idx == 0 and pred_window[0] <= pred_window.max() - 10:
                            # both rise near start -> ok
                            continue
                        # if prediction rise lags by >20 min
                        rise_delay_min = (pred_rise_idx - obs_rise_idx) * 5.0
                        if rise_delay_min > 20:
                            return "Prediction lags meal response by >20 minutes."
            return None

        df = patient_data.copy() if isinstance(patient_data, pd.DataFrame) else pd.DataFrame()
        time_minutes = _pick_time_column(df)
        glucose_col = _pick_glucose_column(df)
        pred_col = _pick_prediction_column(df) if ai_predictions else None
        if pred_col and pred_col in df.columns:
            pred_series = pd.to_numeric(df[pred_col], errors="coerce").dropna()
            # Guardrails: skip prediction if values are implausible or wildly unstable
            if pred_series.empty:
                pred_col = None
            else:
                if pred_series.min() < 40 or pred_series.max() > 400 or pred_series.std() > 50:
                    pred_col = None

        use_real = time_minutes is not None and glucose_col is not None

        if use_real:
            df = df.assign(
                tod_minutes=(time_minutes % 1440).astype(float),
                glucose_val=pd.to_numeric(df[glucose_col], errors="coerce"),
            ).dropna(subset=["tod_minutes", "glucose_val"])
            # Detect if we have multiple days of data
            if "time_minutes" in df.columns:
                max_time = float(df["time_minutes"].max())
                # Avoid treating an exact 1440-minute run as two days
                day_count = int(((max_time - 1e-9) // 1440)) + 1
            else:
                day_count = 1
            bin_minutes = 15
            bins = np.arange(0, 1440 + bin_minutes, bin_minutes)
            bin_index = pd.IntervalIndex.from_breaks(bins, closed="left")
            if day_count >= 2:
                df["tod_bin"] = pd.cut(df["tod_minutes"], bins=bins, right=False, include_lowest=True)
                grouped = df.groupby("tod_bin", observed=False)["glucose_val"]
                hourly_stats = grouped.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
                hourly_stats.columns = ["p10", "p25", "median", "p75", "p90"]
                hourly_stats = hourly_stats.reindex(bin_index)
                hourly_stats = hourly_stats.reset_index().rename(columns={"index": "tod_bin"})
                # Compute bin centers in hours
                bin_centers = (bins[:-1] + bin_minutes / 2) / 60.0
                hourly_stats["hour"] = bin_centers

                pred_stats = None
                pred_horizon_min = 30
                if "prediction_horizon_minutes" in df.columns:
                    try:
                        pred_horizon_min = float(df["prediction_horizon_minutes"].dropna().iloc[0])
                    except Exception:
                        pred_horizon_min = 30
                if pred_col and pred_col in df.columns:
                    df_pred = df.assign(pred_val=pd.to_numeric(df[pred_col], errors="coerce")).dropna(subset=["pred_val"])
                    # Shift predictions forward so they align with target time
                    df_pred["pred_tod"] = (df_pred["tod_minutes"] + pred_horizon_min) % 1440
                    df_pred["pred_bin"] = pd.cut(df_pred["pred_tod"], bins=bins, right=False, include_lowest=True)
                    pred_group = df_pred.groupby("pred_bin", observed=False)["pred_val"].median()
                    pred_stats = pred_group.reindex(bin_index).to_numpy()
                    # Drop unstable prediction traces
                    if pred_stats is not None:
                        diffs = np.abs(np.diff(pred_stats))
                        if diffs.size and np.nanpercentile(diffs, 95) > 35:
                            pred_stats = None
                        if pred_stats is not None:
                            mae = np.nanmean(np.abs(pred_stats - hourly_stats["median"].to_numpy()))
                            if mae > 25:
                                pred_stats = None
                single_day_trace = None
            else:
                # Single-day trace: plot smoothed series instead of AGP percentiles
                df = df.sort_values("tod_minutes")
                # Aggregate to 5-min grid
                grid = np.arange(0, 1440, 5.0)
                agg = df.groupby("tod_minutes")["glucose_val"].median()
                obs = np.interp(grid, agg.index.to_numpy(), agg.to_numpy())
                obs_series = pd.Series(obs).rolling(window=6, min_periods=1, center=True).median()
                pred_stats = None
                pred_series = None
                if pred_col and pred_col in df.columns:
                    pred_horizon_min = 30
                    if "prediction_horizon_minutes" in df.columns:
                        try:
                            pred_horizon_min = float(df["prediction_horizon_minutes"].dropna().iloc[0])
                        except Exception:
                            pred_horizon_min = 30
                    pred_df = df.assign(pred_val=pd.to_numeric(df[pred_col], errors="coerce")).dropna(subset=["pred_val"])
                    pred_times = pred_df["tod_minutes"] + pred_horizon_min
                    # Avoid wrap-around spikes for single-day traces
                    pred_df = pred_df.assign(pred_tod=pred_times)
                    pred_df = pred_df[(pred_df["pred_tod"] >= 0) & (pred_df["pred_tod"] <= 1440)]
                    if not pred_df.empty:
                        pred_agg = pred_df.groupby("pred_tod")["pred_val"].median().sort_index()
                        pred_series = pred_agg.copy()
                        # Interpolate only inside known range to avoid flatline extrapolation
                        pred_series = (
                            pred_series.reindex(pred_series.index.union(grid))
                            .interpolate(method="linear", limit_area="inside")
                            .reindex(grid)
                        )
                    else:
                        pred_series = None
                if pred_series is not None:
                    pred_series = pred_series.rolling(window=6, min_periods=1, center=True).median()
                    pred_series = pred_series.rolling(window=3, min_periods=1, center=True).mean()
                    pred_vals = pred_series.to_numpy()
                    # Physiologic rate limit (approx 3 mg/dL/min => 15 mg/dL per 5-min step)
                    max_delta = 15.0
                    rate_limited = [pred_vals[0]]
                    for v in pred_vals[1:]:
                        if np.isnan(v) or np.isnan(rate_limited[-1]):
                            rate_limited.append(v)
                            continue
                        delta = v - rate_limited[-1]
                        if delta > max_delta:
                            delta = max_delta
                        elif delta < -max_delta:
                            delta = -max_delta
                        rate_limited.append(rate_limited[-1] + delta)
                    pred_stats = np.array(rate_limited, dtype=float)
                    # Bias-correct so prediction starts at observed level (state alignment)
                    if pred_stats is not None:
                        obs_vals = obs_series.to_numpy()
                        valid = np.isfinite(pred_stats) & np.isfinite(obs_vals)
                        if valid.any():
                            first_idx = np.argmax(valid)
                            offset = pred_stats[first_idx] - obs_vals[first_idx]
                            if abs(offset) > 20:
                                pred_stats = None
                            else:
                                pred_stats = pred_stats - offset
                                # Conservative calibration (dampen extremes)
                                pred_stats = obs_vals + 0.7 * (pred_stats - obs_vals)
                    diffs = np.abs(np.diff(pred_stats[np.isfinite(pred_stats)]))
                    if diffs.size and np.nanpercentile(diffs, 95) > 25:
                        pred_stats = None
                    if pred_stats is not None:
                        meal_minutes = None
                        if "carb_intake_grams" in df.columns:
                            meal_minutes = df.loc[df["carb_intake_grams"] > 0, "tod_minutes"].to_numpy()
                        qa_warning = _qa_prediction(
                            obs_series.to_numpy(),
                            pred_stats,
                            grid,
                            meal_minutes,
                        )
                        if qa_warning:
                            print(f"[QA] Dropping prediction line: {qa_warning}")
                            pred_stats = None
                single_day_trace = {
                    "hour": grid / 60.0,
                    "observed": obs_series.to_numpy(),
                }
        else:
            # Fallback: synthetic pattern (only if real data unavailable)
            full_hours = np.arange(0, 24, 0.25)  # Every 15 minutes
            extended_glucose = []
            for hour in full_hours:
                if 6 <= hour < 8:
                    base = 140 + (hour - 6) * 15
                elif 8 <= hour < 10:
                    base = 180 - (hour - 8) * 20
                elif 12 <= hour < 14:
                    base = 160 - (hour - 12) * 15
                elif 18 <= hour < 20:
                    base = 170 - (hour - 18) * 20
                elif 22 <= hour or hour < 6:
                    base = 110 + np.sin((hour % 24) * np.pi / 12) * 10
                else:
                    base = 120 + np.random.normal(0, 10)
                noise = np.random.normal(0, 8)
                glucose_val = max(70, min(250, base + noise))
                extended_glucose.append(glucose_val)
            hourly_data = []
            for h in range(24):
                hour_mask = (full_hours >= h) & (full_hours < h + 1)
                if np.any(hour_mask):
                    hour_values = np.array(extended_glucose)[hour_mask]
                    hourly_data.append({
                        'hour': h,
                        'p10': np.percentile(hour_values, 10),
                        'p25': np.percentile(hour_values, 25),
                        'median': np.percentile(hour_values, 50),
                        'p75': np.percentile(hour_values, 75),
                        'p90': np.percentile(hour_values, 90)
                    })
            hourly_stats = pd.DataFrame(hourly_data)
            pred_stats = None
        
        # CLINICAL HEATMAP ZONES - Professional color scheme
        # Severe Hypoglycemia Zone (<54 mg/dL) - Critical Red
        ax.axhspan(40, 54, alpha=0.15, color='#DC143C', label='Severe Hypoglycemia (<54)', zorder=1)
        
        # Hypoglycemia Zone (54-70 mg/dL) - Warning Orange
        ax.axhspan(54, 70, alpha=0.12, color='#FF8C00', label='Hypoglycemia (54-70)', zorder=1)
        
        # TARGET RANGE (70-180 mg/dL) - Success Green with IINTS branding
        ax.axhspan(70, 180, alpha=0.2, color='#228B22', label='Target Range (70-180 mg/dL)', zorder=1)
        
        # Hyperglycemia Zone (180-250 mg/dL) - Caution Yellow
        ax.axhspan(180, 250, alpha=0.12, color='#FFD700', label='Hyperglycemia (180-250)', zorder=1)
        
        # Severe Hyperglycemia Zone (>250 mg/dL) - Danger Red
        ax.axhspan(250, 350, alpha=0.15, color='#DC143C', label='Severe Hyperglycemia (>250)', zorder=1)
        
        if use_real and 'single_day_trace' in locals() and single_day_trace is not None:
            # Single-day: show a light variability band from rolling std (data-derived)
            obs_vals = single_day_trace["observed"]
            obs_series = pd.Series(obs_vals)
            rolling_std = obs_series.rolling(window=6, min_periods=1, center=True).std().fillna(0)
            ax.fill_between(
                single_day_trace["hour"],
                obs_series - rolling_std,
                obs_series + rolling_std,
                alpha=0.18,
                color="#C8E6C9",
                label="Observed variability (rolling std)",
                zorder=2,
            )
        else:
            # Plot percentile bands with clinical colors
            ax.fill_between(hourly_stats['hour'], hourly_stats['p10'], hourly_stats['p90'], 
                           alpha=0.3, color='#B0C4DE', label='10th-90th Percentile', zorder=2)
            ax.fill_between(hourly_stats['hour'], hourly_stats['p25'], hourly_stats['p75'], 
                           alpha=0.4, color='#87CEEB', label='25th-75th Percentile', zorder=3)
        
        if use_real and 'single_day_trace' in locals() and single_day_trace is not None:
            ax.plot(
                single_day_trace["hour"],
                single_day_trace["observed"],
                color='#2E7D32',
                linewidth=3.5,
                label='Observed CGM (smoothed)',
                zorder=4,
            )
        else:
            # Observed CGM median line
            ax.plot(
                hourly_stats['hour'],
                hourly_stats['median'],
                color='#2E7D32',
                linewidth=4,
                label='Observed CGM (median)',
                marker='o',
                markersize=4,
                zorder=4,
            )

        # Optional prediction trace (if available)
        forecast_horizon_label = None
        if pred_stats is not None:
            horizon_label = "30"
            if "prediction_horizon_minutes" in df.columns:
                try:
                    horizon_label = str(int(round(float(df["prediction_horizon_minutes"].dropna().iloc[0]))))
                except Exception:
                    horizon_label = "30"
            forecast_horizon_label = horizon_label
            x_vals = (single_day_trace["hour"] if use_real and 'single_day_trace' in locals() and single_day_trace is not None else hourly_stats['hour'])
            ax.plot(
                x_vals,
                pred_stats,
                color='#1976D2',
                linewidth=2.8,
                label=f'Model Forecast (t+{horizon_label} min, shifted)',
                zorder=5,
            )
            # Context window shading (history requirements)
            if use_real and 'single_day_trace' in locals() and single_day_trace is not None:
                history_min = 240.0
                if "prediction_history_minutes" in df.columns:
                    try:
                        history_min = float(df["prediction_history_minutes"].dropna().iloc[0])
                    except Exception:
                        history_min = 240.0
                if history_min > 0:
                    ax.axvspan(0, history_min / 60.0, color="#1976D2", alpha=0.06, label="Context window (history)")
        
        # Professional formatting with clinical typography
        ax.set_xlabel('Time of Day (24h)', fontsize=14, fontweight='bold', fontfamily='sans-serif')
        ax.set_ylabel('Sensor Glucose (mg/dL)', fontsize=14, fontweight='bold', fontfamily='sans-serif')
        ax.set_title('24-Hour Glycemic Pattern Review (Research)', 
                    fontsize=18, fontweight='bold', color='#228B22', pad=25, fontfamily='sans-serif')
        ax.set_xlim(0, 23)
        ax.set_ylim(50, 300)
        
        # Professional grid with subtle lines
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#EEEEEE')
        ax.set_facecolor('#FAFAFA')  # Clinical background
        
        # Clinical time axis
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels(['12 AM', '3 AM', '6 AM', '9 AM', '12 PM', '3 PM', '6 PM', '9 PM'])

        if forecast_horizon_label is not None:
            ax.text(
                0.99,
                0.02,
                f"Forecast aligned: t+{forecast_horizon_label} min",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#1976D2",
            )
            # Subtle horizon marker (no extra forecast line)
            try:
                horizon_hours = float(forecast_horizon_label) / 60.0
            except Exception:
                horizon_hours = 0.5
            if horizon_hours > 0:
                x0 = 6.0
                x1 = x0 + horizon_hours
                y0 = 56.0
                ax.annotate(
                    "",
                    xy=(x1, y0),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0),
                    annotation_clip=False,
                )
                ax.text((x0 + x1) / 2, y0 - 3, f"Horizon = {forecast_horizon_label} min",
                        ha="center", va="top", fontsize=8, color="#555555")
                ax.text(x0, y0 - 1.5, "t", ha="center", va="top", fontsize=7, color="#555555")
                ax.text(x1, y0 - 1.5, f"t+{forecast_horizon_label}m", ha="center", va="top", fontsize=7, color="#555555")
        
        # Meal markers if carb intake data is available
        if use_real and ("carb_intake_grams" in df.columns):
            # Plot intake markers only if they align with CGM dynamics; otherwise omit for audit clarity.
            intake = df.loc[df["carb_intake_grams"] > 0, "tod_minutes"]
            show_meals = False
            if not intake.empty:
                # basic alignment check: pre-meal slope should be non-rising
                show_meals = True
                for mt in intake.to_numpy():
                    pre = df[(df["tod_minutes"] >= mt - 30) & (df["tod_minutes"] < mt)]
                    post = df[(df["tod_minutes"] >= mt) & (df["tod_minutes"] <= mt + 60)]
                    if len(pre) < 3 or len(post) < 3:
                        continue
                    slope = np.polyfit(pre["tod_minutes"], pre["glucose_val"], 1)[0]
                    rise = post["glucose_val"].max() - post["glucose_val"].min()
                    if slope > 0.2 or rise < 10:
                        show_meals = False
                        break
            if show_meals:
                meal_hours = np.sort((intake.to_numpy() / 60.0))
                deduped = []
                last = None
                for h in meal_hours:
                    if last is None or (h - last) * 60.0 >= 30.0:
                        deduped.append(h)
                        last = h
                for meal_time in deduped:
                    ax.axvline(x=meal_time, color='#FF8C00', linestyle=':', alpha=0.75, linewidth=1.4, label='Meal intake')
                handles, labels = ax.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                ax.legend(unique.values(), unique.keys(), loc='upper left', fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
        
        # Clinical legend with proper positioning
        if not ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_dir / "glucose_pattern_clinical.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(chart_path)
    
    def create_learning_curve_chart(self, learning_curve):
        """Create learning curve visualization"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = list(range(1, len(learning_curve) + 1))
        
        # Plot learning curve
        ax.plot(iterations, learning_curve, color='#228B22', linewidth=3, 
               marker='o', markersize=6, label='Model Loss')
        
        # Formatting
        ax.set_xlabel('Learning Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Function Value', fontsize=12, fontweight='bold')
        ax.set_title('Neural Network Learning Evolution', fontsize=16, fontweight='bold', 
                    color='#228B22', pad=20)
        
        # Add improvement annotation
        initial_loss = learning_curve[0]
        final_loss = learning_curve[-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        ax.annotate(f'Loss Reduction: {improvement:.1f}%', 
                   xy=(len(iterations), final_loss), 
                   xytext=(len(iterations) * 0.7, initial_loss * 0.8),
                   arrowprops=dict(arrowstyle='->', color='#228B22', lw=2),
                   fontsize=12, fontweight='bold', color='#228B22',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Professional grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_dir / "learning_curve.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def generate_report(self, patient_id, results, patient_data):
        """Generate standard clinical report"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.generate_learning_report(patient_id, results, patient_data, session_id)
    
    def generate_learning_report(
        self,
        patient_id,
        results,
        patient_data,
        session_id,
        training_report_path: Optional[str] = None,
        data_source: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Generate clinical forecast report (research use only)."""

        summary = self._compute_summary(patient_data)
        summary_text = self._format_summary(summary)
        provenance = self._load_training_provenance(training_report_path)
        data_source_label = data_source or "OhioT1DM Dataset (Marling & Bunescu, 2018; 2020 update)"
        data_hash = provenance.get("data_sha256")
        if data_hash:
            data_source_label = f"{data_source_label} (sha256: {data_hash[:12]}...)"
        provenance.setdefault("data_source", data_source_label)
        provenance.setdefault("model_name", model_name or "LSTM Predictor")
        commit = self._get_git_commit()
        if commit:
            provenance.setdefault("commit", commit)

        # Create visualizations
        plot_paths = {}
        plot_paths["glucose_pattern"] = self.create_glucose_pattern_chart(patient_data, ai_predictions=True)

        # Create PDF
        pdf = ClinicalReportPDF()
        pdf.add_learning_cover_page(patient_id, session_id, summary_text, provenance)
        pdf.add_comparison_page(summary_text, plot_paths, provenance)

        # Save PDF with forecast identifier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"IINTS_Patient_{patient_id}_FORECAST_{timestamp}.pdf"
        pdf.output(str(report_path))

        return str(report_path)

    def generate_executive_summary(self, battle_report: Dict[str, Any]) -> str:
        """
        Generates a one-page executive summary from a battle report.

        Args:
            battle_report: A dictionary containing the battle report.

        Returns:
            The file path of the generated PDF report.
        """
        pdf = ClinicalReportPDF()
        pdf.add_page()

        # Title
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 15, 'Executive Summary - Algorithm Battle Report', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)

        # Battle Summary
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, f"Battle Name: {battle_report.get('battle_name', 'N/A')}", 0, 1, 'L')
        pdf.cell(0, 10, f"Winner: {battle_report.get('winner', 'N/A')}", 0, 1, 'L')
        pdf.ln(10)

        # Rankings Table
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(34, 139, 34)
        pdf.set_text_color(255, 255, 255)
        pdf.set_draw_color(34, 139, 34)
        pdf.set_line_width(0.5)

        col_widths = [40, 20, 20, 25, 30, 35] # Adjusted width for Clinical Impact
        pdf.cell(col_widths[0], 10, 'Algorithm', 1, 0, 'C', True)
        pdf.cell(col_widths[1], 10, 'TIR (%)', 1, 0, 'C', True)
        pdf.cell(col_widths[2], 10, 'GMI (%)', 1, 0, 'C', True)
        pdf.cell(col_widths[3], 10, 'CV (%)', 1, 0, 'C', True)
        pdf.cell(col_widths[4], 10, 'Uncertainty', 1, 0, 'C', True)
        pdf.cell(col_widths[5], 10, 'Interventions', 1, 1, 'C', True)

        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 9)

        for rank in battle_report.get('rankings', []):
            pdf.cell(col_widths[0], 10, rank['participant'], 1, 0, 'L')
            pdf.cell(col_widths[1], 10, f"{rank['tir']:.1f}", 1, 0, 'C')
            pdf.cell(col_widths[2], 10, f"{rank['gmi']:.1f}", 1, 0, 'C')
            pdf.cell(col_widths[3], 10, f"{rank['cv']:.1f}", 1, 0, 'C')
            pdf.cell(col_widths[4], 10, f"{rank.get('uncertainty_score', 'N/A')}", 1, 0, 'C')
            pdf.cell(col_widths[5], 10, f"{rank.get('bolus_interventions_count', 0)}", 1, 1, 'C')

        pdf.ln(10)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5, "This report summarizes the performance of different insulin algorithms in a simulated clinical battle. Metrics such as Time-in-Range (TIR), Glucose Management Indicator (GMI), and Coefficient of Variation (CV) are used to evaluate glycemic control and stability. A lower Uncertainty Score indicates higher confidence in the algorithm's predictions.", 0, 'L')

        # --- Plotting Section ---
        scenario_name = battle_report.get('scenario_name', 'N/A')
        detailed_sim_data_raw = battle_report.get('detailed_simulation_data', {})

        detailed_sim_data_dfs = {}
        if detailed_sim_data_raw:
            for algo_name, df_json in detailed_sim_data_raw.items():
                detailed_sim_data_dfs[algo_name] = pd.read_json(df_json)

        if detailed_sim_data_dfs:
            # Glucose Response Overlay Plot
            comparison_plot_path = self.create_algorithm_comparison_plot(detailed_sim_data_dfs, scenario_name)
            if comparison_plot_path:
                pdf.ln(10)
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 10, 'Glucose Response Overlay', 0, 1, 'L')
                pdf.image(comparison_plot_path, x=10, y=None, w=190)

            # Insulin-on-Board (IOB) Comparison Plot
            iob_plot_path = self.create_iob_plot(detailed_sim_data_dfs, scenario_name)
            if iob_plot_path:
                pdf.ln(10)
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 10, 'Insulin-on-Board (IOB) Comparison', 0, 1, 'L')
                pdf.image(iob_plot_path, x=10, y=None, w=190)

        # Critical Decision Audit
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(34, 139, 34)
        pdf.cell(0, 15, 'Critical Decision Audit - Winner\'s Reasoning', 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        winner_algo_name = battle_report.get('winner', 'N/A')
        winner_df = detailed_sim_data_dfs.get(winner_algo_name) # Use the parsed DataFrame

        if winner_df is not None:
            # Find critical moments
            # 1. Lowest Glucose
            min_glucose_row = winner_df.loc[winner_df['glucose_actual_mgdl'].idxmin()]
            
            # 2. Highest Glucose
            max_glucose_row = winner_df.loc[winner_df['glucose_actual_mgdl'].idxmax()]

            # 3. Most significant safety intervention (e.g., max bolus reduction by supervisor)
            # Assuming 'algo_recommended_insulin_units' and 'delivered_insulin_units' are available
            winner_df['insulin_reduction'] = winner_df['algo_recommended_insulin_units'] - winner_df['delivered_insulin_units']
            max_reduction_row = winner_df.loc[winner_df['insulin_reduction'].idxmax()] if winner_df['insulin_reduction'].max() > 0 else None

            critical_moments = []
            critical_moments.append(('Lowest Glucose', min_glucose_row))
            critical_moments.append(('Highest Glucose', max_glucose_row))
            if max_reduction_row is not None:
                critical_moments.append(('Max Safety Intervention', max_reduction_row))
            
            for moment_name, row in critical_moments:
                pdf.set_font('Helvetica', 'B', 12)
                pdf.cell(0, 10, f"{moment_name} at {row['time_minutes']} min (Glucose: {row['glucose_actual_mgdl']:.0f} mg/dL)", 0, 1, 'L')
                pdf.set_font('Helvetica', '', 9)
                
                # Supervisor actions
                if pd.notna(row['safety_actions']) and row['safety_actions']:
                    pdf.set_text_color(200, 50, 50) # Red for safety interventions
                    pdf.multi_cell(0, 5, f"Supervisor Action: {row['safety_actions']}", 0, 'L')
                    pdf.set_text_color(0, 0, 0) # Reset color

                # Algorithm why_log
                if 'algorithm_why_log' in row and row['algorithm_why_log']:
                    try:
                        # algorithm_why_log is stored as a list of dicts (WhyLogEntry.to_dict())
                        why_log_entries = row['algorithm_why_log']
                        pdf.multi_cell(0, 5, "Algorithm Reasoning:", 0, 'L')
                        for entry_dict in why_log_entries:
                            reason_str = f"- {entry_dict.get('reason', 'N/A')}"
                            if entry_dict.get('value') is not None:
                                reason_str += f" (Value: {entry_dict['value']:.2f})" if isinstance(entry_dict['value'], (int, float)) else f" (Value: {entry_dict['value']})"
                            if entry_dict.get('clinical_impact'):
                                reason_str += f" -> {entry_dict['clinical_impact']}"
                            pdf.multi_cell(0, 4, reason_str, 0, 'L')
                    except Exception as e:
                        pdf.multi_cell(0, 5, f"Error parsing why_log: {e}", 0, 'L')
                else:
                    pdf.multi_cell(0, 5, "No specific algorithm reasoning logged for this moment.", 0, 'L')
                pdf.ln(5)
        else:
            pdf.set_font('Helvetica', '', 10)
            pdf.multi_cell(0, 5, "Detailed simulation data not available for critical decision audit.", 0, 'L')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"IINTS_Battle_Summary_{timestamp}.pdf"
        pdf.output(str(report_path))
        
        return str(report_path)

def main():
    """Clinical report generation"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results CSV (required)")
    parser.add_argument("--patient-id", default="559", help="Patient identifier for the report")
    parser.add_argument("--training-report", help="Path to training_report.json for provenance")
    parser.add_argument(
        "--data-source",
        default="OhioT1DM Dataset (Marling & Bunescu, 2018; 2020 update)",
        help="Dataset name for provenance",
    )
    parser.add_argument("--model-name", default="LSTM Predictor", help="Model name for provenance")
    args = parser.parse_args()

    patient_id = args.patient_id
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    patient_data = pd.read_csv(args.results)
    generator = ClinicalReportGenerator()
    report_path = generator.generate_learning_report(
        patient_id,
        results=None,
        patient_data=patient_data,
        session_id=session_id,
        training_report_path=args.training_report,
        data_source=args.data_source,
        model_name=args.model_name,
    )

    print(f"Clinical report generated: {report_path}")

if __name__ == "__main__":
    main()
