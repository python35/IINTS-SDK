from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.data.importer import (
    export_standard_csv,
    import_carelink_csv,
    import_carelink_timeline,
    load_carelink_event_log,
    scenario_from_dataframe,
    summarize_carelink_csv,
)
from iints.utils.plotting import (
    IINTS_BLUE,
    IINTS_GOLD,
    IINTS_NAVY,
    IINTS_ORANGE,
    IINTS_RED,
    IINTS_TEAL,
    apply_plot_style,
)
from iints.utils.run_io import compute_sha256


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round(value: Any, digits: int = 2) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if digits == 0:
        return int(round(number))
    return round(number, digits)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _timestamp_iso(value: Any) -> str:
    return pd.Timestamp(value).to_pydatetime().strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_daily_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    working = timeline.copy()
    working["date"] = working["timestamp_dt"].dt.date
    grouped = working.groupby("date", as_index=False)
    daily = grouped.agg(
        readings=("glucose", "count"),
        mean_glucose=("glucose", "mean"),
        min_glucose=("glucose", "min"),
        max_glucose=("glucose", "max"),
        carbs_grams=("carbs", "sum"),
        insulin_units=("insulin", "sum"),
    )
    tir = working.groupby("date")["glucose"].apply(
        lambda series: ((series >= 70) & (series <= 180)).mean() * 100.0
    )
    low = working.groupby("date")["glucose"].apply(lambda series: (series < 70).mean() * 100.0)
    high = working.groupby("date")["glucose"].apply(lambda series: (series > 180).mean() * 100.0)
    daily["tir_70_180"] = daily["date"].map(tir)
    daily["time_below_70_pct"] = daily["date"].map(low)
    daily["time_above_180_pct"] = daily["date"].map(high)
    daily["date"] = daily["date"].astype(str)
    return daily


def _build_agp_profile(timeline: pd.DataFrame) -> pd.DataFrame:
    working = timeline.copy()
    working["minute_of_day"] = (
        working["timestamp_dt"].dt.hour * 60 + working["timestamp_dt"].dt.minute
    )
    grouped = working.groupby("minute_of_day")["glucose"]
    profile = pd.DataFrame(
        {
            "minute_of_day": list(grouped.groups.keys()),
            "p10": grouped.quantile(0.10).values,
            "p25": grouped.quantile(0.25).values,
            "median": grouped.quantile(0.50).values,
            "p75": grouped.quantile(0.75).values,
            "p90": grouped.quantile(0.90).values,
        }
    ).sort_values("minute_of_day")
    profile["hour_of_day"] = profile["minute_of_day"] / 60.0
    return profile


def _top_counts(series: pd.Series, *, limit: int = 8) -> dict[str, int]:
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return {}
    counts = cleaned.value_counts().head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def _record_at(timeline: pd.DataFrame, position: int) -> dict[str, Any] | None:
    if position < 0 or position >= len(timeline):
        return None
    row = timeline.iloc[position]
    return {
        "timestamp_iso": _timestamp_iso(row["timestamp_dt"]),
        "timestamp_minutes": _round(row["timestamp"], 2),
        "glucose_mgdl": _round(row["glucose"], 1),
        "carbs_grams": _round(row.get("carbs", 0.0), 1),
        "insulin_units": _round(row.get("insulin", 0.0), 3),
        "bolus_units": _round(row.get("bolus_units", 0.0), 3),
        "basal_units": _round(row.get("basal_units", 0.0), 3),
    }


def _prepare_ai_payloads(
    *,
    workbench_dir: Path,
    timeline: pd.DataFrame,
    summary: dict[str, Any],
    metrics_summary: dict[str, Any],
    daily_summary: pd.DataFrame,
    profile: pd.DataFrame,
    alert_counts: dict[str, int],
    sensor_exception_counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    lowest_position = int(timeline["glucose"].idxmin())
    latest_position = len(timeline) - 1

    common = {
        "generated_at_utc": _now_utc(),
        "source": "carelink_personal_workbench",
        "patient_name": summary.get("patient_name") or "Unknown",
        "device": summary.get("device") or "Unknown",
        "cgm": summary.get("cgm") or "Unknown",
        "period": {
            "start": summary.get("start_date"),
            "end": summary.get("end_date"),
            "days_observed": metrics_summary["days_observed"],
        },
        "summary": metrics_summary,
    }

    trace_sample = []
    if len(timeline) <= 72:
        sample_df = timeline
    else:
        step = max(1, len(timeline) // 72)
        sample_df = timeline.iloc[::step].head(72)
    for row in sample_df.itertuples(index=False):
        trace_sample.append(
            {
                "timestamp_iso": _timestamp_iso(row.timestamp_dt),
                "glucose_mgdl": _round(row.glucose, 1),
                "carbs_grams": _round(row.carbs, 1),
                "insulin_units": _round(row.insulin, 3),
            }
        )

    daily_records = daily_summary.round(2).to_dict(orient="records")
    profile_records = profile.round(2).head(96).to_dict(orient="records")
    lowest_records = timeline.nsmallest(8, "glucose")
    highest_records = timeline.nlargest(8, "glucose")

    return {
        "report_payload.json": {
            **common,
            "artifacts": {
                "workbench_dir": str(workbench_dir),
                "standard_csv": str(workbench_dir / "cgm_standard.csv"),
                "scenario": str(workbench_dir / "scenario.json"),
                "summary": str(workbench_dir / "carelink_summary.json"),
                "metrics": str(workbench_dir / "carelink_metrics.json"),
                "dashboard_png": str(workbench_dir / "carelink_dashboard.png"),
                "dashboard_html": str(workbench_dir / "carelink_dashboard.html"),
            },
            "daily_summary": daily_records,
            "profile_24h": profile_records,
            "top_alerts": alert_counts,
            "top_sensor_exceptions": sensor_exception_counts,
            "trace_sample": trace_sample,
        },
        "trends_payload.json": {
            **common,
            "daily_summary": daily_records,
            "profile_24h": profile_records,
            "trace_sample": trace_sample,
        },
        "review_payload.json": {
            **common,
            "daily_summary": daily_records,
            "profile_24h": profile_records,
            "trace_sample": trace_sample,
            "top_alerts": alert_counts,
            "top_sensor_exceptions": sensor_exception_counts,
            "review_focus": {
                "goal": "Judge whether the imported glucose history looks internally coherent and physiologically plausible.",
                "checks": [
                    "time in range versus extreme exposure",
                    "daily variability and day-to-day stability",
                    "consistency between glucose, carbs, and insulin logs",
                    "frequency of alerts and sensor exceptions",
                ],
            },
        },
        "anomalies_payload.json": {
            **common,
            "lowest_readings": [
                _record_at(lowest_records.reset_index(drop=True), idx) for idx in range(len(lowest_records))
            ],
            "highest_readings": [
                _record_at(highest_records.reset_index(drop=True), idx) for idx in range(len(highest_records))
            ],
            "top_alerts": alert_counts,
            "top_sensor_exceptions": sensor_exception_counts,
        },
        "step_riskiest.json": {
            **common,
            "selection_reason": "lowest_glucose",
            "selected_step": _record_at(timeline.reset_index(drop=True), lowest_position),
            "previous_step": _record_at(timeline.reset_index(drop=True), lowest_position - 1),
            "next_step": _record_at(timeline.reset_index(drop=True), lowest_position + 1),
        },
        "step_latest.json": {
            **common,
            "selection_reason": "latest_step",
            "selected_step": _record_at(timeline.reset_index(drop=True), latest_position),
            "previous_step": _record_at(timeline.reset_index(drop=True), latest_position - 1),
            "next_step": None,
        },
    }


def _write_local_mdmp_cert(
    *,
    workbench_dir: Path,
    ai_dir: Path,
    expires_days: int,
    grade: str,
    key_dir: Optional[Path],
) -> dict[str, str]:
    try:
        import mdmp_core  # type: ignore
    except ImportError:
        return {}

    signer_cls = getattr(mdmp_core, "MDMPSigner", None)
    keygen_fn = getattr(mdmp_core, "generate_keypair", None)
    if signer_cls is None or keygen_fn is None:
        return {}

    resolved_key_dir = key_dir.expanduser().resolve() if key_dir is not None else ai_dir / "keys"
    private_key_path = resolved_key_dir / "mdmp_private_v1.pem"
    public_key_path = resolved_key_dir / "mdmp_pub_v1.pem"
    if not private_key_path.is_file() or not public_key_path.is_file():
        keygen_fn(output_dir=resolved_key_dir)

    cert_payload = {
        "mdmp_object": "iints_carelink_local_cert",
        "spec_version": "1.0",
        "grade": grade,
        "generated_at_utc": _now_utc(),
        "workbench_dir": str(workbench_dir),
        "purpose": "local_research_ai",
        "carelink_summary_sha256": f"sha256:{compute_sha256(workbench_dir / 'carelink_summary.json')}",
        "dashboard_png_sha256": f"sha256:{compute_sha256(workbench_dir / 'carelink_dashboard.png')}",
        "notes": "Local development certificate generated by the IINTS CareLink workbench.",
    }
    signer = signer_cls(
        private_key_path=private_key_path,
        signed_by="IINTS-Local-AI",
        key_id="iints_local_ai_v1",
    )
    signed_cert = signer.sign_card(cert_payload, expires_days=expires_days)
    cert_path = ai_dir / "report.signed.mdmp"
    _write_json(cert_path, signed_cert)
    return {
        "mdmp_cert": str(cert_path),
        "mdmp_public_key": str(public_key_path),
        "mdmp_private_key": str(private_key_path),
    }


def _render_dashboard(
    *,
    timeline: pd.DataFrame,
    daily_summary: pd.DataFrame,
    profile: pd.DataFrame,
    summary: dict[str, Any],
    metrics_summary: dict[str, Any],
    output_path: Path,
) -> None:
    apply_plot_style(dpi=180, font_scale=1.05)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_trace, ax_agp, ax_daily, ax_stats = axes.flatten()
    fig.patch.set_facecolor("#f8fbfd")

    fig.suptitle(
        f"Personal CareLink Workbench — {summary.get('patient_name') or 'Unknown'}",
        fontsize=22,
        fontweight="bold",
        color=IINTS_NAVY,
        y=0.98,
    )
    fig.text(
        0.5,
        0.95,
        "Imported MiniMed / CareLink data prepared for personal review, experiments, and local AI explanations.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=IINTS_NAVY,
    )

    ax_trace.axhspan(70, 180, alpha=0.13, color=IINTS_TEAL, zorder=0)
    ax_trace.axhline(70, color=IINTS_RED, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_trace.axhline(180, color=IINTS_ORANGE, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_trace.plot(timeline["timestamp_dt"], timeline["glucose"], color=IINTS_BLUE, linewidth=1.6, zorder=2)
    meal_mask = timeline["carbs"].fillna(0.0) > 0
    if meal_mask.any():
        ax_trace.scatter(
            timeline.loc[meal_mask, "timestamp_dt"],
            timeline.loc[meal_mask, "glucose"],
            s=18,
            color=IINTS_GOLD,
            marker="D",
            alpha=0.9,
            label="Meal event",
            zorder=3,
        )
    ax_trace.set_title("Full Trace")
    ax_trace.set_ylabel("Glucose (mg/dL)")
    ax_trace.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_trace.tick_params(axis="x", rotation=25)
    ax_trace.grid(alpha=0.2)

    ax_agp.axhspan(70, 180, alpha=0.13, color=IINTS_TEAL, zorder=0)
    ax_agp.fill_between(profile["hour_of_day"], profile["p10"], profile["p90"], color=IINTS_BLUE, alpha=0.10, label="10-90%")
    ax_agp.fill_between(profile["hour_of_day"], profile["p25"], profile["p75"], color=IINTS_BLUE, alpha=0.22, label="25-75%")
    ax_agp.plot(profile["hour_of_day"], profile["median"], color=IINTS_NAVY, linewidth=2.0, label="Median")
    ax_agp.axhline(70, color=IINTS_RED, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_agp.axhline(180, color=IINTS_ORANGE, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_agp.set_title("24h Glucose Pattern")
    ax_agp.set_xlabel("Hour of day")
    ax_agp.set_ylabel("Glucose (mg/dL)")
    ax_agp.set_xlim(0, 24)
    ax_agp.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax_agp.legend(frameon=False, loc="upper left")
    ax_agp.grid(alpha=0.2)

    ax_daily.bar(daily_summary["date"], daily_summary["tir_70_180"], color=IINTS_TEAL, alpha=0.9)
    ax_daily.axhline(70, color=IINTS_NAVY, linestyle="--", linewidth=1.0, alpha=0.7)
    ax_daily.set_title("Daily Time in Range")
    ax_daily.set_ylabel("TIR 70-180 (%)")
    ax_daily.set_ylim(0, 100)
    ax_daily.tick_params(axis="x", rotation=40)
    ax_daily.grid(axis="y", alpha=0.2)

    ax_stats.axis("off")
    stat_lines = [
        f"Period: {summary.get('start_date')} -> {summary.get('end_date')}",
        f"Device: {summary.get('device') or 'Unknown'}",
        f"CGM: {summary.get('cgm') or 'Unknown'}",
        "",
        f"Mean glucose: {metrics_summary['mean_glucose_mgdl']:.1f} mg/dL",
        f"Median glucose: {metrics_summary['median_glucose_mgdl']:.1f} mg/dL",
        f"GMI: {metrics_summary['gmi_pct']:.2f}%",
        f"CV: {metrics_summary['cv_pct']:.1f}%",
        f"TIR 70-180: {metrics_summary['time_in_range_70_180_pct']:.1f}%",
        f"Time <70: {metrics_summary['time_below_70_pct']:.2f}%",
        f"Time >180: {metrics_summary['time_above_180_pct']:.1f}%",
        "",
        f"Days observed: {metrics_summary['days_observed']}",
        f"Readings: {metrics_summary['reading_count']}",
        f"Total carbs: {metrics_summary['total_carbs_grams']:.1f} g",
        f"Total insulin: {metrics_summary['total_insulin_units']:.2f} U",
        f"Alerts logged: {metrics_summary['alert_count']}",
        f"Sensor exceptions: {metrics_summary['sensor_exception_count']}",
    ]
    ax_stats.text(
        0.02,
        0.98,
        "\n".join(stat_lines),
        ha="left",
        va="top",
        fontsize=10.5,
        color=IINTS_NAVY,
        bbox={"facecolor": "white", "edgecolor": "#cfd8dc", "boxstyle": "round,pad=0.6", "alpha": 0.95},
    )

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_poster(
    *,
    timeline: pd.DataFrame,
    daily_summary: pd.DataFrame,
    profile: pd.DataFrame,
    summary: dict[str, Any],
    metrics_summary: dict[str, Any],
    output_path: Path,
) -> None:
    apply_plot_style(dpi=220, font_scale=1.15)
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#f8fbfd")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.9], width_ratios=[1.25, 1.05, 0.95])
    ax_trace = fig.add_subplot(grid[0, :2])
    ax_agp = fig.add_subplot(grid[0, 2])
    ax_daily = fig.add_subplot(grid[1, 0])
    ax_story = fig.add_subplot(grid[1, 1:])

    patient_name = summary.get("patient_name") or "Unknown"
    fig.suptitle(
        f"IINTS Personal Glucose Poster — {patient_name}",
        fontsize=24,
        fontweight="bold",
        color=IINTS_NAVY,
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        "Imported CareLink / MiniMed data turned into an experiment-ready, AI-ready personal review workspace.",
        ha="center",
        fontsize=11.5,
        color=IINTS_NAVY,
    )

    ax_trace.axhspan(70, 180, alpha=0.13, color=IINTS_TEAL, zorder=0)
    ax_trace.axhline(70, color=IINTS_RED, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_trace.axhline(180, color=IINTS_ORANGE, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_trace.plot(timeline["timestamp_dt"], timeline["glucose"], color=IINTS_BLUE, linewidth=1.5, zorder=2)
    meal_mask = timeline["carbs"].fillna(0.0) > 0
    if meal_mask.any():
        ax_trace.scatter(
            timeline.loc[meal_mask, "timestamp_dt"],
            timeline.loc[meal_mask, "glucose"],
            s=18,
            color=IINTS_GOLD,
            marker="D",
            alpha=0.9,
            zorder=3,
        )
    ax_trace.set_title("Two-Week Trace")
    ax_trace.set_ylabel("Glucose (mg/dL)")
    ax_trace.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_trace.tick_params(axis="x", rotation=25)
    ax_trace.grid(alpha=0.2)

    ax_agp.axhspan(70, 180, alpha=0.13, color=IINTS_TEAL, zorder=0)
    ax_agp.fill_between(profile["hour_of_day"], profile["p10"], profile["p90"], color=IINTS_BLUE, alpha=0.10)
    ax_agp.fill_between(profile["hour_of_day"], profile["p25"], profile["p75"], color=IINTS_BLUE, alpha=0.24)
    ax_agp.plot(profile["hour_of_day"], profile["median"], color=IINTS_NAVY, linewidth=2.0)
    ax_agp.axhline(70, color=IINTS_RED, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_agp.axhline(180, color=IINTS_ORANGE, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_agp.set_title("24h Pattern")
    ax_agp.set_xlabel("Hour")
    ax_agp.set_ylabel("mg/dL")
    ax_agp.set_xlim(0, 24)
    ax_agp.set_xticks([0, 6, 12, 18, 24])
    ax_agp.grid(alpha=0.2)

    ax_daily.bar(daily_summary["date"], daily_summary["tir_70_180"], color=IINTS_TEAL, alpha=0.92)
    ax_daily.axhline(70, color=IINTS_NAVY, linestyle="--", linewidth=1.0, alpha=0.8)
    ax_daily.set_title("Daily Time In Range")
    ax_daily.set_ylabel("TIR 70-180 (%)")
    ax_daily.set_ylim(0, 100)
    ax_daily.tick_params(axis="x", rotation=40)
    ax_daily.grid(axis="y", alpha=0.2)

    ax_story.axis("off")
    story_lines = [
        "What this workspace enables",
        "",
        f"- Mean glucose: {metrics_summary['mean_glucose_mgdl']:.1f} mg/dL",
        f"- Time in range: {metrics_summary['time_in_range_70_180_pct']:.1f}%",
        f"- GMI: {metrics_summary['gmi_pct']:.2f}% | CV: {metrics_summary['cv_pct']:.1f}%",
        f"- Hypo exposure: {metrics_summary['time_below_70_pct']:.2f}%",
        f"- Hyper exposure: {metrics_summary['time_above_180_pct']:.1f}%",
        "",
        "From here you can:",
        "- inspect the PNG/HTML dashboard",
        "- run IINTS experiments using the generated scenario.json",
        "- ask the local Mistral model to explain trends and anomalies",
        "",
        "Research use only. Not medical advice.",
    ]
    ax_story.text(
        0.02,
        0.98,
        "\n".join(story_lines),
        ha="left",
        va="top",
        fontsize=11,
        color=IINTS_NAVY,
        bbox={"facecolor": "white", "edgecolor": "#d5e3e8", "boxstyle": "round,pad=0.7", "alpha": 0.97},
    )

    fig.text(
        0.5,
        0.03,
        f"Period: {summary.get('start_date')} -> {summary.get('end_date')}   |   Device: {summary.get('device') or 'Unknown'}   |   CGM: {summary.get('cgm') or 'Unknown'}",
        ha="center",
        fontsize=10,
        color=IINTS_NAVY,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _build_html_dashboard(
    *,
    summary: dict[str, Any],
    metrics_summary: dict[str, Any],
    dashboard_png: Path,
    output_path: Path,
    workbench_dir: Path,
) -> None:
    rows = [
        ("Patient", summary.get("patient_name") or "Unknown"),
        ("Period", f"{summary.get('start_date')} -> {summary.get('end_date')}"),
        ("Device", summary.get("device") or "Unknown"),
        ("CGM", summary.get("cgm") or "Unknown"),
        ("Mean glucose", f"{metrics_summary['mean_glucose_mgdl']:.1f} mg/dL"),
        ("GMI", f"{metrics_summary['gmi_pct']:.2f}%"),
        ("CV", f"{metrics_summary['cv_pct']:.1f}%"),
        ("TIR 70-180", f"{metrics_summary['time_in_range_70_180_pct']:.1f}%"),
        ("Time <70", f"{metrics_summary['time_below_70_pct']:.2f}%"),
        ("Time >180", f"{metrics_summary['time_above_180_pct']:.1f}%"),
    ]
    command_block = "\n".join(
        [
            f"iints run --algo algorithms/example_algorithm.py --scenario-path {workbench_dir / 'scenario.json'}",
            f"iints ai report {workbench_dir} --model ministral-3:3b",
            f"iints ai explain {workbench_dir} --model ministral-3:3b",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IINTS CareLink Workbench</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #183642; background: #f8fbfd; }}
    h1, h2 {{ color: #264653; }}
    .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 24px; align-items: start; }}
    .card {{ background: white; border: 1px solid #d5e3e8; border-radius: 14px; padding: 18px; box-shadow: 0 10px 24px rgba(38,70,83,0.06); }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ padding: 8px 0; border-bottom: 1px solid #eef4f6; vertical-align: top; }}
    td:first-child {{ color: #5f7a84; width: 42%; }}
    img {{ width: 100%; border-radius: 12px; border: 1px solid #d5e3e8; background: white; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    pre {{ background: #102a33; color: #f8fbfd; padding: 14px; border-radius: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Personal CareLink Workbench</h1>
  <p>This workspace was generated from a CareLink / MiniMed export so you can inspect your data, reuse it in IINTS, and ask the local AI assistant to explain the trends.</p>
  <div class="grid">
    <div class="card">
      <h2>Summary</h2>
      <table>
        {''.join(f'<tr><td>{label}</td><td>{value}</td></tr>' for label, value in rows)}
      </table>
      <h2 style="margin-top: 22px;">Next Commands</h2>
      <pre>{command_block}</pre>
    </div>
    <div>
      <img src="{dashboard_png.name}" alt="CareLink dashboard">
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def build_carelink_workbench(
    input_csv: str | Path,
    *,
    output_dir: str | Path = "./results/carelink_workbench",
    scenario_name: str = "Imported CareLink Scenario",
    scenario_version: str = "1.0",
    carb_threshold: float = 0.1,
    create_dev_mdmp_cert: bool = True,
    grade: str = "research_grade",
    expires_days: int = 30,
    key_dir: str | Path | None = None,
) -> dict[str, str]:
    """
    Build a reusable personal-data workspace from a CareLink CSV export.

    The workspace includes:
    - imported standard CSV and scenario JSON
    - personal dashboard PNG + HTML
    - metrics / summary JSON
    - AI-ready payloads under `ai/`
    - optional local development MDMP certificate for the AI assistant
    """
    source_path = Path(input_csv).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"CareLink CSV not found: {source_path}")

    workbench_dir = Path(output_dir).expanduser().resolve()
    workbench_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_carelink_csv(source_path)
    timeline = import_carelink_timeline(source_path)
    standard_df = import_carelink_csv(source_path)
    scenario = scenario_from_dataframe(
        standard_df,
        scenario_name=scenario_name,
        scenario_version=scenario_version,
        carb_threshold=carb_threshold,
        description="Imported CareLink scenario prepared for IINTS experiments.",
    )
    raw_events, _metadata = load_carelink_event_log(source_path)

    metrics = ClinicalMetricsCalculator().calculate(timeline["glucose"], timeline["timestamp"])
    days_observed = int(timeline["timestamp_dt"].dt.date.nunique())
    total_carbs = float(timeline["carbs"].sum())
    total_insulin = float(timeline["insulin"].sum())
    metrics_summary = {
        "days_observed": days_observed,
        "reading_count": int(len(timeline)),
        "mean_glucose_mgdl": float(metrics.mean_glucose),
        "median_glucose_mgdl": float(metrics.median_glucose),
        "gmi_pct": float(metrics.gmi),
        "cv_pct": float(metrics.cv),
        "sd_mgdl": float(metrics.sd),
        "time_in_range_70_180_pct": float(metrics.tir_70_180),
        "time_below_70_pct": float(metrics.tir_below_70),
        "time_below_54_pct": float(metrics.tir_below_54),
        "time_above_180_pct": float(metrics.tir_above_180),
        "time_above_250_pct": float(metrics.tir_above_250),
        "data_coverage_pct": float(metrics.data_coverage),
        "readings_per_day": float(metrics.readings_per_day),
        "total_carbs_grams": total_carbs,
        "average_daily_carbs_grams": total_carbs / max(days_observed, 1),
        "total_insulin_units": total_insulin,
        "average_daily_insulin_units": total_insulin / max(days_observed, 1),
        "alert_count": int(summary.get("alert_rows", 0)),
        "sensor_exception_count": int(summary.get("sensor_exception_rows", 0)),
    }

    daily_summary = _build_daily_summary(timeline)
    profile = _build_agp_profile(timeline)
    alert_counts = _top_counts(raw_events.get("Alert", pd.Series(dtype="string")))
    sensor_exception_counts = _top_counts(raw_events.get("Sensor Exception", pd.Series(dtype="string")))

    standard_csv = workbench_dir / "cgm_standard.csv"
    scenario_path = workbench_dir / "scenario.json"
    summary_path = workbench_dir / "carelink_summary.json"
    metrics_path = workbench_dir / "carelink_metrics.json"
    timeline_path = workbench_dir / "carelink_timeline.csv"
    dashboard_png = workbench_dir / "carelink_dashboard.png"
    poster_png = workbench_dir / "carelink_poster.png"
    dashboard_html = workbench_dir / "carelink_dashboard.html"

    export_standard_csv(standard_df, standard_csv)
    scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    _write_json(summary_path, summary)
    _write_json(metrics_path, metrics_summary)

    timeline_export = timeline.copy()
    timeline_export.insert(1, "timestamp_iso", timeline_export["timestamp_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    timeline_export.to_csv(timeline_path, index=False)

    _render_dashboard(
        timeline=timeline,
        daily_summary=daily_summary,
        profile=profile,
        summary=summary,
        metrics_summary=metrics_summary,
        output_path=dashboard_png,
    )
    _render_poster(
        timeline=timeline,
        daily_summary=daily_summary,
        profile=profile,
        summary=summary,
        metrics_summary=metrics_summary,
        output_path=poster_png,
    )
    _build_html_dashboard(
        summary=summary,
        metrics_summary=metrics_summary,
        dashboard_png=dashboard_png,
        output_path=dashboard_html,
        workbench_dir=workbench_dir,
    )

    ai_dir = workbench_dir / "ai"
    payloads = _prepare_ai_payloads(
        workbench_dir=workbench_dir,
        timeline=timeline.reset_index(drop=True),
        summary=summary,
        metrics_summary=metrics_summary,
        daily_summary=daily_summary,
        profile=profile,
        alert_counts=alert_counts,
        sensor_exception_counts=sensor_exception_counts,
    )
    written: dict[str, str] = {
        "standard_csv": str(standard_csv),
        "scenario": str(scenario_path),
        "summary": str(summary_path),
        "metrics": str(metrics_path),
        "timeline": str(timeline_path),
        "dashboard_png": str(dashboard_png),
        "poster_png": str(poster_png),
        "dashboard_html": str(dashboard_html),
    }
    for filename, payload in payloads.items():
        target = ai_dir / filename
        _write_json(target, payload)
        written[filename.removesuffix(".json")] = str(target)

    if create_dev_mdmp_cert:
        cert_paths = _write_local_mdmp_cert(
            workbench_dir=workbench_dir,
            ai_dir=ai_dir,
            expires_days=expires_days,
            grade=grade,
            key_dir=Path(key_dir) if key_dir is not None else None,
        )
        written.update(cert_paths)

    return written
