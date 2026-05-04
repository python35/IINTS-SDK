from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .realism_reference import ReferenceComparison
from .realism_validator import RealismReport


def _status_class(status: str) -> str:
    return {
        "passed": "passed",
        "warning": "warning",
        "failed": "failed",
        "skipped": "skipped",
    }.get(status, "skipped")


def _verdict_class(verdict: str) -> str:
    mapping = {
        "likely_realistic": "passed",
        "needs_review": "warning",
        "likely_unrealistic": "failed",
    }
    return mapping.get(verdict, "skipped")


def _format_metric(value: object, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _trace_svg(dataframe: pd.DataFrame) -> str:
    df = dataframe.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    if "carbs" not in df.columns:
        df["carbs"] = 0.0
    if "insulin" not in df.columns:
        df["insulin"] = 0.0
    df["carbs"] = pd.to_numeric(df["carbs"], errors="coerce").fillna(0.0)
    df["insulin"] = pd.to_numeric(df["insulin"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["timestamp", "glucose"]).sort_values("timestamp")
    if df.empty:
        return "<p>No valid glucose trace available for preview.</p>"

    width = 920.0
    height = 260.0
    margin_left = 56.0
    margin_right = 16.0
    margin_top = 16.0
    margin_bottom = 30.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    ts_min = float(df["timestamp"].min())
    ts_max = float(df["timestamp"].max())
    if ts_max <= ts_min:
        ts_max = ts_min + 5.0

    glucose_min = float(min(df["glucose"].min(), 50.0))
    glucose_max = float(max(df["glucose"].max(), 220.0))
    glucose_pad = max(10.0, (glucose_max - glucose_min) * 0.08)
    y_low = glucose_min - glucose_pad
    y_high = glucose_max + glucose_pad

    def x_pos(timestamp: float) -> float:
        return margin_left + ((timestamp - ts_min) / (ts_max - ts_min)) * plot_width

    def y_pos(glucose: float) -> float:
        scaled = (glucose - y_low) / (y_high - y_low)
        return margin_top + plot_height - (scaled * plot_height)

    timestamp_values = df["timestamp"].astype(float).tolist()
    glucose_values = df["glucose"].astype(float).tolist()
    points = " ".join(
        f"{x_pos(timestamp):.2f},{y_pos(glucose):.2f}"
        for timestamp, glucose in zip(timestamp_values, glucose_values)
    )
    meal_circles = []
    for timestamp, glucose in zip(
        df.loc[df["carbs"] >= 10.0, "timestamp"].astype(float).tolist(),
        df.loc[df["carbs"] >= 10.0, "glucose"].astype(float).tolist(),
    ):
        meal_circles.append(
            f'<circle cx="{x_pos(timestamp):.2f}" cy="{y_pos(glucose):.2f}" r="4.5" '
            'fill="#f97316" stroke="#fff7ed" stroke-width="1.5" />'
        )
    insulin_bars = []
    for timestamp, glucose in zip(
        df.loc[df["insulin"] >= 0.3, "timestamp"].astype(float).tolist(),
        df.loc[df["insulin"] >= 0.3, "glucose"].astype(float).tolist(),
    ):
        cx = x_pos(timestamp)
        cy = y_pos(glucose)
        insulin_bars.append(
            f'<line x1="{cx:.2f}" y1="{cy + 10:.2f}" x2="{cx:.2f}" y2="{cy - 12:.2f}" '
            'stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" />'
        )

    target_top = y_pos(180.0)
    target_bottom = y_pos(70.0)
    x_axis_y = margin_top + plot_height
    xticks = []
    total_hours = max(1, int(round((ts_max - ts_min) / 60.0)))
    tick_hours = [0, 6, 12, 18, 24] if total_hours >= 18 else [0, 4, 8, 12]
    for tick_hour in tick_hours:
        tick_ts = ts_min + (tick_hour * 60.0)
        if tick_ts > ts_max:
            continue
        x = x_pos(tick_ts)
        xticks.append(
            f'<text x="{x:.2f}" y="{height - 8:.2f}" text-anchor="middle" fill="#475569" font-size="11">{tick_hour:02d}:00</text>'
        )

    yticks = []
    for glucose in (70, 120, 180, 240):
        if glucose < y_low or glucose > y_high:
            continue
        y = y_pos(float(glucose))
        yticks.append(
            f'<line x1="{margin_left:.2f}" y1="{y:.2f}" x2="{width - margin_right:.2f}" y2="{y:.2f}" '
            'stroke="#e2e8f0" stroke-width="1" />'
            f'<text x="{margin_left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" fill="#475569" font-size="11">{glucose}</text>'
        )

    return f"""
<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Glucose trace preview">
  <rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" rx="18" fill="#ffffff" />
  <rect x="{margin_left:.2f}" y="{target_top:.2f}" width="{plot_width:.2f}" height="{target_bottom - target_top:.2f}" fill="#dcfce7" opacity="0.7" />
  {''.join(yticks)}
  <line x1="{margin_left:.2f}" y1="{x_axis_y:.2f}" x2="{width - margin_right:.2f}" y2="{x_axis_y:.2f}" stroke="#cbd5e1" stroke-width="1.5" />
  <polyline fill="none" stroke="#0f766e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{points}" />
  {''.join(insulin_bars)}
  {''.join(meal_circles)}
  {''.join(xticks)}
</svg>
"""


def _comparison_rows(comparisons: list[ReferenceComparison]) -> str:
    rows = []
    for comparison in comparisons:
        rows.append(
            "<tr>"
            f"<td>{html.escape(comparison.label)}</td>"
            f"<td><span class=\"chip {_status_class(comparison.status)}\">{html.escape(comparison.status)}</span></td>"
            f"<td>{_format_metric(comparison.observed_value)}</td>"
            f"<td>{comparison.band.target_low:.1f} – {comparison.band.target_high:.1f}</td>"
            f"<td>{comparison.band.warning_low:.1f} – {comparison.band.warning_high:.1f}</td>"
            f"<td>{html.escape(comparison.detail)}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_realism_dashboard_html(
    report: RealismReport,
    dataframe: pd.DataFrame,
    *,
    title: str = "IINTS Physiological Realism Dashboard",
    source_label: Optional[str] = None,
) -> str:
    source_text = source_label or "Current trace"
    cards = [
        ("Verdict", report.verdict.replace("_", " ")),
        ("Realism score", f"{report.realism_score:.2f}"),
        ("Mean glucose", _format_metric(report.metrics.get("mean_glucose_mgdl"), " mg/dL")),
        ("CV", _format_metric(report.metrics.get("cv_pct"), "%")),
        ("TIR 70-180", _format_metric(report.metrics.get("tir_70_180_pct"), "%")),
        ("Meals", _format_metric(report.metrics.get("meal_count"))),
    ]
    card_html = "".join(
        "<div class=\"card stat\">"
        f"<div class=\"label\">{html.escape(label)}</div>"
        f"<div class=\"value\">{html.escape(value)}</div>"
        "</div>"
        for label, value in cards
    )
    checks_html = "".join(
        "<tr>"
        f"<td>{html.escape(check.title)}</td>"
        f"<td><span class=\"chip {_status_class(check.status)}\">{html.escape(check.status)}</span></td>"
        f"<td>{html.escape(check.detail)}</td>"
        "</tr>"
        for check in report.checks
    )
    meal_html = "".join(
        "<tr>"
        f"<td>{response.meal_time_minutes:.0f}</td>"
        f"<td>{response.carbs_grams:.1f}</td>"
        f"<td>{response.rise_mgdl:.1f}</td>"
        f"<td>{response.peak_lag_minutes:.1f}</td>"
        f"<td>{response.matched_insulin_units:.1f}</td>"
        "</tr>"
        for response in report.meal_responses[:12]
    )
    meal_rows_html = meal_html or '<tr><td colspan="5">No meal responses were available.</td></tr>'
    reference_block = ""
    if report.reference_profile is not None:
        reference_block = f"""
        <section class="card">
          <h2>Reference Envelope</h2>
          <p><strong>{html.escape(report.reference_profile.label)}</strong><br>{html.escape(report.reference_profile.description)}</p>
          <p class="muted">{html.escape(report.reference_profile.source)}</p>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Status</th>
                <th>Observed</th>
                <th>Target band</th>
                <th>Outer envelope</th>
                <th>Interpretation</th>
              </tr>
            </thead>
            <tbody>{_comparison_rows(report.reference_comparisons)}</tbody>
          </table>
        </section>
        """

    embedded_report = html.escape(json.dumps(report.to_dict(), indent=2))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --ink: #0f172a;
      --muted: #475569;
      --card: #ffffff;
      --line: #dbe4ee;
      --pass: #166534;
      --pass-bg: #dcfce7;
      --warn: #9a3412;
      --warn-bg: #ffedd5;
      --fail: #991b1b;
      --fail-bg: #fee2e2;
      --skip: #475569;
      --skip-bg: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #ecfeff, var(--bg) 38%);
      color: var(--ink);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 2.1rem; }}
    h2 {{ font-size: 1.2rem; }}
    p {{ line-height: 1.5; }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 18px 0 22px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
      margin-bottom: 18px;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 8px;
    }}
    .stat .value {{
      font-size: 1.7rem;
      font-weight: 700;
    }}
    .chip {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .chip.passed {{ background: var(--pass-bg); color: var(--pass); }}
    .chip.warning {{ background: var(--warn-bg); color: var(--warn); }}
    .chip.failed {{ background: var(--fail-bg); color: var(--fail); }}
    .chip.skipped {{ background: var(--skip-bg); color: var(--skip); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 10px 8px;
      border-top: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-size: 0.85rem; }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 18px;
      border-radius: 18px;
      overflow: auto;
      font-size: 0.84rem;
    }}
  </style>
</head>
<body>
  <main>
    <header class="card">
      <div class="chip {_verdict_class(report.verdict)}">{html.escape(report.verdict.replace('_', ' '))}</div>
      <h1>{html.escape(title)}</h1>
      <p><strong>Source:</strong> {html.escape(source_text)}</p>
      <p>{html.escape(report.summary)}</p>
    </header>

    <section class="grid">{card_html}</section>

    <section class="card">
      <h2>Trace Preview</h2>
      <p class="muted">Orange markers show meal annotations. Blue stems show insulin events.</p>
      {_trace_svg(dataframe)}
    </section>

    <section class="card">
      <h2>Check Breakdown</h2>
      <table>
        <thead>
          <tr><th>Check</th><th>Status</th><th>Detail</th></tr>
        </thead>
        <tbody>{checks_html}</tbody>
      </table>
    </section>

    {reference_block}

    <section class="card">
      <h2>Meal Response Snapshot</h2>
      <table>
        <thead>
          <tr><th>Minute</th><th>Carbs (g)</th><th>Rise (mg/dL)</th><th>Peak lag (min)</th><th>Matched insulin (U)</th></tr>
        </thead>
        <tbody>{meal_rows_html}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Embedded JSON</h2>
      <pre>{embedded_report}</pre>
    </section>
  </main>
</body>
</html>
"""


def write_realism_dashboard(
    report: RealismReport,
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "IINTS Physiological Realism Dashboard",
    source_label: Optional[str] = None,
) -> Path:
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        build_realism_dashboard_html(report, dataframe, title=title, source_label=source_label),
        encoding="utf-8",
    )
    return resolved
