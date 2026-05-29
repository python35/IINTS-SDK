from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _triggered_mask(df: pd.DataFrame) -> pd.Series:
    if "safety_triggered" not in df.columns:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    values = df["safety_triggered"]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _reason_counts(reasons: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for raw_reason in reasons:
        if raw_reason is None or (isinstance(raw_reason, float) and math.isnan(raw_reason)):
            continue
        for chunk in str(raw_reason).split(";"):
            label = chunk.strip()
            if not label:
                continue
            label = label.split(":")[0].strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def summarize_safety_trace(
    results_df: pd.DataFrame,
    safety_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize a run in terms a reviewer can inspect without reading code."""
    if results_df.empty:
        return {
            "total_steps": 0,
            "duration_minutes": 0.0,
            "intervention_count": 0,
            "intervention_rate_pct": 0.0,
            "top_reasons": {},
            "glucose": {},
            "insulin": {},
            "time_in_ranges_pct": {},
            "terminated_early": bool((safety_report or {}).get("terminated_early", False)),
            "termination_reason": (safety_report or {}).get("termination_reason"),
        }

    time = _numeric_series(results_df, "time_minutes")
    glucose = _numeric_series(results_df, "glucose_actual_mgdl")
    delivered = _numeric_series(results_df, "delivered_insulin_units")
    proposed = _numeric_series(results_df, "algo_recommended_insulin_units")
    triggered = _triggered_mask(results_df)
    overrides = results_df[triggered]
    reason_counts = _reason_counts(overrides["safety_reason"]) if "safety_reason" in overrides.columns else {}

    low = float((glucose < 70).mean() * 100.0)
    very_low = float((glucose < 54).mean() * 100.0)
    target = float(((glucose >= 70) & (glucose <= 180)).mean() * 100.0)
    high = float(((glucose > 180) & (glucose <= 250)).mean() * 100.0)
    very_high = float((glucose > 250).mean() * 100.0)
    dose_delta = (proposed - delivered).clip(lower=0)

    report = safety_report or {}
    summary = {
        "total_steps": int(len(results_df)),
        "duration_minutes": float(time.max() - time.min()) if len(time) > 1 else 0.0,
        "intervention_count": int(triggered.sum()),
        "intervention_rate_pct": float(triggered.mean() * 100.0),
        "top_reasons": reason_counts,
        "glucose": {
            "min_mgdl": float(glucose.min()),
            "max_mgdl": float(glucose.max()),
            "mean_mgdl": float(glucose.mean()),
            "latest_mgdl": float(glucose.iloc[-1]),
            "steps_below_70": int((glucose < 70).sum()),
            "steps_below_54": int((glucose < 54).sum()),
            "steps_above_250": int((glucose > 250).sum()),
        },
        "insulin": {
            "delivered_total_units": float(delivered.sum()),
            "algorithm_requested_total_units": float(proposed.sum()),
            "blocked_or_reduced_total_units": float(dose_delta.sum()),
            "max_single_delivered_units": float(delivered.max()),
            "max_single_requested_units": float(proposed.max()),
        },
        "time_in_ranges_pct": {
            "very_low_lt_54": very_low,
            "low_54_69": max(0.0, low - very_low),
            "target_70_180": target,
            "high_181_250": high,
            "very_high_gt_250": very_high,
        },
        "terminated_early": bool(report.get("terminated_early", False)),
        "termination_reason": report.get("termination_reason"),
        "safety_report_keys": sorted(str(key) for key in report.keys()),
    }
    return _json_ready(summary)


def _svg_glucose_trace(df: pd.DataFrame, width: int = 920, height: int = 260) -> str:
    if df.empty:
        return "<p>No glucose data available.</p>"

    working = df.copy()
    if len(working) > 600:
        step = max(1, len(working) // 600)
        working = working.iloc[::step].copy()

    time = _numeric_series(working, "time_minutes")
    glucose = _numeric_series(working, "glucose_actual_mgdl")
    if glucose.empty:
        return "<p>No glucose data available.</p>"

    min_t = float(time.min())
    max_t = float(time.max())
    min_g = min(40.0, float(glucose.min()) - 10.0)
    max_g = max(260.0, float(glucose.max()) + 10.0)
    if max_t <= min_t:
        max_t = min_t + 1.0
    if max_g <= min_g:
        max_g = min_g + 1.0

    left, right, top, bottom = 52.0, 20.0, 20.0, 38.0
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_for(value: float) -> float:
        return left + ((value - min_t) / (max_t - min_t)) * plot_w

    def y_for(value: float) -> float:
        return top + (1.0 - ((value - min_g) / (max_g - min_g))) * plot_h

    points = " ".join(f"{x_for(float(t)):.1f},{y_for(float(g)):.1f}" for t, g in zip(time, glucose))
    target_top = y_for(180.0)
    target_bottom = y_for(70.0)
    target_y = min(target_top, target_bottom)
    target_h = abs(target_bottom - target_top)
    low_y = y_for(70.0)
    high_y = y_for(180.0)

    intervention_marks = ""
    if "safety_triggered" in working.columns:
        triggered = _triggered_mask(working)
        for t, g in zip(time[triggered], glucose[triggered]):
            intervention_marks += (
                f'<circle cx="{x_for(float(t)):.1f}" cy="{y_for(float(g)):.1f}" r="4.5" '
                'fill="#b91c1c" stroke="#fff" stroke-width="1.2" />'
            )

    return f"""
<svg class="trace" viewBox="0 0 {width} {height}" role="img" aria-label="Glucose trace with safety interventions">
  <rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#f8fafc" />
  <rect x="{left}" y="{target_y:.1f}" width="{plot_w}" height="{target_h:.1f}" fill="#dcfce7" opacity="0.9" />
  <line x1="{left}" y1="{low_y:.1f}" x2="{width-right}" y2="{low_y:.1f}" stroke="#16a34a" stroke-width="1" />
  <line x1="{left}" y1="{high_y:.1f}" x2="{width-right}" y2="{high_y:.1f}" stroke="#16a34a" stroke-width="1" />
  <polyline points="{points}" fill="none" stroke="#0f172a" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" />
  {intervention_marks}
  <text x="12" y="{low_y + 4:.1f}" font-size="12" fill="#166534">70</text>
  <text x="10" y="{high_y + 4:.1f}" font-size="12" fill="#166534">180</text>
  <text x="{left}" y="{height-12}" font-size="12" fill="#475569">{min_t:.0f} min</text>
  <text x="{width-right-74}" y="{height-12}" font-size="12" fill="#475569">{max_t:.0f} min</text>
</svg>
"""


def build_safety_visualizer_html(
    results_df: pd.DataFrame,
    safety_report: Optional[Dict[str, Any]] = None,
    *,
    title: str = "IINTS Safety Contract Visualizer",
) -> str:
    summary = summarize_safety_trace(results_df, safety_report=safety_report)
    reasons = summary.get("top_reasons", {})
    reason_rows = "\n".join(
        f"<tr><td>{html.escape(str(reason))}</td><td>{count}</td></tr>" for reason, count in reasons.items()
    ) or "<tr><td>No safety interventions</td><td>0</td></tr>"
    ranges = summary.get("time_in_ranges_pct", {})
    range_rows = "\n".join(
        f"<tr><td>{html.escape(str(label).replace('_', ' '))}</td><td>{_safe_float(value):.1f}%</td></tr>"
        for label, value in ranges.items()
    )
    insulin = summary.get("insulin", {})
    glucose = summary.get("glucose", {})
    embedded = html.escape(json.dumps(summary, indent=2))
    trace_svg = _svg_glucose_trace(results_df)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --paper: #f8fafc;
      --card: #ffffff;
      --green: #166534;
      --red: #991b1b;
      --blue: #1d4ed8;
    }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      color: var(--ink);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 18px 48px;
    }}
    header {{
      border-bottom: 4px solid var(--ink);
      margin-bottom: 20px;
      padding-bottom: 14px;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 44px); }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin: 18px 0;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 16px 34px rgba(15, 23, 42, 0.08);
      padding: 18px;
    }}
    .metric {{ font-size: 30px; font-weight: 800; letter-spacing: -0.03em; }}
    .metric.good {{ color: var(--green); }}
    .metric.warn {{ color: var(--red); }}
    .label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 6px; text-align: left; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 14px;
      padding: 14px;
      overflow: auto;
    }}
    .trace {{ width: 100%; height: auto; display: block; }}
    .boundary {{
      border-left: 5px solid var(--blue);
      background: #eff6ff;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Research-only visualization of the safety supervisor: glucose trace, intervention points, blocked insulin, and top safety reasons.</p>
  </header>

  <section class="card boundary">
    <h2>Interpretation Boundary</h2>
    <p>This is not a medical-device safety certificate. It is a transparent research artifact for inspecting how a simulated safety layer behaved during one run.</p>
  </section>

  <section class="grid">
    <div class="card"><div class="label">Safety interventions</div><div class="metric warn">{summary.get("intervention_count", 0)}</div><p>{_safe_float(summary.get("intervention_rate_pct")):.1f}% of simulation steps</p></div>
    <div class="card"><div class="label">Mean glucose</div><div class="metric">{_safe_float(glucose.get("mean_mgdl")):.1f}</div><p>mg/dL</p></div>
    <div class="card"><div class="label">Delivered insulin</div><div class="metric">{_safe_float(insulin.get("delivered_total_units")):.2f}</div><p>total units in simulation</p></div>
    <div class="card"><div class="label">Blocked / reduced insulin</div><div class="metric">{_safe_float(insulin.get("blocked_or_reduced_total_units")):.2f}</div><p>units prevented by safety logic</p></div>
  </section>

  <section class="card">
    <h2>Glucose Trace And Safety Events</h2>
    {trace_svg}
  </section>

  <section class="grid">
    <div class="card">
      <h2>Top Safety Reasons</h2>
      <table><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody>{reason_rows}</tbody></table>
    </div>
    <div class="card">
      <h2>Time In Ranges</h2>
      <table><thead><tr><th>Range</th><th>Percent</th></tr></thead><tbody>{range_rows}</tbody></table>
    </div>
  </section>

  <section class="card">
    <h2>Embedded JSON Summary</h2>
    <pre>{embedded}</pre>
  </section>
</main>
</body>
</html>
"""


def write_safety_visualizer(
    results_df: pd.DataFrame,
    output_html: str | Path,
    *,
    output_json: str | Path | None = None,
    safety_report: Optional[Dict[str, Any]] = None,
    title: str = "IINTS Safety Contract Visualizer",
) -> Dict[str, str]:
    html_path = Path(output_html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_safety_trace(results_df, safety_report=safety_report)
    html_path.write_text(
        build_safety_visualizer_html(results_df, safety_report=safety_report, title=title),
        encoding="utf-8",
    )
    outputs = {"html": str(html_path)}
    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        outputs["json"] = str(json_path)
    return outputs
