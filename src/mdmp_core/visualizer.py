from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from typing import Any
import json


def build_dashboard_html(report: dict[str, Any], *, title: str = "MDMP Report") -> str:
    report_json = json.dumps(report, indent=2)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows = "\n".join(
        f"<tr><td>{escape(str(c.get('name', '')))}</td><td>{'PASS' if c.get('passed') else 'FAIL'}</td><td>{escape(str(c.get('detail', '')))}</td></tr>"
        for c in report.get("checks", [])
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #0f172a; }}
    .card {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }}
    .k {{ color: #475569; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .05em; }}
    .v {{ font-size: 1.4rem; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; text-align: left; padding: 0.6rem; }}
    pre {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p>Generated: {escape(stamp)}</p>

  <div class=\"grid\">
    <div class=\"card\"><div class=\"k\">Grade</div><div class=\"v\">{escape(str(report.get('grade', '-')))}</div></div>
    <div class=\"card\"><div class=\"k\">Grade Reason</div><div class=\"v\">{escape(str(report.get('grade_reason', '-')))}</div></div>
    <div class=\"card\"><div class=\"k\">Compliance</div><div class=\"v\">{escape(str(report.get('compliance_score', '-')))}%</div></div>
    <div class=\"card\"><div class=\"k\">Rows</div><div class=\"v\">{escape(str(report.get('row_count', '-')))}</div></div>
    <div class=\"card\"><div class=\"k\">Consent Verified</div><div class=\"v\">{escape(str(report.get('consent_verified', '-')))}</div></div>
  </div>

  <div class=\"card\">
    <h2>Validation Checks</h2>
    <table>
      <thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class=\"card\">
    <h2>Raw Report JSON</h2>
    <pre>{escape(report_json)}</pre>
  </div>
</body>
</html>
"""
