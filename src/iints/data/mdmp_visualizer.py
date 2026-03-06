from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from typing import Any, Dict, Optional


def build_mdmp_dashboard_html(
    report: Dict[str, Any],
    *,
    title: str = "IINTS MDMP Certification Dashboard",
    generated_at_utc: Optional[str] = None,
) -> str:
    """Build a single-file interactive HTML dashboard for an MDMP report."""

    stamp = generated_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    escaped_title = html.escape(title)
    escaped_stamp = html.escape(stamp)
    report_json = json.dumps(report, ensure_ascii=False)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-soft: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --ok: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
      --accent: #38bdf8;
      --border: #334155;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      background: radial-gradient(circle at 0 0, #172554, var(--bg) 36%), var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    .header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    .title {
      margin: 0;
      font-size: clamp(24px, 3.5vw, 38px);
      letter-spacing: 0.4px;
    }
    .subtitle {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(30, 41, 59, 0.6);
      padding: 10px 14px;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 14px;
    }
    .card {
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(31, 41, 55, 0.96), rgba(17, 24, 39, 0.96));
      border-radius: 16px;
      padding: 16px;
    }
    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-7 { grid-column: span 7; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    @media (max-width: 960px) {
      .span-4, .span-5, .span-7, .span-8 { grid-column: span 12; }
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-bottom: 4px;
    }
    .value {
      font-size: 26px;
      font-weight: 700;
      line-height: 1.15;
      word-break: break-word;
    }
    .value.small { font-size: 18px; }
    .value.code {
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.4;
      overflow-wrap: anywhere;
      color: #bfdbfe;
      background: rgba(15, 23, 42, 0.5);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      margin-top: 4px;
    }
    .progress {
      margin-top: 12px;
      height: 12px;
      width: 100%;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.15);
      border: 1px solid var(--border);
      overflow: hidden;
    }
    .progress > div {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #38bdf8, #22c55e);
      transition: width 240ms ease;
    }
    .grade-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-top: 8px;
      min-height: 44px;
      padding: 8px 14px;
      border-radius: 12px;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.3px;
      border: 1px solid transparent;
      text-transform: uppercase;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .controls input, .controls select, .controls button {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.65);
      color: var(--text);
      padding: 8px 10px;
      font-size: 13px;
      font-family: var(--sans);
    }
    .controls button {
      cursor: pointer;
    }
    .controls button:hover {
      border-color: #64748b;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.15);
      vertical-align: top;
    }
    th {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.4px;
      font-size: 11px;
      font-weight: 600;
    }
    .pass { color: var(--ok); font-weight: 700; }
    .fail { color: var(--bad); font-weight: 700; }
    .hint {
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .tiny {
      color: #94a3b8;
      font-size: 12px;
      margin-top: 10px;
    }
    .split {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    @media (max-width: 720px) {
      .split {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <h1 class="title">__TITLE__</h1>
        <div class="subtitle">Generated: __STAMP__</div>
      </div>
      <div class="badge">Research Use Only - Not for clinical dosing</div>
    </div>

    <div class="grid">
      <section class="card span-4">
        <div class="label">MDMP Grade</div>
        <div class="value" id="gradeText">-</div>
        <div class="grade-pill" id="gradePill">-</div>
      </section>
      <section class="card span-4">
        <div class="label">Compliance Score</div>
        <div class="value" id="complianceValue">-</div>
        <div class="progress"><div id="complianceBar"></div></div>
        <div class="tiny" id="statusText">-</div>
      </section>
      <section class="card span-4">
        <div class="label">Certification</div>
        <div class="value small" id="certifiedValue">-</div>
        <div class="tiny">Protocol: <span id="protocolValue">-</span></div>
        <div class="tiny">Rows validated: <span id="rowCountValue">-</span></div>
      </section>

      <section class="card span-7">
        <div class="label">Audit Summary</div>
        <div class="split">
          <div>
            <div class="label">Checks Passed</div>
            <div class="value small" id="checksPassed">-</div>
          </div>
          <div>
            <div class="label">Checks Failed</div>
            <div class="value small" id="checksFailed">-</div>
          </div>
          <div>
            <div class="label">Failed Rows (Total)</div>
            <div class="value small" id="failedRowsTotal">-</div>
          </div>
          <div>
            <div class="label">Output Columns</div>
            <div class="value small" id="outputCols">-</div>
          </div>
        </div>
      </section>

      <section class="card span-5">
        <div class="label">Fingerprints</div>
        <div class="label">Contract SHA-256</div>
        <div class="value code" id="contractFp">-</div>
        <div class="label" style="margin-top:8px;">Dataset SHA-256</div>
        <div class="value code" id="datasetFp">-</div>
      </section>

      <section class="card span-12">
        <div class="controls">
          <label for="fileInput">Load JSON report:</label>
          <input id="fileInput" type="file" accept=".json,application/json">
          <select id="statusFilter">
            <option value="all">All checks</option>
            <option value="pass">Passed only</option>
            <option value="fail">Failed only</option>
          </select>
          <button id="resetBtn" type="button">Reset Embedded Report</button>
        </div>
        <table>
          <thead>
            <tr>
              <th style="width: 18%;">Check</th>
              <th style="width: 10%;">Passed</th>
              <th style="width: 12%;">Failed rows</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody id="checksBody"></tbody>
        </table>
        <div class="hint">
          Dashboard is deterministic from the loaded JSON report. Keep the report file with your run artifacts for reproducibility.
        </div>
      </section>
    </div>
  </div>

  <script id="mdmp-report" type="application/json">__REPORT_JSON__</script>
  <script>
    const initialReport = JSON.parse(document.getElementById("mdmp-report").textContent);
    let currentReport = initialReport;
    let filterState = "all";

    const el = (id) => document.getElementById(id);

    function safeNum(value) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : 0;
    }

    function gradeStyle(grade) {
      if (grade === "clinical_grade") {
        return { bg: "rgba(34, 197, 94, 0.16)", color: "#86efac", border: "rgba(34, 197, 94, 0.55)" };
      }
      if (grade === "research_grade") {
        return { bg: "rgba(245, 158, 11, 0.16)", color: "#fcd34d", border: "rgba(245, 158, 11, 0.55)" };
      }
      return { bg: "rgba(239, 68, 68, 0.14)", color: "#fca5a5", border: "rgba(239, 68, 68, 0.55)" };
    }

    function renderSummary(report) {
      const score = safeNum(report.compliance_score);
      const grade = String(report.mdmp_grade || "draft");
      const checks = Array.isArray(report.checks) ? report.checks : [];
      const passed = checks.filter((item) => Boolean(item.passed)).length;
      const failed = checks.length - passed;
      const failedRows = checks.reduce((acc, item) => acc + safeNum(item.failed_rows), 0);

      el("gradeText").textContent = grade.replace("_", " ");
      const gradePill = el("gradePill");
      gradePill.textContent = grade;
      const gs = gradeStyle(grade);
      gradePill.style.background = gs.bg;
      gradePill.style.color = gs.color;
      gradePill.style.borderColor = gs.border;

      el("complianceValue").textContent = score.toFixed(2) + "%";
      el("complianceBar").style.width = Math.max(0, Math.min(100, score)).toFixed(2) + "%";
      el("statusText").textContent = Boolean(report.is_compliant) ? "Status: PASS" : "Status: FAIL";

      el("certifiedValue").textContent = Boolean(report.certified_for_medical_research) ? "Certified for medical research: YES" : "Certified for medical research: NO";
      el("protocolValue").textContent = String(report.mdmp_protocol_version || "n/a");
      el("rowCountValue").textContent = String(safeNum(report.row_count));

      el("checksPassed").textContent = String(passed);
      el("checksFailed").textContent = String(failed);
      el("failedRowsTotal").textContent = String(failedRows);
      const outputColumns = Array.isArray(report.output_columns) ? report.output_columns : [];
      el("outputCols").textContent = outputColumns.length ? String(outputColumns.length) : "0";

      el("contractFp").textContent = String(report.contract_fingerprint_sha256 || "n/a");
      el("datasetFp").textContent = String(report.dataset_fingerprint_sha256 || "n/a");
    }

    function renderChecks(report) {
      const checks = Array.isArray(report.checks) ? report.checks : [];
      const filtered = checks.filter((item) => {
        if (filterState === "pass") return Boolean(item.passed);
        if (filterState === "fail") return !Boolean(item.passed);
        return true;
      });

      const rows = filtered.map((item) => {
        const passed = Boolean(item.passed);
        const statusClass = passed ? "pass" : "fail";
        const statusText = passed ? "YES" : "NO";
        const detail = String(item.detail || "");
        return (
          "<tr>" +
            "<td><code>" + String(item.name || "unknown") + "</code></td>" +
            "<td class='" + statusClass + "'>" + statusText + "</td>" +
            "<td>" + String(safeNum(item.failed_rows)) + "</td>" +
            "<td>" + detail.replace(/</g, "&lt;").replace(/>/g, "&gt;") + "</td>" +
          "</tr>"
        );
      }).join("");

      el("checksBody").innerHTML = rows || "<tr><td colspan='4'>No checks available for current filter.</td></tr>";
    }

    function render(report) {
      renderSummary(report);
      renderChecks(report);
    }

    function loadFromFile(file) {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const parsed = JSON.parse(String(reader.result || "{}"));
          currentReport = parsed;
          render(currentReport);
        } catch (error) {
          alert("Could not parse JSON report: " + error);
        }
      };
      reader.readAsText(file);
    }

    el("statusFilter").addEventListener("change", (event) => {
      filterState = event.target.value;
      render(currentReport);
    });

    el("fileInput").addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) loadFromFile(file);
    });

    el("resetBtn").addEventListener("click", () => {
      currentReport = initialReport;
      filterState = "all";
      el("statusFilter").value = "all";
      el("fileInput").value = "";
      render(currentReport);
    });

    render(currentReport);
  </script>
</body>
</html>
"""

    return (
        template.replace("__TITLE__", escaped_title)
        .replace("__STAMP__", escaped_stamp)
        .replace("__REPORT_JSON__", report_json)
    )
