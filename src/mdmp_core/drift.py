from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


WARN_THRESHOLD = 0.05
CRITICAL_THRESHOLD = 0.15


@dataclass
class ColumnDrift:
    column: str
    metrics: Dict[str, dict] = field(default_factory=dict)
    severity: str = "ok"  # ok | warn | critical

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "metrics": self.metrics,
            "severity": self.severity,
        }


@dataclass
class DriftReport:
    dataset_before: str
    dataset_after: str
    columns: List[ColumnDrift]
    overall_severity: str
    row_count_before: int
    row_count_after: int

    @property
    def has_drift(self) -> bool:
        return self.overall_severity != "ok"

    def to_dict(self) -> dict:
        return {
            "dataset_before": self.dataset_before,
            "dataset_after": self.dataset_after,
            "row_count_before": self.row_count_before,
            "row_count_after": self.row_count_after,
            "overall_severity": self.overall_severity,
            "has_drift": self.has_drift,
            "columns": [c.to_dict() for c in self.columns],
        }


def _severity(delta_pct: float) -> str:
    if delta_pct > CRITICAL_THRESHOLD:
        return "critical"
    if delta_pct > WARN_THRESHOLD:
        return "warn"
    return "ok"


def _severity_rank(value: str) -> int:
    return {"ok": 0, "warn": 1, "critical": 2}.get(value, 0)


def _load_columns(path: str) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return result

    for col in rows[0].keys():
        values: List[float] = []
        for row in rows:
            try:
                values.append(float(row[col]))
            except (ValueError, KeyError, TypeError):
                continue
        if values:
            result[col] = values

    return result


def _row_count(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        lines = sum(1 for _ in handle)
    return max(lines - 1, 0)


def compute_drift(
    path_before: str,
    path_after: str,
    columns: Optional[List[str]] = None,
) -> DriftReport:
    data_before = _load_columns(path_before)
    data_after = _load_columns(path_after)

    rows_before = _row_count(path_before)
    rows_after = _row_count(path_after)

    common = set(data_before.keys()) & set(data_after.keys())
    if columns:
        common &= set(columns)

    column_results: List[ColumnDrift] = []
    worst_severity = "ok"

    for col in sorted(common):
        b = data_before[col]
        a = data_after[col]

        if len(b) < 2 or len(a) < 2:
            continue

        col_drift = ColumnDrift(column=col)
        col_worst = "ok"

        for metric_name, fn in [
            ("mean", statistics.mean),
            ("stdev", statistics.stdev),
            ("min", min),
            ("max", max),
            ("median", statistics.median),
        ]:
            try:
                val_b = float(fn(b))
                val_a = float(fn(a))
                delta = val_a - val_b
                delta_pct = abs(delta / val_b) if val_b != 0 else 0.0
                sev = _severity(delta_pct)

                col_drift.metrics[metric_name] = {
                    "before": round(val_b, 6),
                    "after": round(val_a, 6),
                    "delta": round(delta, 6),
                    "delta_pct": round(delta_pct * 100.0, 2),
                    "severity": sev,
                }

                if _severity_rank(sev) > _severity_rank(col_worst):
                    col_worst = sev
            except (statistics.StatisticsError, OverflowError, TypeError, ValueError):
                continue

        col_drift.severity = col_worst
        if _severity_rank(col_worst) > _severity_rank(worst_severity):
            worst_severity = col_worst

        column_results.append(col_drift)

    return DriftReport(
        dataset_before=path_before,
        dataset_after=path_after,
        columns=column_results,
        overall_severity=worst_severity,
        row_count_before=rows_before,
        row_count_after=rows_after,
    )


def format_drift_report(report: DriftReport) -> str:
    icons = {"ok": "✓", "warn": "⚠", "critical": "✗"}
    lines = [
        "MDMP Drift Report",
        "═" * 50,
        f"Before: {report.dataset_before}",
        f"After:  {report.dataset_after}",
        f"Rows:   {report.row_count_before} -> {report.row_count_after}",
        "",
    ]

    for col in report.columns:
        lines.append(f"Column: {col.column}")
        for metric, data in col.metrics.items():
            icon = icons[data["severity"]]
            lines.append(
                f"  {metric:<8} "
                f"{data['before']:>10.3f} -> {data['after']:>10.3f}  "
                f"({data['delta_pct']:+.1f}%)  {icon} {data['severity'].upper()}"
            )
        lines.append("")

    overall_icon = icons[report.overall_severity]
    lines += [
        "═" * 50,
        f"Overall severity: {overall_icon} {report.overall_severity.upper()}",
    ]

    if report.overall_severity != "ok":
        lines.append("Recommendation: re-validate and re-grade before AI training.")

    return "\n".join(lines)


def severity_at_or_above(actual: str, threshold: str) -> bool:
    return _severity_rank(actual) >= _severity_rank(threshold)


def valid_severities() -> tuple[str, ...]:
    return ("ok", "warn", "critical")
