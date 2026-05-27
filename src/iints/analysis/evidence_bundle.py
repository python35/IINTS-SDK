from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import shutil

import pandas as pd

from iints.utils.run_io import compute_sha256


EVIDENCE_SCOPE = (
    "Research and education only. This bundle is not clinical evidence, "
    "not a medical-device submission, and not dosing advice."
)


@dataclass(frozen=True)
class EvidenceRun:
    label: str
    source_dir: Path
    bundle_dir: Path
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "source_dir": str(self.source_dir),
            "bundle_dir": str(self.bundle_dir),
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }


def _safe_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "run"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _copy_if_exists(source: Path, target_dir: Path, name: Optional[str] = None) -> Optional[Path]:
    if not source.is_file():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (name or source.name)
    shutil.copy2(source, target)
    return target


def _find_first(root: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    for name in names:
        matches = sorted(root.glob(f"**/{name}"))
        if matches:
            return matches[0]
    return None


def _glucose_metrics(results_csv: Optional[Path]) -> Dict[str, Any]:
    if results_csv is None or not results_csv.is_file():
        return {}
    try:
        df = pd.read_csv(results_csv)
    except Exception:
        return {}
    glucose_column = None
    for candidate in ("glucose_actual_mgdl", "glucose", "glucose_mgdl", "current_glucose"):
        if candidate in df.columns:
            glucose_column = candidate
            break
    if glucose_column is None or df.empty:
        return {"rows": int(len(df))}
    glucose = pd.to_numeric(df[glucose_column], errors="coerce").dropna()
    if glucose.empty:
        return {"rows": int(len(df))}
    time_column = "time_minutes" if "time_minutes" in df.columns else "timestamp" if "timestamp" in df.columns else None
    duration_minutes = None
    if time_column is not None:
        time_values = pd.to_numeric(df[time_column], errors="coerce").dropna()
        if not time_values.empty:
            duration_minutes = float(time_values.max() - time_values.min())
    return {
        "rows": int(len(df)),
        "duration_minutes": duration_minutes,
        "mean_glucose_mgdl": round(float(glucose.mean()), 3),
        "min_glucose_mgdl": round(float(glucose.min()), 3),
        "max_glucose_mgdl": round(float(glucose.max()), 3),
        "tir_70_180_pct": round(float(((glucose >= 70) & (glucose <= 180)).mean() * 100.0), 3),
        "tir_below_70_pct": round(float((glucose < 70).mean() * 100.0), 3),
        "tir_below_54_pct": round(float((glucose < 54).mean() * 100.0), 3),
    }


def _bundle_one_run(label: str, run_dir: Path, output_runs_dir: Path) -> EvidenceRun:
    source = run_dir.expanduser().resolve()
    warnings: list[str] = []
    run_target = output_runs_dir / _safe_slug(label)
    artifacts_dir = run_target / "artifacts"
    run_target.mkdir(parents=True, exist_ok=True)

    results_csv = _find_first(source, ("results.csv", "sim_results.csv"))
    summary_json = _find_first(source, ("summary.json", "demo_summary.json", "run_summary.json"))
    manifest_json = _find_first(source, ("run_manifest.json", "manifest.json"))
    safety_json = _find_first(source, ("safety_report.json", "safety.json"))
    report_pdf = _find_first(source, ("report.pdf", "clinical_report.pdf", "research_report.pdf"))

    artifacts: dict[str, str] = {}
    for key, path in {
        "results_csv": results_csv,
        "summary_json": summary_json,
        "run_manifest_json": manifest_json,
        "safety_report_json": safety_json,
        "report_pdf": report_pdf,
    }.items():
        if path is None:
            continue
        copied = _copy_if_exists(path, artifacts_dir)
        if copied is not None:
            artifacts[key] = str(copied)
            artifacts[f"{key}_sha256"] = compute_sha256(copied)

    if results_csv is None:
        warnings.append("No results.csv-style file found; metrics are limited.")
    if manifest_json is None:
        warnings.append("No run manifest found; reproducibility evidence is incomplete.")
    if safety_json is None:
        warnings.append("No safety report found; safety evidence is incomplete.")

    metrics = _glucose_metrics(results_csv)
    if safety_json is not None:
        safety_payload = _read_json(safety_json)
        for key in ("terminated_early", "critical_events", "supervisor_interventions"):
            if key in safety_payload:
                metrics[key] = safety_payload[key]
    if summary_json is not None:
        summary_payload = _read_json(summary_json)
        for key in ("status", "completion_ratio", "completed_duration_minutes", "requested_duration_minutes"):
            if key in summary_payload:
                metrics[key] = summary_payload[key]

    run_card = {
        "label": label,
        "source_dir": str(source),
        "artifacts": artifacts,
        "metrics": metrics,
        "warnings": warnings,
        "scope": EVIDENCE_SCOPE,
    }
    (run_target / "RUN_CARD.json").write_text(json.dumps(run_card, indent=2), encoding="utf-8")
    return EvidenceRun(label=label, source_dir=source, bundle_dir=run_target, artifacts=artifacts, metrics=metrics, warnings=warnings)


def _load_optional_json(root: Optional[Path], names: Iterable[str]) -> Dict[str, Any]:
    if root is None:
        return {}
    path = _find_first(root.expanduser().resolve(), names)
    return _read_json(path) if path is not None else {}


def _write_markdown(path: Path, title: str, payload: Dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        EVIDENCE_SCOPE,
        "",
        "## Summary",
        "",
        f"- Created UTC: `{payload['created_utc']}`",
        f"- Runs: `{len(payload['runs'])}`",
        f"- Local AI evidence: `{'yes' if payload.get('local_ai') else 'no'}`",
        f"- Pump bench evidence: `{'yes' if payload.get('pump_bench') else 'no'}`",
        "",
        "## Runs",
        "",
    ]
    if not payload["runs"]:
        lines.append("No simulation runs were attached.")
        lines.append("")
    for run in payload["runs"]:
        metrics = run.get("metrics", {})
        lines.extend(
            [
                f"### {run['label']}",
                "",
                f"- Source: `{run['source_dir']}`",
                f"- Bundle: `{run['bundle_dir']}`",
                f"- Rows: `{metrics.get('rows', 'n/a')}`",
                f"- Mean glucose: `{metrics.get('mean_glucose_mgdl', 'n/a')}` mg/dL",
                f"- TIR 70-180: `{metrics.get('tir_70_180_pct', 'n/a')}`%",
                f"- Time <70: `{metrics.get('tir_below_70_pct', 'n/a')}`%",
                f"- Time <54: `{metrics.get('tir_below_54_pct', 'n/a')}`%",
                "",
            ]
        )
        for warning in run.get("warnings", []):
            lines.append(f"- Warning: {warning}")
        if run.get("warnings"):
            lines.append("")
    lines.extend(
        [
            "## Required Interpretation",
            "",
            "- Use this as reproducibility and engineering evidence.",
            "- Do not present this as clinical performance evidence.",
            "- Keep local AI and pump exports behind deterministic safety gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_model_card(path: Path, payload: Dict[str, Any]) -> None:
    local_ai = payload.get("local_ai") or {}
    gate = local_ai.get("training_safety_gate") or local_ai.get("closed_loop_evaluation", {}).get("safety_gate") or {}
    lines = [
        "# IINTS Research Model Card",
        "",
        EVIDENCE_SCOPE,
        "",
        "## Intended Use",
        "",
        "Local AI models generated by IINTS are for simulator research, reproducibility studies, and bench-only demonstrations.",
        "",
        "## Not Intended For",
        "",
        "- real insulin delivery",
        "- diagnosis or treatment",
        "- autonomous therapy without certified medical-device controls",
        "",
        "## Evidence Inputs",
        "",
        f"- Attached runs: `{len(payload['runs'])}`",
        f"- Local AI summary present: `{'yes' if local_ai else 'no'}`",
        "",
        "## Safety Gate",
        "",
        f"- Status: `{gate.get('status', 'not_available')}`",
        f"- Passed: `{gate.get('passed', 'not_available')}`",
        f"- Score: `{gate.get('score', 'not_available')}`",
        "",
    ]
    if gate.get("critical_failures"):
        lines.extend(["### Critical Failures", ""])
        lines.extend(f"- {item}" for item in gate["critical_failures"])
        lines.append("")
    lines.extend(
        [
            "## Promotion Rule",
            "",
            "A model can only move toward hardware bench testing after realism checks, closed-loop evaluation, MDMP/evidence review, and deterministic supervisor gates pass.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_evidence_bundle(
    run_dirs: Iterable[tuple[str, Path]],
    *,
    output_dir: Path,
    title: str = "IINTS Research Evidence Bundle",
    local_ai_dir: Optional[Path] = None,
    pump_bundle_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create a compact evidence pack for demos, reviews, and EUCYS-style explanation."""

    root = output_dir.expanduser().resolve()
    runs_dir = root / "runs"
    root.mkdir(parents=True, exist_ok=True)

    runs = [_bundle_one_run(label, Path(run_dir), runs_dir).to_dict() for label, run_dir in run_dirs]
    local_ai = _load_optional_json(local_ai_dir, ("LOCAL_AI_RESEARCH_SUMMARY.json", "summary.json"))
    pump_bench = _load_optional_json(pump_bundle_dir, ("manifest.json", "iints_pump_manifest.json"))

    payload: Dict[str, Any] = {
        "title": title,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": EVIDENCE_SCOPE,
        "runs": runs,
        "local_ai": local_ai,
        "pump_bench": pump_bench,
        "artifacts": {
            "summary_json": str(root / "evidence_summary.json"),
            "readme_md": str(root / "README.md"),
            "model_card_md": str(root / "MODEL_CARD.md"),
            "run_index_csv": str(root / "run_index.csv"),
        },
    }

    (root / "evidence_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(root / "README.md", title, payload)
    _write_model_card(root / "MODEL_CARD.md", payload)

    rows = []
    for run in runs:
        metrics = run.get("metrics", {})
        rows.append(
            {
                "label": run["label"],
                "source_dir": run["source_dir"],
                "bundle_dir": run["bundle_dir"],
                "rows": metrics.get("rows"),
                "mean_glucose_mgdl": metrics.get("mean_glucose_mgdl"),
                "tir_70_180_pct": metrics.get("tir_70_180_pct"),
                "tir_below_70_pct": metrics.get("tir_below_70_pct"),
                "tir_below_54_pct": metrics.get("tir_below_54_pct"),
                "warnings": "; ".join(run.get("warnings", [])),
            }
        )
    pd.DataFrame(rows).to_csv(root / "run_index.csv", index=False)
    return payload
