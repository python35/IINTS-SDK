"""Prepare a durable desktop run handover using the SDK's existing exporters."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from iints_desktop.engine import DesktopRunResult


RUN_BUNDLE_FILENAME = "desktop_run_bundle.json"


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_run_bundle(
    run: DesktopRunResult,
    *,
    auto_prepare: bool = True,
    ai_review: bool = False,
    ai_model: str = "ministral-3:8b",
    ai_host: str = "http://127.0.0.1:11434",
    progress: Callable[[str, float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Keep successful outputs even when an optional exporter is unavailable.

    Each stage is recorded before and after execution. The completed simulation
    can therefore be reopened even if the app closes during post-processing.
    """
    root = run.output_dir
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        **asdict(run),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preparing",
        "steps": [],
        "artifacts": [],
        "metrics": {},
        "manifest_path": root / RUN_BUNDLE_FILENAME,
        "source_sha256": _source_digest(run.results_csv) if run.results_csv and run.results_csv.is_file() else None,
    }

    def save() -> None:
        _json_file(root / RUN_BUNDLE_FILENAME, payload)

    def artifact(label: str, value: Any) -> None:
        if value and Path(value).is_file():
            payload["artifacts"].append({"label": label, "path": str(value)})

    artifact("Results CSV", run.results_csv)
    artifact("PDF report", run.report_pdf)
    artifact("Run configuration", run.config_path)

    def stage(key: str, title: str, percent: float, action: Callable[[], None], enabled: bool = True) -> None:
        row: dict[str, Any] = {"key": key, "title": title, "status": "running", "detail": ""}
        payload["steps"].append(row)
        if not enabled or (cancelled and cancelled()):
            row.update(status="skipped", detail="Disabled in run preferences." if not enabled else "Preparation cancelled; simulation outputs preserved.")
            save()
            return
        save()
        if progress:
            progress(key, percent, title)
        try:
            action()
        except Exception as error:  # optional preparation must not erase a successful run
            row.update(status="warning", detail=f"{type(error).__name__}: {error}")
        else:
            row["status"] = "ready"
        save()

    def inspect() -> None:
        from iints_desktop.results import load_results_preview, screen_results_csv

        if run.results_csv is None:
            raise FileNotFoundError("The simulation did not produce a results CSV.")
        preview = load_results_preview(run.results_csv, max_rows=80)
        payload["metrics"] = preview.metrics
        payload["row_count"] = preview.row_count
        payload["screen"] = asdict(screen_results_csv(run.results_csv))
        artifact("Glucose figure", preview.graph_path)
        metrics_path = root / "desktop_metrics.json"
        _json_file(metrics_path, {"metrics": preview.metrics, "row_count": preview.row_count, "screen": payload["screen"]})
        artifact("Summary metrics", metrics_path)

    def certify() -> None:
        from iints_desktop.mdmp import create_desktop_mdmp_certificate

        if run.results_csv is None:
            raise FileNotFoundError("No results CSV available for the data-contract check.")
        result = create_desktop_mdmp_certificate(run.results_csv, quick=False)
        payload["mdmp"] = {key: value for key, value in asdict(result).items() if key != "private_key_path"}
        payload["mdmp"]["quick"] = False
        artifact("MDMP certificate", result.certificate_path)
        artifact("MDMP report", result.report_path)
        artifact("Verification public key", result.public_key_path)

    def review() -> None:
        from iints_desktop.local_ai import ask_local_ai, check_local_ai

        readiness = check_local_ai(model=ai_model, host=ai_host)
        if not readiness.available:
            raise RuntimeError(readiness.message)
        if run.results_csv is None:
            raise FileNotFoundError("No results CSV available for review.")
        answer = ask_local_ai(
            question="Review this simulated run. Describe limitations and concrete research checks using the supplied facts.",
            model=ai_model, host=ai_host, result_csv=run.results_csv,
        )
        payload["ai_review"] = asdict(answer)
        review_path = root / "local_ai_review.md"
        review_path.write_text(f"# Local AI review\n\nModel: {answer.model}\n\n{answer.answer}\n", encoding="utf-8")
        _json_file(root / "local_ai_review.json", asdict(answer))
        artifact("Local AI review", review_path)

    stage("review", "Calculate metrics and glucose figure", 82, inspect)
    stage("mdmp", "Check the complete CSV data contract", 88, certify, auto_prepare)
    stage("ai", "Review results with local Ollama", 92, review, ai_review)
    payload["status"] = "ready_with_warnings" if any(row["status"] == "warning" for row in payload["steps"]) else "ready"
    # Finalise the guide before hashing. The mutable desktop handover manifest
    # itself is excluded from the academic inventory to avoid circular hashes.
    guide = root / "OPEN_FIRST.md"
    lines = [f"# {run.workflow_title}", "", f"Run: {run.run_id} | Preset: {run.preset_name} | Seed: {run.seed}", "", "## Outputs", ""]
    lines += [f"- {item['label']}: `{Path(item['path']).relative_to(root)}`" for item in payload["artifacts"]]
    lines += ["", "## Preparation", ""]
    lines += [f"- {row['title']}: {row['status']}. {row['detail']}" for row in payload["steps"]]
    lines += ["", "Reproducibility metadata, when enabled, is stored in ro-crate-metadata.json, academic_audit.json and ACADEMIC_BUNDLE.md in this folder.", "Check desktop_run_bundle.json for the final preparation status, including any packaging warnings.", "", "These outputs describe a research simulation. Data-contract compliance is not clinical validation."]
    guide.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact("Run guide", guide)
    save()
    if auto_prepare and not (cancelled and cancelled()):
        if progress:
            progress("package", 97, "Prepare reproducibility metadata and checksums")
        try:
            from iints.research.academic_bundle import build_academic_bundle

            package = build_academic_bundle(root, title=run.workflow_title)
            payload["academic_bundle"] = asdict(package)
            artifact("Reproducibility metadata", package.ro_crate_metadata)
            artifact("Reproducibility audit", package.audit_json)
            payload["steps"].append({"key": "package", "title": "Reproducibility package", "status": "ready", "detail": ""})
        except Exception as error:
            payload["steps"].append({"key": "package", "title": "Reproducibility package", "status": "warning", "detail": str(error)})
            payload["status"] = "ready_with_warnings"
    save()
    return payload


def load_run_bundle(csv_path: Path) -> dict[str, Any] | None:
    """Reopen preparation state only when it belongs to the selected CSV."""
    path = csv_path.parent / RUN_BUNDLE_FILENAME
    try:
        if not path.is_file() or path.stat().st_size > 2_000_000:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or Path(payload.get("results_csv", "")).resolve() != csv_path.resolve():
            return None
        if payload.get("source_sha256") != _source_digest(csv_path):
            return None
    except (ValueError, TypeError, OSError, UnicodeError):
        return None
    root = csv_path.parent.resolve()

    def safe_path(value: Any, *, directory: bool = False) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            candidate = Path(value).resolve()
            exists = candidate.is_dir() if directory else candidate.is_file()
            return str(candidate) if candidate.is_relative_to(root) and exists else None
        except (OSError, ValueError):
            return None

    # Local manifests are editable: validate every path exposed to native open
    # buttons, not just the general artifact list (also resolves symlinks).
    payload["output_dir"] = str(root)
    for key in ("report_pdf", "config_path", "manifest_path"):
        payload[key] = safe_path(payload.get(key))
    artifacts = []
    items = payload.get("artifacts")
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("label"), str):
            candidate = safe_path(item.get("path"))
            if candidate:
                artifacts.append({"label": item["label"], "path": candidate})
    payload["artifacts"] = artifacts
    steps = payload.get("steps")
    payload["steps"] = [row for row in steps if isinstance(row, dict) and isinstance(row.get("title"), str)] if isinstance(steps, list) else []
    for section, path_keys in {
        "mdmp": ("certificate_path", "report_path", "public_key_path"),
        "academic_bundle": ("ro_crate_metadata", "audit_json", "sources_json", "readme_md"),
    }.items():
        data = payload.get(section)
        if not isinstance(data, dict):
            payload.pop(section, None)
            continue
        data.pop("private_key_path", None)
        for key in path_keys:
            data[key] = safe_path(data.get(key))
        if "run_dir" in data:
            data["run_dir"] = str(root)
    if not isinstance(payload.get("ai_review"), dict):
        payload.pop("ai_review", None)
    return payload
