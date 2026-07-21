"""Safe COPASI model inspection and explicit configured-task execution.

IINTS does not generate or silently tune COPASI parameter-estimation tasks.
Researchers configure those tasks in COPASI, inspect the resulting ``.cps``
file here, and then explicitly opt in to batch execution with ``CopasiSE``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

from defusedxml import ElementTree as SafeElementTree

from .external_models_common import (
    find_executable,
    local_name,
    normalised_bool,
    read_local_file,
    run_external_command,
    safe_stem,
    sha256_bytes,
    timestamp_token,
    utc_now,
    write_json,
)


MAX_COPASI_BYTES = 25 * 1024 * 1024
COPASI_RUN_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class COPASIModelSummary:
    """Static summary of a COPASI-ML document."""

    model_path: Path
    sha256: str
    byte_size: int
    model_name: str
    copasi_version_major: str
    copasi_version_minor: str
    copasi_version_build: str
    tasks: tuple[dict[str, Any], ...]
    scheduled_task_count: int
    sensitivity_task_count: int
    parameter_estimation_task_count: int
    external_file_references: tuple[str, ...]
    warnings: tuple[str, ...]
    readiness_status: str


@dataclass(frozen=True)
class COPASIRunResult:
    run_dir: Path
    copied_model: Path
    output_model: Path
    report_txt: Path
    stdout_log: Path
    stderr_log: Path
    manifest_json: Path
    inspection_json: Path
    review_md: Path
    engine_path: Path
    selected_task: str | None
    return_code: int


def _copasi_task_kind(raw: str) -> str:
    lowered = raw.strip().lower().replace(" ", "")
    if "sensitiv" in lowered:
        return "sensitivity_analysis"
    if "parameter" in lowered and ("fit" in lowered or "estimat" in lowered):
        return "parameter_estimation"
    if lowered in {"parameterfitting", "parameterestimation"}:
        return "parameter_estimation"
    return raw.strip() or "unknown"


def _first_descendant(element: Any, name: str) -> Any | None:
    return next((child for child in element.iter() if local_name(str(child.tag)) == name), None)


def _task_rows(root: Any) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for task in root.iter():
        if local_name(str(task.tag)) != "Task":
            continue
        raw_type = str(task.attrib.get("type") or "")
        method = _first_descendant(task, "Method")
        report = _first_descendant(task, "Report")
        rows.append(
            {
                "name": str(task.attrib.get("name") or ""),
                "key": str(task.attrib.get("key") or ""),
                "raw_type": raw_type,
                "kind": _copasi_task_kind(raw_type),
                "scheduled": normalised_bool(task.attrib.get("scheduled")) is True,
                "update_model": normalised_bool(task.attrib.get("updateModel")) is True,
                "method_name": str(method.attrib.get("name") or "") if method is not None else "",
                "method_type": str(method.attrib.get("type") or "") if method is not None else "",
                "report_reference": str(report.attrib.get("reference") or "") if report is not None else "",
                "report_target": str(report.attrib.get("target") or "") if report is not None else "",
            }
        )
    return tuple(rows)


def _external_file_references(root: Any) -> tuple[str, ...]:
    references: set[str] = set()
    for element in root.iter():
        attributes = {str(key).lower(): str(value) for key, value in element.attrib.items()}
        parameter_type = attributes.get("type", "").lower()
        parameter_name = attributes.get("name", "").lower()
        if parameter_type == "file" or "file name" in parameter_name or "filename" in parameter_name:
            value = attributes.get("value", "").strip()
            if value:
                references.add(value)
        if local_name(str(element.tag)) == "Report":
            target = attributes.get("target", "").strip()
            if target:
                references.add(target)
    return tuple(sorted(references))


def inspect_copasi_model(model_path: Path) -> COPASIModelSummary:
    """Inspect COPASI tasks without running sensitivity or fitting methods."""

    resolved, payload = read_local_file(
        model_path,
        label="COPASI model",
        suffixes={".cps"},
        max_bytes=MAX_COPASI_BYTES,
        reject_xml_entities=True,
    )
    try:
        root = SafeElementTree.fromstring(payload)
    except Exception as exc:
        raise ValueError(f"Could not parse COPASI-ML safely: {exc}") from exc
    if local_name(str(root.tag)).lower() != "copasi":
        raise ValueError("COPASI model root element is not <COPASI>.")

    model = next((item for item in root.iter() if local_name(str(item.tag)) == "Model"), None)
    if model is None:
        raise ValueError("COPASI document does not contain a <Model> element.")
    tasks = _task_rows(root)
    scheduled_count = sum(1 for task in tasks if task["scheduled"])
    sensitivity_count = sum(1 for task in tasks if task["kind"] == "sensitivity_analysis")
    fitting_count = sum(1 for task in tasks if task["kind"] == "parameter_estimation")
    references = _external_file_references(root)

    warnings: list[str] = []
    if not tasks:
        warnings.append("No COPASI tasks were found.")
    if scheduled_count == 0:
        warnings.append("No task is scheduled; CopasiSE would not perform an analysis by default.")
    if sensitivity_count == 0:
        warnings.append("No sensitivity-analysis task is configured.")
    if fitting_count == 0:
        warnings.append("No parameter-estimation task is configured.")
    updating = [str(task["name"] or task["raw_type"]) for task in tasks if task["scheduled"] and task["update_model"]]
    if updating:
        warnings.append(
            "Scheduled tasks update model state or parameters: " + ", ".join(updating[:10]) + "."
        )
    if references:
        warnings.append(
            "The model references external data/report files; verify paths and dataset provenance before execution."
        )
    readiness = "ready_for_explicit_run" if scheduled_count > 0 else "needs_configuration"
    return COPASIModelSummary(
        model_path=resolved,
        sha256=sha256_bytes(payload),
        byte_size=len(payload),
        model_name=str(model.attrib.get("name") or model.attrib.get("key") or resolved.stem),
        copasi_version_major=str(root.attrib.get("versionMajor") or ""),
        copasi_version_minor=str(root.attrib.get("versionMinor") or ""),
        copasi_version_build=str(root.attrib.get("versionDevel") or root.attrib.get("versionBuild") or ""),
        tasks=tasks,
        scheduled_task_count=scheduled_count,
        sensitivity_task_count=sensitivity_count,
        parameter_estimation_task_count=fitting_count,
        external_file_references=references,
        warnings=tuple(warnings),
        readiness_status=readiness,
    )


def copasi_summary_payload(summary: COPASIModelSummary, *, include_local_path: bool = True) -> dict[str, Any]:
    payload = asdict(summary)
    payload["model_path"] = str(summary.model_path) if include_local_path else summary.model_path.name
    return payload


def _copasi_executable(configured: Path | None = None) -> Path | None:
    if configured is not None:
        resolved = configured.expanduser().resolve()
        return resolved if resolved.is_file() else None
    return find_executable(
        environment_variable="COPASISE_EXECUTABLE",
        names=("CopasiSE", "copasise"),
        common_paths=(
            Path("/Applications/COPASI.app/Contents/MacOS/CopasiSE"),
            Path("/usr/local/bin/CopasiSE"),
            Path("/opt/COPASI/bin/CopasiSE"),
        ),
    )


def copasi_status(*, executable: Path | None = None) -> dict[str, Any]:
    """Return CopasiSE availability without loading or running a model."""

    resolved = _copasi_executable(executable)
    if resolved is None:
        return {
            "available": False,
            "engine": "CopasiSE",
            "path": None,
            "version_hint": None,
            "message": "CopasiSE was not found. Set COPASISE_EXECUTABLE or install COPASI.",
        }
    version_hint = "unknown"
    try:
        probe = run_external_command([str(resolved), "--license"], timeout_seconds=10)
        text = (probe.stdout or probe.stderr).strip()
        version_hint = text.splitlines()[0][:200] if text else "installed (version not reported)"
    except Exception:
        version_hint = "installed (version probe unavailable)"
    return {
        "available": True,
        "engine": "CopasiSE",
        "path": str(resolved),
        "version_hint": version_hint,
        "message": "CopasiSE is available for explicitly scheduled local COPASI tasks.",
    }


def run_copasi_model(
    model_path: Path,
    output_dir: Path,
    *,
    scheduled_task: str | None = None,
    timeout_seconds: int = 900,
    allow_external_execution: bool = False,
    executable: Path | None = None,
) -> COPASIRunResult:
    """Execute one configured COPASI task through CopasiSE.

    No task or objective is generated by IINTS. The source ``.cps`` remains
    unchanged and is copied into the evidence directory after inspection.
    """

    if not allow_external_execution:
        raise PermissionError(
            "COPASI execution is opt-in. Pass allow_external_execution=True after reviewing the configured tasks."
        )
    if timeout_seconds < 1 or timeout_seconds > 24 * 60 * 60:
        raise ValueError("timeout_seconds must be between 1 and 86,400.")
    summary = inspect_copasi_model(model_path)
    task_names = {str(task["name"]) for task in summary.tasks if task.get("name")}
    if scheduled_task is not None:
        scheduled_task = scheduled_task.strip()
        if not scheduled_task or len(scheduled_task) > 256 or any(ord(char) < 32 for char in scheduled_task):
            raise ValueError("scheduled_task must be a short printable COPASI task name.")
        if scheduled_task not in task_names:
            raise ValueError(f"Unknown COPASI task '{scheduled_task}'.")
    elif summary.scheduled_task_count == 0:
        raise ValueError("No task is scheduled in the COPASI model; select and configure a task in COPASI first.")

    engine = _copasi_executable(executable)
    if engine is None:
        raise RuntimeError("CopasiSE was not found. Set COPASISE_EXECUTABLE or install COPASI.")

    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / (
        f"copasi_{safe_stem(summary.model_path.stem)}_{timestamp_token()}_{summary.sha256[:8]}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    copied_model = run_dir / summary.model_path.name
    shutil.copy2(summary.model_path, copied_model)
    output_model = run_dir / "copasi_fitted_or_updated.cps"
    report_path = run_dir / "copasi_report.txt"
    stdout_path = run_dir / "copasi_stdout.log"
    stderr_path = run_dir / "copasi_stderr.log"
    config_dir = run_dir / "config"
    temp_dir = run_dir / "tmp"
    home_dir = run_dir / "home"
    for directory in (config_dir, temp_dir, home_dir):
        directory.mkdir()

    command = [
        str(engine),
        "--nologo",
        "--maxTime",
        str(timeout_seconds),
        "--configdir",
        str(config_dir),
        "--tmp",
        str(temp_dir),
        "--home",
        str(home_dir),
        "--save",
        str(output_model),
        "--report-file",
        str(report_path),
    ]
    if scheduled_task is not None:
        command.extend(["--scheduled-task", scheduled_task])
    command.append(str(summary.model_path))
    environment = os.environ.copy()
    environment.update({"HOME": str(home_dir), "TMPDIR": str(temp_dir)})
    result = run_external_command(
        command,
        cwd=summary.model_path.parent,
        timeout_seconds=timeout_seconds + 30,
        environment=environment,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    inspection_path = run_dir / "copasi_model_summary.json"
    write_json(inspection_path, copasi_summary_payload(summary, include_local_path=False))
    review_path = run_dir / "COPASI_REVIEW.md"
    review_path.write_text(
        "\n".join(
            [
                "# IINTS COPASI Analysis Run",
                "",
                "This is an independent, explicitly configured COPASI analysis. It does not calibrate the IINTS "
                "patient simulator automatically.",
                "",
                f"- Source model: `{summary.model_path.name}`",
                f"- SHA-256: `{summary.sha256}`",
                f"- Selected task override: `{scheduled_task or 'model-scheduled task'}`",
                f"- CopasiSE: `{engine}`",
                f"- Return code: `{result.returncode}`",
                "",
                "## Required review",
                "",
                "1. Confirm experimental data provenance, units, residual definition, and weighting.",
                "2. Report parameter bounds, optimizer, random seeds, convergence criteria, and failed starts.",
                "3. Treat local sensitivity as operating-point dependent; do not equate it with identifiability.",
                "4. Use profile likelihood or equivalent diagnostics before claiming parameter identifiability.",
                "5. Never use fitted values for treatment or automatic dosing.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = run_dir / "copasi_run_manifest.json"
    manifest = {
        "schema_version": COPASI_RUN_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "research_only": True,
        "medical_device": False,
        "engine": {"name": "CopasiSE", "path": str(engine), "return_code": result.returncode},
        "model": {"file_name": summary.model_path.name, "sha256": summary.sha256},
        "task": {
            "override": scheduled_task,
            "configured_tasks": list(summary.tasks),
            "scheduled_task_count": summary.scheduled_task_count,
        },
        "execution": {"timeout_seconds": timeout_seconds, "argv_without_executable": command[1:]},
        "outputs": {
            "copied_model": copied_model.name,
            "output_model": output_model.name if output_model.exists() else None,
            "report": report_path.name if report_path.exists() else None,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
            "inspection": inspection_path.name,
            "review": review_path.name,
        },
        "limitations": [
            "A successful solver run is not evidence that parameters are structurally or practically identifiable.",
            "IINTS does not automatically import fitted parameters into its patient models.",
            "External data paths and report definitions remain the researcher's responsibility.",
            "This output must not be used for treatment decisions.",
        ],
    }
    write_json(manifest_path, manifest)
    if result.returncode != 0:
        raise RuntimeError(
            f"CopasiSE exited with code {result.returncode}. Evidence and logs remain in {run_dir}."
        )
    return COPASIRunResult(
        run_dir=run_dir,
        copied_model=copied_model,
        output_model=output_model,
        report_txt=report_path,
        stdout_log=stdout_path,
        stderr_log=stderr_path,
        manifest_json=manifest_path,
        inspection_json=inspection_path,
        review_md=review_path,
        engine_path=engine,
        selected_task=scheduled_task,
        return_code=result.returncode,
    )
