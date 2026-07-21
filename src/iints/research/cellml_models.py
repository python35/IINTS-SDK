"""Static CellML inspection and independent OpenCOR validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as SafeElementTree

from .external_models_common import (
    find_executable,
    local_name,
    namespace,
    read_local_file,
    run_external_command,
    safe_stem,
    sha256_bytes,
    timestamp_token,
    utc_now,
    write_json,
)


MAX_CELLML_BYTES = 25 * 1024 * 1024
CELLML_VALIDATION_SCHEMA_VERSION = "1.0"
RECOGNISED_CELLML_NAMESPACES = {
    "http://www.cellml.org/cellml/1.0#",
    "http://www.cellml.org/cellml/1.1#",
    "http://www.cellml.org/cellml/2.0#",
}


@dataclass(frozen=True)
class CellMLModelSummary:
    model_path: Path
    sha256: str
    byte_size: int
    namespace: str
    cellml_version: str
    model_name: str
    component_count: int
    variable_count: int
    units_count: int
    connection_count: int
    math_block_count: int
    import_count: int
    components: tuple[dict[str, Any], ...]
    imports: tuple[str, ...]
    warnings: tuple[str, ...]
    readiness_status: str
    opencor_validation_performed: bool = False


@dataclass(frozen=True)
class CellMLValidationResult:
    run_dir: Path
    copied_model: Path
    inspection_json: Path
    validation_log: Path
    manifest_json: Path
    review_md: Path
    engine_path: Path
    valid: bool
    return_code: int


def _cellml_version(model_namespace: str) -> str:
    if "/1.0" in model_namespace:
        return "1.0"
    if "/1.1" in model_namespace:
        return "1.1"
    if "/2.0" in model_namespace:
        return "2.0"
    return "unknown"


def _attribute_by_local_name(element: Any, wanted: str) -> str:
    for key, value in element.attrib.items():
        if local_name(str(key)) == wanted:
            return str(value)
    return ""


def _component_rows(root: Any) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for component in root.iter():
        if local_name(str(component.tag)) != "component":
            continue
        variables: list[dict[str, str]] = []
        for variable in component:
            if local_name(str(variable.tag)) != "variable":
                continue
            variables.append(
                {
                    key: str(value)
                    for key in (
                        "name",
                        "units",
                        "initial_value",
                        "interface",
                        "public_interface",
                        "private_interface",
                    )
                    if (value := variable.attrib.get(key)) is not None
                }
            )
        rows.append({"name": str(component.attrib.get("name") or ""), "variables": variables})
    return tuple(rows)


def inspect_cellml_model(model_path: Path) -> CellMLModelSummary:
    """Inspect one local CellML document without resolving imports or equations."""

    resolved, payload = read_local_file(
        model_path,
        label="CellML model",
        suffixes={".cellml", ".xml"},
        max_bytes=MAX_CELLML_BYTES,
        reject_xml_entities=True,
    )
    try:
        root = SafeElementTree.fromstring(payload)
    except Exception as exc:
        raise ValueError(f"Could not parse CellML XML safely: {exc}") from exc
    if local_name(str(root.tag)) != "model":
        raise ValueError("CellML document root element is not <model>.")
    model_namespace = namespace(str(root.tag))
    elements = list(root.iter())
    components = _component_rows(root)
    imports = tuple(
        sorted(
            {
                href
                for element in elements
                if local_name(str(element.tag)) == "import"
                and (href := _attribute_by_local_name(element, "href")).strip()
            }
        )
    )
    counts = {
        name: sum(1 for element in elements if local_name(str(element.tag)) == name)
        for name in ("component", "variable", "units", "connection", "math", "import")
    }
    warnings: list[str] = []
    if model_namespace not in RECOGNISED_CELLML_NAMESPACES:
        warnings.append("The model does not use a recognised CellML 1.0, 1.1, or 2.0 namespace.")
    if counts["component"] == 0:
        warnings.append("No CellML components were found.")
    if counts["variable"] == 0:
        warnings.append("No CellML variables were found.")
    if counts["math"] == 0:
        warnings.append("No MathML blocks were found; the model may contain metadata only.")
    if imports:
        warnings.append(
            "The model contains imports. Static inspection does not resolve them; pin and review each dependency."
        )
    missing_units = sum(
        1
        for component in components
        for variable in component["variables"]
        if not variable.get("units")
    )
    if missing_units:
        warnings.append(f"{missing_units} variables do not declare units.")
    readiness = (
        "inspectable"
        if model_namespace in RECOGNISED_CELLML_NAMESPACES and counts["component"] > 0
        else "needs_review"
    )
    return CellMLModelSummary(
        model_path=resolved,
        sha256=sha256_bytes(payload),
        byte_size=len(payload),
        namespace=model_namespace,
        cellml_version=_cellml_version(model_namespace),
        model_name=str(root.attrib.get("name") or resolved.stem),
        component_count=counts["component"],
        variable_count=counts["variable"],
        units_count=counts["units"],
        connection_count=counts["connection"],
        math_block_count=counts["math"],
        import_count=counts["import"],
        components=components,
        imports=imports,
        warnings=tuple(warnings),
        readiness_status=readiness,
    )


def cellml_summary_payload(summary: CellMLModelSummary, *, include_local_path: bool = True) -> dict[str, Any]:
    payload = asdict(summary)
    payload["model_path"] = str(summary.model_path) if include_local_path else summary.model_path.name
    return payload


def _opencor_executable(configured: Path | None = None) -> Path | None:
    if configured is not None:
        resolved = configured.expanduser().resolve()
        return resolved if resolved.is_file() else None
    return find_executable(
        environment_variable="OPENCOR_EXECUTABLE",
        names=("OpenCOR", "opencor"),
        common_paths=(
            Path("/Applications/OpenCOR.app/Contents/MacOS/OpenCOR"),
            Path("/usr/local/bin/OpenCOR"),
            Path("/opt/OpenCOR/OpenCOR"),
        ),
    )


def opencor_status(*, executable: Path | None = None) -> dict[str, Any]:
    resolved = _opencor_executable(executable)
    if resolved is None:
        return {
            "available": False,
            "engine": "OpenCOR",
            "path": None,
            "version": None,
            "message": "OpenCOR was not found. Set OPENCOR_EXECUTABLE or install OpenCOR.",
        }
    version = "unknown"
    try:
        result = run_external_command([str(resolved), "--version"], timeout_seconds=10)
        version = (result.stdout or result.stderr).strip().splitlines()[0][:200] or "unknown"
    except Exception:
        version = "installed (version probe unavailable)"
    return {
        "available": True,
        "engine": "OpenCOR",
        "path": str(resolved),
        "version": version,
        "message": "OpenCOR is available for independent CellML validation.",
    }


def validate_cellml_model(
    model_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: int = 120,
    executable: Path | None = None,
) -> CellMLValidationResult:
    """Validate a local CellML document with OpenCOR's CellMLTools plugin."""

    summary = inspect_cellml_model(model_path)
    engine = _opencor_executable(executable)
    if engine is None:
        raise RuntimeError("OpenCOR was not found. Set OPENCOR_EXECUTABLE or install OpenCOR.")
    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / (
        f"cellml_{safe_stem(summary.model_path.stem)}_{timestamp_token()}_{summary.sha256[:8]}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    copied_model = run_dir / summary.model_path.name
    copied_model.write_bytes(summary.model_path.read_bytes())
    result = run_external_command(
        [str(engine), "-c", "CellMLTools::validate", str(summary.model_path)],
        cwd=summary.model_path.parent,
        timeout_seconds=timeout_seconds,
    )
    validation_log = run_dir / "opencor_validation.log"
    validation_log.write_text(
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        encoding="utf-8",
    )
    valid = result.returncode == 0
    inspection_path = run_dir / "cellml_model_summary.json"
    inspection_payload = cellml_summary_payload(summary, include_local_path=False)
    inspection_payload["opencor_validation_performed"] = True
    write_json(inspection_path, inspection_payload)
    review_path = run_dir / "CELLML_REVIEW.md"
    review_path.write_text(
        "\n".join(
            [
                "# IINTS CellML / OpenCOR Validation",
                "",
                "The model was validated independently with OpenCOR. Validation checks CellML correctness; it does "
                "not establish biological validity, parameter relevance, or clinical suitability.",
                "",
                f"- Model: `{summary.model_path.name}`",
                f"- SHA-256: `{summary.sha256}`",
                f"- CellML version: `{summary.cellml_version}`",
                f"- OpenCOR result: `{'valid' if valid else 'invalid or warnings/errors reported'}`",
                "",
                "## Required review",
                "",
                "1. Resolve and pin every imported CellML dependency.",
                "2. Confirm units, initial conditions, solver settings, and population assumptions.",
                "3. Compare outputs as an independent reference; never silently map them into IINTS parameters.",
                "4. Do not use this workflow for diagnosis, dosing, or treatment.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = run_dir / "cellml_validation_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": CELLML_VALIDATION_SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "research_only": True,
            "medical_device": False,
            "engine": {"name": "OpenCOR", "path": str(engine), "return_code": result.returncode},
            "model": {
                "file_name": summary.model_path.name,
                "sha256": summary.sha256,
                "cellml_version": summary.cellml_version,
                "imports": list(summary.imports),
            },
            "validation": {"command": "CellMLTools::validate", "valid": valid},
            "outputs": {
                "copied_model": copied_model.name,
                "inspection": inspection_path.name,
                "validation_log": validation_log.name,
                "review": review_path.name,
            },
            "limitations": [
                "OpenCOR validation does not prove biological correctness.",
                "Imported models are not resolved or trusted automatically.",
                "CellML outputs never calibrate IINTS automatically.",
                "This output must not be used for treatment decisions.",
            ],
        },
    )
    return CellMLValidationResult(
        run_dir=run_dir,
        copied_model=copied_model,
        inspection_json=inspection_path,
        validation_log=validation_log,
        manifest_json=manifest_path,
        review_md=review_path,
        engine_path=engine,
        valid=valid,
        return_code=result.returncode,
    )
