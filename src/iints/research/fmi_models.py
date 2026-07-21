"""FMI archive inspection and explicitly trusted FMPy execution.

An FMU can contain native binaries or source code. Static inspection is the
default and never loads those binaries. Simulation requires a separate,
explicit trust flag and should be performed only for reviewed FMUs.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import importlib
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable
import zipfile

from defusedxml import ElementTree as SafeElementTree

from .external_models_common import (
    local_name,
    read_local_file,
    safe_stem,
    sha256_bytes,
    timestamp_token,
    utc_now,
    write_json,
)


MAX_FMU_BYTES = 500 * 1024 * 1024
MAX_FMU_ENTRIES = 20_000
MAX_FMU_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MODEL_DESCRIPTION_BYTES = 25 * 1024 * 1024
FMI_RUN_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FMUModelSummary:
    model_path: Path
    sha256: str
    byte_size: int
    fmi_version: str
    model_name: str
    instantiation_token: str
    generation_tool: str
    generation_date_time: str
    interfaces: tuple[dict[str, str], ...]
    default_experiment: dict[str, str]
    variable_count: int
    variables: tuple[dict[str, str], ...]
    platforms: tuple[str, ...]
    archive_entry_count: int
    uncompressed_bytes: int
    has_native_binaries: bool
    has_sources: bool
    has_resources: bool
    warnings: tuple[str, ...]
    readiness_status: str


@dataclass(frozen=True)
class FMURunResult:
    run_dir: Path
    results_csv: Path
    inspection_json: Path
    manifest_json: Path
    review_md: Path
    engine: str
    engine_version: str
    row_count: int
    columns: tuple[str, ...]


def _safe_archive_members(archive: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], int]:
    members = archive.infolist()
    if len(members) > MAX_FMU_ENTRIES:
        raise ValueError(f"FMU contains more than {MAX_FMU_ENTRIES:,} archive entries.")
    total = 0
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"FMU contains an unsafe archive path: {member.filename!r}.")
        unix_mode = member.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise ValueError(f"FMU contains a symbolic link, which is not accepted: {member.filename!r}.")
        if member.flag_bits & 0x1:
            raise ValueError("Encrypted FMU entries are not accepted.")
        total += int(member.file_size)
        if total > MAX_FMU_UNCOMPRESSED_BYTES:
            raise ValueError("FMU uncompressed content exceeds the 2 GiB safety limit.")
        if member.file_size > 1024 * 1024 and member.compress_size > 0:
            ratio = member.file_size / member.compress_size
            if ratio > 1_000:
                raise ValueError(f"FMU entry has a suspicious compression ratio: {member.filename!r}.")
    return members, total


def _variable_rows(root: Any) -> tuple[int, tuple[dict[str, str], ...]]:
    rows: list[dict[str, str]] = []
    total = 0
    model_variables = next(
        (element for element in root if local_name(str(element.tag)) == "ModelVariables"),
        None,
    )
    if model_variables is None:
        return 0, ()
    scalar_types = {
        "Real",
        "Integer",
        "Boolean",
        "String",
        "Enumeration",
        "Float32",
        "Float64",
        "Int8",
        "UInt8",
        "Int16",
        "UInt16",
        "Int32",
        "UInt32",
        "Int64",
        "UInt64",
        "Binary",
        "Clock",
    }
    for element in model_variables:
        element_type = local_name(str(element.tag))
        if element_type == "ScalarVariable":
            type_element = next(
                (child for child in element if local_name(str(child.tag)) in scalar_types),
                None,
            )
            row = {
                "name": str(element.attrib.get("name") or ""),
                "value_reference": str(element.attrib.get("valueReference") or ""),
                "causality": str(element.attrib.get("causality") or ""),
                "variability": str(element.attrib.get("variability") or ""),
                "initial": str(element.attrib.get("initial") or ""),
                "type": local_name(str(type_element.tag)) if type_element is not None else "unknown",
                "unit": str(type_element.attrib.get("unit") or "") if type_element is not None else "",
                "start": str(type_element.attrib.get("start") or "") if type_element is not None else "",
            }
        elif element_type in scalar_types:
            row = {
                "name": str(element.attrib.get("name") or ""),
                "value_reference": str(element.attrib.get("valueReference") or ""),
                "causality": str(element.attrib.get("causality") or ""),
                "variability": str(element.attrib.get("variability") or ""),
                "initial": str(element.attrib.get("initial") or ""),
                "type": element_type,
                "unit": str(element.attrib.get("unit") or ""),
                "start": str(element.attrib.get("start") or ""),
            }
        else:
            continue
        total += 1
        if len(rows) < 1_000:
            rows.append(row)
    return total, tuple(rows)


def _all_variable_names(model_path: Path) -> set[str]:
    """Read declared variable names without extracting or loading the FMU."""

    resolved, _payload = read_local_file(
        model_path,
        label="FMU",
        suffixes={".fmu"},
        max_bytes=MAX_FMU_BYTES,
    )
    try:
        with zipfile.ZipFile(resolved, "r") as archive:
            members, _uncompressed_bytes = _safe_archive_members(archive)
            description = next(
                (member for member in members if member.filename == "modelDescription.xml"),
                None,
            )
            if description is None:
                raise ValueError("FMU does not contain modelDescription.xml at the archive root.")
            if description.file_size > MAX_MODEL_DESCRIPTION_BYTES:
                raise ValueError("FMU modelDescription.xml exceeds the 25 MiB safety limit.")
            description_payload = archive.read(description)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"FMU is not a valid ZIP archive: {exc}") from exc
    upper = description_payload.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("DTD and entity declarations are not accepted in modelDescription.xml.")
    try:
        root = SafeElementTree.fromstring(description_payload)
    except Exception as exc:
        raise ValueError(f"Could not parse FMU modelDescription.xml safely: {exc}") from exc

    model_variables = next(
        (element for element in root if local_name(str(element.tag)) == "ModelVariables"),
        None,
    )
    if model_variables is None:
        return set()
    supported_types = {
        "ScalarVariable",
        "Float32",
        "Float64",
        "Int8",
        "UInt8",
        "Int16",
        "UInt16",
        "Int32",
        "UInt32",
        "Int64",
        "UInt64",
        "Boolean",
        "String",
        "Binary",
        "Enumeration",
        "Clock",
    }
    return {
        str(element.attrib["name"])
        for element in model_variables
        if local_name(str(element.tag)) in supported_types and element.attrib.get("name")
    }


def inspect_fmu_model(model_path: Path) -> FMUModelSummary:
    """Inspect FMU metadata without extracting or loading executable code."""

    resolved, payload = read_local_file(
        model_path,
        label="FMU",
        suffixes={".fmu"},
        max_bytes=MAX_FMU_BYTES,
    )
    try:
        with zipfile.ZipFile(resolved, "r") as archive:
            members, uncompressed_bytes = _safe_archive_members(archive)
            by_name = {member.filename: member for member in members}
            description = by_name.get("modelDescription.xml")
            if description is None:
                raise ValueError("FMU does not contain modelDescription.xml at the archive root.")
            if description.file_size > MAX_MODEL_DESCRIPTION_BYTES:
                raise ValueError("FMU modelDescription.xml exceeds the 25 MiB safety limit.")
            description_payload = archive.read(description)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"FMU is not a valid ZIP archive: {exc}") from exc
    upper = description_payload.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("DTD and entity declarations are not accepted in modelDescription.xml.")
    try:
        root = SafeElementTree.fromstring(description_payload)
    except Exception as exc:
        raise ValueError(f"Could not parse FMU modelDescription.xml safely: {exc}") from exc
    if local_name(str(root.tag)) != "fmiModelDescription":
        raise ValueError("FMU modelDescription root element is not <fmiModelDescription>.")

    interface_names = {"ModelExchange", "CoSimulation", "ScheduledExecution"}
    interfaces = tuple(
        {
            "type": local_name(str(element.tag)),
            "model_identifier": str(element.attrib.get("modelIdentifier") or ""),
            "needs_execution_tool": str(element.attrib.get("needsExecutionTool") or ""),
        }
        for element in root
        if local_name(str(element.tag)) in interface_names
    )
    default_element = next(
        (element for element in root if local_name(str(element.tag)) == "DefaultExperiment"),
        None,
    )
    default_experiment = (
        {
            key: str(default_element.attrib[key])
            for key in ("startTime", "stopTime", "tolerance", "stepSize")
            if key in default_element.attrib
        }
        if default_element is not None
        else {}
    )
    variable_count, variables = _variable_rows(root)
    names = [member.filename for member in members]
    platforms = tuple(
        sorted(
            {
                PurePosixPath(name).parts[1]
                for name in names
                if len(PurePosixPath(name).parts) >= 3 and PurePosixPath(name).parts[0] == "binaries"
            }
        )
    )
    has_binaries = any(name.startswith("binaries/") and not name.endswith("/") for name in names)
    has_sources = any(name.startswith("sources/") and not name.endswith("/") for name in names)
    has_resources = any(name.startswith("resources/") and not name.endswith("/") for name in names)
    warnings: list[str] = []
    if not interfaces:
        warnings.append("No FMI Model Exchange, Co-Simulation, or Scheduled Execution interface was declared.")
    if has_binaries:
        warnings.append(
            "The FMU contains native binaries. FMI does not sandbox operating-system access; execute only if trusted."
        )
    elif has_sources:
        warnings.append("The FMU contains source code that an execution tool may compile or load.")
    if not platforms and has_binaries:
        warnings.append("Native binaries were found but no standard platform directory could be inferred.")
    if variable_count == 0:
        warnings.append("No model variables were found.")
    missing_units = sum(1 for variable in variables if variable["type"] in {"Real", "Float32", "Float64"} and not variable["unit"])
    if missing_units:
        warnings.append(
            f"At least {missing_units} previewed floating-point variables do not declare units."
        )
    readiness = "inspectable_untrusted" if interfaces else "needs_review"
    return FMUModelSummary(
        model_path=resolved,
        sha256=sha256_bytes(payload),
        byte_size=len(payload),
        fmi_version=str(root.attrib.get("fmiVersion") or "unknown"),
        model_name=str(root.attrib.get("modelName") or resolved.stem),
        instantiation_token=str(root.attrib.get("instantiationToken") or root.attrib.get("guid") or ""),
        generation_tool=str(root.attrib.get("generationTool") or ""),
        generation_date_time=str(root.attrib.get("generationDateAndTime") or ""),
        interfaces=interfaces,
        default_experiment=default_experiment,
        variable_count=variable_count,
        variables=variables,
        platforms=platforms,
        archive_entry_count=len(members),
        uncompressed_bytes=uncompressed_bytes,
        has_native_binaries=has_binaries,
        has_sources=has_sources,
        has_resources=has_resources,
        warnings=tuple(warnings),
        readiness_status=readiness,
    )


def fmu_summary_payload(summary: FMUModelSummary, *, include_local_path: bool = True) -> dict[str, Any]:
    payload = asdict(summary)
    payload["model_path"] = str(summary.model_path) if include_local_path else summary.model_path.name
    return payload


def _load_fmpy() -> Any:
    try:
        return importlib.import_module("fmpy")
    except ImportError as exc:
        raise RuntimeError(
            "FMPy is required to execute an FMU. Install the optional engine with "
            "`python -m pip install 'iints-sdk-python35[fmi]'`. Static FMU inspection needs no FMPy."
        ) from exc


def fmpy_status() -> dict[str, Any]:
    try:
        module = _load_fmpy()
    except RuntimeError as exc:
        return {"available": False, "engine": "FMPy", "version": None, "message": str(exc)}
    return {
        "available": True,
        "engine": "FMPy",
        "version": str(getattr(module, "__version__", "unknown")),
        "message": "FMPy is available. Native FMU execution still requires explicit trust.",
    }


def _normalise_outputs(summary: FMUModelSummary, requested: Iterable[str]) -> tuple[str, ...]:
    values = [value.strip() for value in requested if value and value.strip()]
    if len(values) > 256:
        raise ValueError("At most 256 FMU output variables may be selected.")
    if any(len(value) > 256 or any(ord(character) < 32 for character in value) for value in values):
        raise ValueError("FMU output names must be short, printable single-line values.")
    declared = {str(variable["name"]) for variable in summary.variables if variable.get("name")}
    unknown = sorted(set(values) - declared)
    if unknown and summary.variable_count > len(summary.variables):
        # The public summary intentionally caps its variable preview. Re-read only
        # the XML names so large, reviewed FMUs remain selectable without loading
        # native binaries or inflating every inspection artifact.
        declared = _all_variable_names(summary.model_path)
        unknown = sorted(set(values) - declared)
    if unknown:
        raise ValueError(f"Unknown FMU variables: {', '.join(unknown[:10])}.")
    return tuple(dict.fromkeys(values))


def _fmpy_result_rows(result: Any) -> tuple[list[str], list[list[float]]]:
    dtype = getattr(result, "dtype", None)
    names = list(getattr(dtype, "names", ()) or ())
    if not names:
        raise RuntimeError("FMPy returned a result without named columns.")
    rows: list[list[float]] = []
    for record in result:
        row = [float(record[name]) for name in names]
        if any(value != value or value in {float("inf"), float("-inf")} for value in row):
            raise RuntimeError("FMPy returned a non-finite simulation value.")
        rows.append(row)
    if not rows:
        raise RuntimeError("FMPy returned no simulation rows.")
    return [str(name) for name in names], rows


def run_fmu_model(
    model_path: Path,
    output_dir: Path,
    *,
    start: float,
    end: float,
    output_interval: float,
    variables: Iterable[str] = (),
    timeout_seconds: int = 300,
    allow_native_execution: bool = False,
    _engine_module: Any | None = None,
) -> FMURunResult:
    """Execute a reviewed FMU with FMPy after explicit native-code consent."""

    if not allow_native_execution:
        raise PermissionError(
            "FMU execution can load native code. Pass allow_native_execution=True only for a trusted FMU."
        )
    if not all(value == value and abs(value) != float("inf") for value in (start, end, output_interval)):
        raise ValueError("FMU timing values must be finite.")
    if end <= start or output_interval <= 0:
        raise ValueError("end must be greater than start and output_interval must be positive.")
    expected_rows = int((end - start) / output_interval) + 1
    if expected_rows > 1_000_001:
        raise ValueError("Requested FMU run exceeds the 1,000,001-row safety limit.")
    if timeout_seconds < 1 or timeout_seconds > 24 * 60 * 60:
        raise ValueError("timeout_seconds must be between 1 and 86,400.")
    summary = inspect_fmu_model(model_path)
    outputs = _normalise_outputs(summary, variables)
    module = _engine_module or _load_fmpy()
    engine_version = str(getattr(module, "__version__", "unknown"))
    simulate = getattr(module, "simulate_fmu", None)
    if not callable(simulate):
        raise RuntimeError("Installed FMPy module does not expose simulate_fmu().")
    try:
        raw_result = simulate(
            str(summary.model_path),
            validate=True,
            start_time=float(start),
            stop_time=float(end),
            output_interval=float(output_interval),
            output=list(outputs) if outputs else None,
            timeout=float(timeout_seconds),
        )
    except Exception as exc:
        raise RuntimeError(f"FMPy could not execute the trusted FMU: {exc}") from exc
    columns, rows = _fmpy_result_rows(raw_result)

    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"fmu_{safe_stem(summary.model_path.stem)}_{timestamp_token()}_{summary.sha256[:8]}"
    run_dir.mkdir(parents=False, exist_ok=False)
    results_path = run_dir / "fmu_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    inspection_path = run_dir / "fmu_model_summary.json"
    write_json(inspection_path, fmu_summary_payload(summary, include_local_path=False))
    review_path = run_dir / "FMU_REVIEW.md"
    review_path.write_text(
        "\n".join(
            [
                "# IINTS FMI / FMPy Device-Physics Run",
                "",
                "This run executed a user-trusted FMU. FMUs may contain native code and are not sandboxed by FMI.",
                "",
                f"- FMU: `{summary.model_path.name}`",
                f"- SHA-256: `{summary.sha256}`",
                f"- FMI version: `{summary.fmi_version}`",
                f"- FMPy version: `{engine_version}`",
                f"- Rows: `{len(rows)}`",
                "",
                "## Required review",
                "",
                "1. Verify the FMU publisher, hash, license, platform binary, and source-model version.",
                "2. Confirm units and interfaces for pump flow, motor state, pressure, occlusion, and sensors.",
                "3. Validate the FMU independently against bench measurements before using it as a reference.",
                "4. Keep device-physics outputs separate from patient physiology unless a reviewed coupling is defined.",
                "5. Never use this output for real pump control or treatment.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = run_dir / "fmi_run_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": FMI_RUN_SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "research_only": True,
            "medical_device": False,
            "native_code_execution": True,
            "explicit_user_trust_required": True,
            "model": {
                "file_name": summary.model_path.name,
                "sha256": summary.sha256,
                "fmi_version": summary.fmi_version,
                "interfaces": list(summary.interfaces),
                "platforms": list(summary.platforms),
            },
            "engine": {"name": "FMPy", "version": engine_version},
            "simulation": {
                "start": float(start),
                "end": float(end),
                "output_interval": float(output_interval),
                "requested_variables": list(outputs),
                "columns": columns,
                "timeout_seconds": timeout_seconds,
            },
            "outputs": {
                "results_csv": results_path.name,
                "inspection": inspection_path.name,
                "review": review_path.name,
                "row_count": len(rows),
            },
            "limitations": [
                "FMI does not sandbox native code or operating-system access.",
                "Execution success does not validate pump, sensor, motor, or fluid physics.",
                "No FMU output is coupled to IINTS physiology automatically.",
                "This output must not control a real device or treatment.",
            ],
        },
    )
    return FMURunResult(
        run_dir=run_dir,
        results_csv=results_path,
        inspection_json=inspection_path,
        manifest_json=manifest_path,
        review_md=review_path,
        engine="FMPy",
        engine_version=engine_version,
        row_count=len(rows),
        columns=tuple(columns),
    )
