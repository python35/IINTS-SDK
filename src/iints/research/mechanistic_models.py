"""Safe inspection and optional execution of external SBML reference models.

This module creates an explicit boundary between IINTS patient simulation and
third-party systems-biology models.  A reference model is never allowed to
silently calibrate glucose, insulin, or dosing parameters.  Instead, the SDK
records the model hash, engine, selections, units metadata, and simulation
settings so that researchers can compare mechanistic assumptions explicitly.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import mimetypes
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urlsplit

from defusedxml import ElementTree as SafeElementTree


MAX_SBML_BYTES = 25 * 1024 * 1024
MECHANISTIC_RUN_SCHEMA_VERSION = "1.0"
SUPPORTED_SBML_SUFFIXES = {".xml", ".sbml"}


@dataclass(frozen=True)
class SBMLModelSummary:
    """Static, non-executing summary of one local SBML document."""

    model_path: Path
    sha256: str
    byte_size: int
    sbml_level: int | None
    sbml_version: int | None
    namespace: str
    model_id: str
    model_name: str
    counts: dict[str, int]
    model_units: dict[str, str]
    species: tuple[dict[str, Any], ...]
    parameters: tuple[dict[str, Any], ...]
    compartments: tuple[dict[str, Any], ...]
    package_prefixes: tuple[str, ...]
    warnings: tuple[str, ...]
    readiness_status: str
    schema_validation_performed: bool = False


@dataclass(frozen=True)
class MechanisticRunResult:
    """Files produced by an isolated libRoadRunner SBML simulation."""

    run_dir: Path
    results_csv: Path
    manifest_json: Path
    model_summary_json: Path
    report_md: Path
    engine: str
    engine_version: str
    row_count: int
    selections: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.index("}")]
    return ""


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalised_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    return None


def _element_rows(elements: list[Any], element_name: str, fields: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for element in elements:
        if _local_name(str(element.tag)) != element_name:
            continue
        row: dict[str, Any] = {}
        for field in fields:
            raw = element.attrib.get(field)
            if raw is None:
                continue
            if field in {"constant", "boundaryCondition", "hasOnlySubstanceUnits", "reversible"}:
                row[field] = _normalised_bool(raw)
            else:
                row[field] = raw
        rows.append(row)
    return tuple(rows)


def _model_list_rows(
    model: Any,
    list_name: str,
    element_name: str,
    fields: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    for child in list(model):
        if _local_name(str(child.tag)) == list_name:
            return _element_rows(list(child), element_name, fields)
    return ()


def _package_prefixes(xml_text: str) -> tuple[str, ...]:
    prefixes = {
        match.group(1)
        for match in re.finditer(
            r"xmlns:([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*[\"']https?://www\.sbml\.org/sbml/level3/version1/([^\"']+)[\"']",
            xml_text,
        )
    }
    return tuple(sorted(prefixes))


def _read_sbml_bytes(model_path: Path) -> tuple[Path, bytes]:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SBML model not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_SBML_SUFFIXES:
        raise ValueError("SBML model must use a .xml or .sbml extension.")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError("SBML model is empty.")
    if size > MAX_SBML_BYTES:
        raise ValueError(f"SBML model exceeds the {MAX_SBML_BYTES // (1024 * 1024)} MiB safety limit.")
    payload = resolved.read_bytes()
    upper = payload.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("DTD and entity declarations are not accepted in local SBML models.")
    return resolved, payload


def inspect_sbml_model(model_path: Path) -> SBMLModelSummary:
    """Inspect one local SBML document without executing model equations.

    The parser is entity-safe and size-limited.  This is a structural audit,
    not full SBML schema validation or evidence that the model is biologically
    valid for diabetes research.
    """

    resolved, payload = _read_sbml_bytes(model_path)
    try:
        root = SafeElementTree.fromstring(payload)
    except Exception as exc:
        raise ValueError(f"Could not parse SBML XML safely: {exc}") from exc

    if _local_name(str(root.tag)) != "sbml":
        raise ValueError("XML root element is not <sbml>.")
    namespace = _namespace(str(root.tag))
    elements = list(root.iter())
    model = next((element for element in elements if _local_name(str(element.tag)) == "model"), None)
    if model is None:
        raise ValueError("SBML document does not contain a <model> element.")

    count_names = {
        "compartments": "compartment",
        "species": "species",
        "parameters": "parameter",
        "reactions": "reaction",
        "rules": "assignmentRule",
        "rate_rules": "rateRule",
        "algebraic_rules": "algebraicRule",
        "events": "event",
        "initial_assignments": "initialAssignment",
        "unit_definitions": "unitDefinition",
        "function_definitions": "functionDefinition",
    }
    counts = {
        key: sum(1 for element in elements if _local_name(str(element.tag)) == local)
        for key, local in count_names.items()
    }
    species = _model_list_rows(
        model,
        "listOfSpecies",
        "species",
        (
            "id",
            "name",
            "compartment",
            "initialAmount",
            "initialConcentration",
            "substanceUnits",
            "boundaryCondition",
            "constant",
            "hasOnlySubstanceUnits",
        ),
    )
    parameters = _model_list_rows(
        model,
        "listOfParameters",
        "parameter",
        ("id", "name", "value", "units", "constant"),
    )
    all_parameter_count = sum(
        1 for element in elements if _local_name(str(element.tag)) in {"parameter", "localParameter"}
    )
    counts["parameters"] = len(parameters)
    counts["local_parameters"] = max(0, all_parameter_count - len(parameters))
    compartments = _model_list_rows(
        model,
        "listOfCompartments",
        "compartment",
        ("id", "name", "size", "units", "spatialDimensions", "constant"),
    )
    model_units = {
        key: value
        for key in ("timeUnits", "substanceUnits", "extentUnits", "volumeUnits", "areaUnits", "lengthUnits")
        if (value := model.attrib.get(key)) is not None
    }

    kinetic_laws = [element for element in elements if _local_name(str(element.tag)) == "kineticLaw"]
    local_parameter_object_ids: set[int] = set()
    duplicate_local_ids: set[str] = set()
    for kinetic_law in kinetic_laws:
        local_ids: list[str] = []
        for element in kinetic_law.iter():
            if _local_name(str(element.tag)) not in {"parameter", "localParameter"}:
                continue
            local_parameter_object_ids.add(id(element))
            if identifier := element.attrib.get("id"):
                local_ids.append(str(identifier))
        duplicate_local_ids.update(
            identifier for identifier, count in Counter(local_ids).items() if count > 1
        )
    ids = [
        str(element.attrib["id"])
        for element in elements
        if id(element) not in local_parameter_object_ids and element.attrib.get("id")
    ]
    duplicate_ids = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    warnings: list[str] = []
    if not namespace.startswith("http://www.sbml.org/sbml/"):
        warnings.append("The document does not use a recognised core SBML namespace.")
    level = _optional_int(root.attrib.get("level"))
    version = _optional_int(root.attrib.get("version"))
    if level is None or version is None:
        warnings.append("SBML level/version metadata is missing or invalid.")
    if duplicate_ids:
        warnings.append(f"Duplicate model-scope SBML identifiers detected: {', '.join(duplicate_ids[:10])}.")
    if duplicate_local_ids:
        warnings.append(
            "Duplicate local parameter identifiers detected within a kinetic law: "
            f"{', '.join(sorted(duplicate_local_ids)[:10])}."
        )
    dynamic_count = counts["reactions"] + counts["rules"] + counts["rate_rules"] + counts["events"]
    if dynamic_count == 0:
        warnings.append("No reactions, assignment/rate rules, or events were found.")
    if counts["species"] == 0:
        warnings.append("No species were found; explicit output selections may be required.")
    if not model_units:
        warnings.append("No model-level units are declared; verify all quantities before comparison with IINTS.")
    packages = _package_prefixes(payload.decode("utf-8", errors="replace"))
    if packages:
        warnings.append(
            "SBML Level 3 package namespaces are present "
            f"({', '.join(packages)}); engine support must be checked explicitly."
        )

    readiness = (
        "inspectable"
        if not duplicate_ids and not duplicate_local_ids and level is not None and version is not None
        else "needs_review"
    )
    return SBMLModelSummary(
        model_path=resolved,
        sha256=_sha256_bytes(payload),
        byte_size=len(payload),
        sbml_level=level,
        sbml_version=version,
        namespace=namespace,
        model_id=str(model.attrib.get("id") or ""),
        model_name=str(model.attrib.get("name") or ""),
        counts=counts,
        model_units=model_units,
        species=species,
        parameters=parameters,
        compartments=compartments,
        package_prefixes=packages,
        warnings=tuple(warnings),
        readiness_status=readiness,
    )


def sbml_summary_payload(summary: SBMLModelSummary, *, include_local_path: bool = True) -> dict[str, Any]:
    """Convert an inspection result to JSON-safe metadata."""

    payload = asdict(summary)
    payload["model_path"] = str(summary.model_path) if include_local_path else summary.model_path.name
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_roadrunner() -> Any:
    try:
        return importlib.import_module("roadrunner")
    except ImportError as exc:
        raise RuntimeError(
            "libRoadRunner is required to execute SBML models. Install the optional engine with "
            "`python -m pip install 'iints-sdk-python35[mechanistic]'`. Inspection works without it."
        ) from exc


def roadrunner_status() -> dict[str, Any]:
    """Return availability information without importing model files."""

    try:
        module = _load_roadrunner()
    except RuntimeError as exc:
        return {"available": False, "engine": "libRoadRunner", "version": None, "message": str(exc)}
    return {
        "available": True,
        "engine": "libRoadRunner",
        "version": str(getattr(module, "__version__", "unknown")),
        "message": "libRoadRunner is available for local SBML execution.",
    }


def _species_ids(summary: SBMLModelSummary) -> set[str]:
    return {str(row["id"]) for row in summary.species if row.get("id")}


def _parameter_ids(summary: SBMLModelSummary) -> set[str]:
    return {str(row["id"]) for row in summary.parameters if row.get("id")}


def _normalise_selections(summary: SBMLModelSummary, requested: Iterable[str]) -> tuple[str, ...]:
    species_ids = _species_ids(summary)
    species_by_id = {str(row["id"]): row for row in summary.species if row.get("id")}
    parameter_ids = _parameter_ids(summary)
    cleaned = [value.strip() for value in requested if value and value.strip()]
    if len(cleaned) > 256:
        raise ValueError("At most 256 output variables may be selected.")
    if any(len(value) > 256 or any(ord(character) < 32 for character in value) for value in cleaned):
        raise ValueError("Output variable selections must be short, printable single-line values.")
    if not cleaned:
        cleaned = sorted(species_ids)
    if not cleaned:
        raise ValueError("The model has no species; provide explicit supported output variables.")

    selections: list[str] = ["time"]
    for value in cleaned:
        if value == "time":
            continue
        explicit_amount = value.startswith("amount:")
        explicit_concentration = value.startswith("concentration:")
        bracketed = value.startswith("[") and value.endswith("]")
        if explicit_amount:
            identifier = value.removeprefix("amount:").strip()
        elif explicit_concentration:
            identifier = value.removeprefix("concentration:").strip()
        else:
            identifier = value[1:-1] if bracketed else value
        if identifier in species_ids:
            has_only_substance_units = species_by_id[identifier].get("hasOnlySubstanceUnits") is True
            if explicit_amount:
                selection = identifier
            elif explicit_concentration or bracketed:
                selection = f"[{identifier}]"
            else:
                selection = identifier if has_only_substance_units else f"[{identifier}]"
        elif identifier in parameter_ids:
            if explicit_amount or explicit_concentration or bracketed:
                raise ValueError(
                    f"Selection qualifier for '{value}' is valid only for a declared species ID."
                )
            selection = identifier
        else:
            raise ValueError(
                f"Unknown SBML selection '{value}'. Select a declared species or global parameter ID."
            )
        if selection not in selections:
            selections.append(selection)
    return tuple(selections)


def _validated_source_url(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > 2048 or any(ord(character) < 32 for character in cleaned):
        raise ValueError("source_url must be a printable HTTPS URL no longer than 2048 characters.")
    parsed = urlsplit(cleaned)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("source_url must use HTTPS and include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source_url must not contain embedded credentials.")
    return cleaned


def _validated_model_license(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("model_license cannot be empty; use NOASSERTION when unknown.")
    if len(cleaned) > 128 or any(ord(character) < 32 for character in cleaned):
        raise ValueError("model_license must be a printable single-line value no longer than 128 characters.")
    return cleaned


def _selection_metadata(
    summary: SBMLModelSummary,
    selections: tuple[str, ...],
) -> list[dict[str, Any]]:
    species_by_id = {str(row["id"]): row for row in summary.species if row.get("id")}
    parameters_by_id = {str(row["id"]): row for row in summary.parameters if row.get("id")}
    rows: list[dict[str, Any]] = []
    for selection in selections:
        if selection == "time":
            rows.append(
                {
                    "selection": selection,
                    "identifier": "time",
                    "semantic": "time",
                    "declared_units": summary.model_units.get("timeUnits", "not declared"),
                }
            )
            continue
        concentration = selection.startswith("[") and selection.endswith("]")
        identifier = selection[1:-1] if concentration else selection
        if identifier in species_by_id:
            species = species_by_id[identifier]
            substance_units = species.get("substanceUnits") or summary.model_units.get(
                "substanceUnits", "not declared"
            )
            rows.append(
                {
                    "selection": selection,
                    "identifier": identifier,
                    "semantic": "species_concentration" if concentration else "species_amount",
                    "substance_units": substance_units,
                    "compartment": species.get("compartment", "not declared"),
                    "declared_units": (
                        f"{substance_units} per compartment volume; verify compartment units"
                        if concentration
                        else substance_units
                    ),
                }
            )
        elif identifier in parameters_by_id:
            parameter = parameters_by_id[identifier]
            rows.append(
                {
                    "selection": selection,
                    "identifier": identifier,
                    "semantic": "global_parameter",
                    "declared_units": parameter.get("units", "not declared"),
                }
            )
    return rows


def _result_rows(result: Any, fallback_columns: tuple[str, ...]) -> tuple[list[str], list[list[float]]]:
    raw_columns = getattr(result, "colnames", None)
    columns = [str(value) for value in raw_columns] if raw_columns is not None else list(fallback_columns)
    rows: list[list[float]] = []
    for raw_row in result:
        row = [float(value) for value in raw_row]
        if len(row) != len(columns):
            raise RuntimeError("libRoadRunner returned a row with an unexpected number of columns.")
        if not all(math.isfinite(value) for value in row):
            raise RuntimeError("libRoadRunner returned a non-finite simulation value.")
        rows.append(row)
    if not rows:
        raise RuntimeError("libRoadRunner returned no simulation rows.")
    return columns, rows


def _summary_statistics(columns: list[str], rows: list[list[float]]) -> dict[str, dict[str, float]]:
    statistics: dict[str, dict[str, float]] = {}
    for index, column in enumerate(columns):
        values = [row[index] for row in rows]
        statistics[column] = {
            "minimum": min(values),
            "maximum": max(values),
            "initial": values[0],
            "final": values[-1],
        }
    return statistics


def _integrator_metadata(runner: Any) -> tuple[str, dict[str, Any]]:
    """Read solver identity/settings without depending on one RoadRunner API generation."""

    name = "unknown"
    settings: dict[str, Any] = {}
    get_current_name = getattr(runner, "getCurrentIntegratorName", None)
    if callable(get_current_name):
        try:
            name = str(get_current_name())
        except Exception:
            name = "unknown"

    integrator = getattr(runner, "integrator", None)
    if integrator is None:
        return name, settings
    get_name = getattr(integrator, "getName", None)
    if name == "unknown" and callable(get_name):
        try:
            name = str(get_name())
        except Exception:
            name = "unknown"
    get_settings = getattr(integrator, "getSettingsMap", None)
    if callable(get_settings):
        try:
            raw_settings = dict(get_settings())
        except Exception:
            raw_settings = {}
        for key, value in raw_settings.items():
            if isinstance(value, float) and not math.isfinite(value):
                settings[str(key)] = str(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                settings[str(key)] = value
            else:
                settings[str(key)] = str(value)
    return name, settings


def _write_run_report(
    path: Path,
    *,
    summary: SBMLModelSummary,
    engine_version: str,
    integrator_name: str,
    integrator_settings: dict[str, Any],
    selections: tuple[str, ...],
    start: float,
    end: float,
    points: int,
    model_license: str,
    source_url: str | None,
) -> None:
    source_line = source_url or "not recorded"
    warnings = list(summary.warnings) or ["No structural inspection warnings."]
    lines = [
        "# IINTS Mechanistic Reference Model Run",
        "",
        "This run executes a local SBML model independently from the IINTS patient simulator.",
        "Execution success is not biological or clinical validation. The run does not calibrate IINTS "
        "automatically or support treatment decisions.",
        "",
        "## Provenance",
        "",
        f"- Model file: `{summary.model_path.name}`",
        f"- Model SHA-256: `{summary.sha256}`",
        f"- Model source: `{source_line}`",
        f"- Model license: `{model_license}`",
        f"- Engine: `libRoadRunner {engine_version}`",
        f"- Integrator: `{integrator_name}`",
        f"- Integrator settings: `{json.dumps(integrator_settings, sort_keys=True)}`",
        f"- SBML: level `{summary.sbml_level}`, version `{summary.sbml_version}`",
        "",
        "## Simulation",
        "",
        f"- Start: `{start}` model-time units",
        f"- End: `{end}` model-time units",
        f"- Points: `{points}`",
        f"- Selections: `{', '.join(selections)}`",
        "",
        "## Required Human Review",
        "",
        "1. Confirm the model license and source record before redistribution.",
        "2. Confirm every time, concentration, amount, and volume unit before comparing with IINTS.",
        "3. Confirm the model population, assumptions, and parameter context match the research question.",
        "4. Treat cross-model disagreement as evidence to investigate, not proof that either model is correct.",
        "",
        "## Structural Inspection Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_sbml_model(
    model_path: Path,
    output_dir: Path,
    *,
    start: float = 0.0,
    end: float = 1440.0,
    points: int = 289,
    variables: Iterable[str] = (),
    source_url: str | None = None,
    model_license: str = "NOASSERTION",
    _engine_module: Any | None = None,
) -> MechanisticRunResult:
    """Execute a local SBML model through libRoadRunner with full provenance.

    ``start`` and ``end`` remain in the model's declared time units.  IINTS does
    not infer conversions to minutes or concentrations to mg/dL.
    """

    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Simulation end must be finite and greater than start.")
    if points < 2 or points > 1_000_001:
        raise ValueError("points must be between 2 and 1,000,001.")
    model_license = _validated_model_license(model_license)
    source_url = _validated_source_url(source_url)

    summary = inspect_sbml_model(model_path)
    selections = _normalise_selections(summary, variables)
    engine_module = _engine_module or _load_roadrunner()
    engine_version = str(getattr(engine_module, "__version__", "unknown"))
    try:
        runner = engine_module.RoadRunner(str(summary.model_path))
        runner.timeCourseSelections = list(selections)
        raw_result = runner.simulate(float(start), float(end), int(points))
    except Exception as exc:
        raise RuntimeError(f"libRoadRunner could not execute the SBML model: {exc}") from exc
    columns, rows = _result_rows(raw_result, selections)
    integrator_name, integrator_settings = _integrator_metadata(runner)
    selection_metadata = _selection_metadata(summary, selections)

    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", summary.model_path.stem).strip("._") or "model"
    run_dir = output_root / f"sbml_{safe_stem}_{timestamp}_{summary.sha256[:8]}"
    run_dir.mkdir(parents=False, exist_ok=False)

    results_path = run_dir / "reference_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

    summary_path = run_dir / "sbml_model_summary.json"
    _write_json(summary_path, sbml_summary_payload(summary, include_local_path=False))
    report_path = run_dir / "REFERENCE_MODEL_REPORT.md"
    _write_run_report(
        report_path,
        summary=summary,
        engine_version=engine_version,
        integrator_name=integrator_name,
        integrator_settings=integrator_settings,
        selections=selections,
        start=float(start),
        end=float(end),
        points=int(points),
        model_license=model_license,
        source_url=source_url,
    )

    manifest = {
        "schema_version": MECHANISTIC_RUN_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "research_only": True,
        "medical_device": False,
        "evidence_source_ids": ["sbml_2019_l3v2_core", "libroadrunner_2015"],
        "model": {
            "file_name": summary.model_path.name,
            "sha256": summary.sha256,
            "source_url": source_url,
            "license": model_license,
            "sbml_level": summary.sbml_level,
            "sbml_version": summary.sbml_version,
        },
        "engine": {
            "name": "libRoadRunner",
            "version": engine_version,
            "integrator": integrator_name,
            "integrator_settings": integrator_settings,
        },
        "simulation": {
            "start": float(start),
            "end": float(end),
            "points": int(points),
            "selections": list(selections),
            "selection_metadata": selection_metadata,
            "time_units": summary.model_units.get("timeUnits", "not declared"),
        },
        "outputs": {
            "results_csv": results_path.name,
            "model_summary_json": summary_path.name,
            "report_md": report_path.name,
            "media_type": mimetypes.guess_type(results_path.name)[0] or "text/csv",
            "row_count": len(rows),
            "summary_statistics": _summary_statistics(columns, rows),
        },
        "limitations": [
            "Execution success is not biological or clinical validation.",
            "No model-time or concentration unit is converted to IINTS units automatically.",
            "The reference model never calibrates IINTS parameters without an explicit reviewed workflow.",
            "This output must not be used for insulin dosing or treatment decisions.",
        ],
    }
    manifest_path = run_dir / "mechanistic_run_manifest.json"
    _write_json(manifest_path, manifest)
    return MechanisticRunResult(
        run_dir=run_dir,
        results_csv=results_path,
        manifest_json=manifest_path,
        model_summary_json=summary_path,
        report_md=report_path,
        engine="libRoadRunner",
        engine_version=engine_version,
        row_count=len(rows),
        selections=selections,
    )
