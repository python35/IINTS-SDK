from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from iints.data.registry import get_dataset, load_dataset_registry

DEFAULT_RESEARCH_DATASET_IDS: tuple[str, ...] = (
    "hupa_ucm",
    "azt1d",
    "t1d_uom",
    "ohio_t1dm",
    "dclp3_idcl",
    "jaeb_loop",
    "t1dexi",
    "t1dexip",
    "d1namo",
    "openaps_data_commons",
    "metabonet",
    "glucose_ml",
)

TASK_COLUMNS: tuple[str, ...] = (
    "glucose_forecasting",
    "controller_training",
    "exercise_research",
    "closed_loop_benchmarking",
    "multimodal_research",
    "external_validation",
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _task_flags(entry: Dict[str, Any]) -> dict[str, bool]:
    text = " ".join(
        _as_list(entry.get("ai_tasks"))
        + _as_list(entry.get("modalities"))
        + [str(entry.get("description", "")), str(entry.get("recommended_use", ""))]
    ).lower()
    return {
        "glucose_forecasting": any(token in text for token in ("forecast", "prediction", "cgm")),
        "controller_training": any(token in text for token in ("controller", "policy", "imitation")),
        "exercise_research": any(token in text for token in ("exercise", "activity", "heart rate", "wearable")),
        "closed_loop_benchmarking": any(token in text for token in ("closed-loop", "closed loop", "aid", "loop", "control-iq")),
        "multimodal_research": any(token in text for token in ("multimodal", "sleep", "activity", "wearable", "food")),
        "external_validation": str(entry.get("access", "")) not in {"bundled"},
    }


def resolve_research_dataset_entries(dataset_ids: Sequence[str] | None = None) -> list[Dict[str, Any]]:
    """Return registry entries in a stable research-priority order."""
    if dataset_ids:
        return [get_dataset(dataset_id) for dataset_id in dataset_ids]

    registry = {entry.get("id"): entry for entry in load_dataset_registry()}
    entries: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for dataset_id in DEFAULT_RESEARCH_DATASET_IDS:
        entry = registry.get(dataset_id)
        if entry is not None:
            entries.append(entry)
            seen.add(dataset_id)
    for entry in load_dataset_registry():
        dataset_id = str(entry.get("id", ""))
        if dataset_id and dataset_id not in seen and entry.get("access") != "bundled":
            entries.append(entry)
            seen.add(dataset_id)
    return entries


def build_research_dataset_matrix(dataset_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for priority, entry in enumerate(resolve_research_dataset_entries(dataset_ids), start=1):
        flags = _task_flags(entry)
        row: dict[str, Any] = {
            "priority": priority,
            "dataset_id": entry.get("id", ""),
            "name": entry.get("name", ""),
            "access": entry.get("access", ""),
            "source": entry.get("source", ""),
            "license": entry.get("license", ""),
            "landing_page": entry.get("landing_page", ""),
            "doi": entry.get("doi", ""),
            "iints_status": entry.get("iints_status", "catalogued"),
            "recommended_use": entry.get("recommended_use", entry.get("description", "")),
            "recommended_prep_command": entry.get("recommended_prep_command", "iints import-data --help"),
            "modalities": "; ".join(_as_list(entry.get("modalities"))),
            "ai_tasks": "; ".join(_as_list(entry.get("ai_tasks"))),
            "limitations": "; ".join(_as_list(entry.get("limitations"))),
        }
        for column in TASK_COLUMNS:
            row[column] = flags[column]
        matrix.append(row)
    return matrix


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_plan_markdown(matrix: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# IINTS Diabetes Research Dataset Plan",
        "",
        "This plan lists the external datasets that are useful for IINTS local-AI research.",
        "It is a research acquisition and provenance document, not a medical-device validation claim.",
        "",
        "## Rules",
        "",
        "- Keep every raw dataset in its own folder under `data_packs/public/<dataset_id>/raw`.",
        "- Keep converted files under `data_packs/public/<dataset_id>/processed`.",
        "- Preserve source dataset IDs in every blended file to prevent leakage.",
        "- Split by subject, never by random rows, for model evaluation.",
        "- Record license/access terms before training local AI models.",
        "- Use MDMP certification and EU AI Pact review before using a dataset as evidence.",
        "",
        "## Priority Matrix",
        "",
        "| Priority | Dataset | Access | Best use | IINTS status |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in matrix:
        lines.append(
            f"| {row['priority']} | `{row['dataset_id']}` - {row['name']} | {row['access']} | {row['recommended_use']} | {row['iints_status']} |"
        )

    lines.extend([
        "",
        "## Acquisition Checklist",
        "",
    ])
    for row in matrix:
        lines.extend(
            [
                f"### `{row['dataset_id']}` - {row['name']}",
                "",
                f"- Source: {row['source']}",
                f"- Access: {row['access']}",
                f"- License/access terms: {row['license']}",
                f"- Landing page: {row['landing_page']}",
                f"- DOI: {row['doi'] or 'n/a'}",
                f"- Modalities: {row['modalities'] or 'not recorded'}",
                f"- AI tasks: {row['ai_tasks'] or 'not recorded'}",
                f"- Recommended command: `{row['recommended_prep_command']}`",
                f"- Limitations: {row['limitations'] or 'Review source documentation before use.'}",
                "",
            ]
        )

    lines.extend([
        "## Model Training Flow",
        "",
        "```bash",
        "# 1. Prepare each dataset separately.",
        "iints research prepare-hupa",
        "iints research prepare-azt1d",
        "iints research prepare-ohio",
        "",
        "# 2. Blend only prepared, leakage-safe predictor datasets.",
        "iints research blend-datasets \\",
        "  --source hupa=data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv \\",
        "  --source azt1d=data_packs/public/azt1d/processed/azt1d_merged.csv \\",
        "  --source ohio=data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv \\",
        "  --output data_packs/processed/iints_research_blend.csv",
        "",
        "# 3. Train local research models only after data-quality review.",
        "iints research train-local-ai \\",
        "  --run results/jetson_research_day \\",
        "  --output-dir results/local_ai_lab",
        "```",
        "",
        "## Research Boundary",
        "",
        "These datasets can support simulation realism, glucose forecasting, and bench-only controller research.",
        "They do not make IINTS a certified medical device and must not be used for real insulin dosing.",
    ])
    return "\n".join(lines) + "\n"


def _render_bibtex(entries: list[Dict[str, Any]]) -> str:
    chunks: list[str] = []
    for entry in entries:
        citation = entry.get("citation", {})
        bibtex = citation.get("bibtex") if isinstance(citation, dict) else None
        if bibtex:
            chunks.append(str(bibtex).strip())
    return "\n\n".join(chunks) + ("\n" if chunks else "")


def write_research_dataset_plan(output_dir: Path, dataset_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Write a local dataset acquisition plan for AI research."""
    entries = resolve_research_dataset_entries(dataset_ids)
    matrix = build_research_dataset_matrix(dataset_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = output_dir / "research_dataset_matrix.csv"
    snapshot_json = output_dir / "dataset_registry_snapshot.json"
    plan_md = output_dir / "DATASET_ACQUISITION_PLAN.md"
    citations_bib = output_dir / "SOURCE_CITATIONS.bib"
    manifest_json = output_dir / "research_dataset_plan_manifest.json"

    _write_csv(matrix_csv, matrix)
    _write_json(snapshot_json, entries)
    plan_md.write_text(_render_plan_markdown(matrix), encoding="utf-8")
    citations_bib.write_text(_render_bibtex(entries), encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(entries),
        "dataset_ids": [entry.get("id", "") for entry in entries],
        "research_use_only": True,
        "outputs": {
            "plan_md": str(plan_md),
            "matrix_csv": str(matrix_csv),
            "snapshot_json": str(snapshot_json),
            "citations_bib": str(citations_bib),
            "manifest_json": str(manifest_json),
        },
    }
    _write_json(manifest_json, manifest)
    return manifest
