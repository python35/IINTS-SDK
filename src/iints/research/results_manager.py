"""Scalable result indexing for IINTS research runs.

The SDK can generate many nested result folders over time. This module builds a
small, durable catalogue so users can search, compare, and archive runs without
opening every folder by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, cast

import numpy as np
import pandas as pd

from iints.utils.academic_artifacts import style_excel_workbook


GLUCOSE_COLUMNS = (
    "glucose_actual_mgdl",
    "glucose",
    "cgm_mgdl",
    "glucose_truth_mgdl",
)
TIME_COLUMNS = ("time_minutes", "timestamp", "time", "minute")
CARB_COLUMNS = ("carb_intake_grams", "carbs", "carbohydrates_grams")
INSULIN_COLUMNS = ("delivered_insulin_units", "insulin", "insulin_units")

ARTIFACT_TYPES = {
    ".csv": "table",
    ".tsv": "table",
    ".xlsx": "spreadsheet",
    ".xls": "spreadsheet",
    ".json": "metadata",
    ".md": "markdown",
    ".pdf": "report_pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".svg": "image",
    ".html": "html_report",
    ".htm": "html_report",
    ".parquet": "dataset",
    ".pt": "model",
    ".onnx": "model",
    ".zip": "archive",
}

CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResultsIndexBundle:
    """Paths created by a results-management indexing run."""

    output_dir: Path
    run_index_csv: Path
    artifact_inventory_csv: Path
    report_md: Path
    manifest_json: Path
    catalog_sqlite: Path
    workbook_xlsx: Path | None
    raw_long_csv: Path | None
    run_count: int
    artifact_count: int
    runs_updated: int
    runs_reused: int
    artifacts_updated: int
    artifacts_reused: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if np.isfinite(result):
        return result
    return None


def _count_positive(series: pd.Series | None) -> int:
    if series is None:
        return 0
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return int((numeric > 0).sum())


def _sum_numeric(series: pd.Series | None) -> float:
    if series is None:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _catalog_root_fingerprint(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()


def _open_catalog(path: Path, root: Path) -> sqlite3.Connection:
    """Open the local incremental catalogue and bind it to one results root."""

    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            relative_results_csv TEXT PRIMARY KEY,
            source_size_bytes INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            summary_json TEXT NOT NULL,
            indexed_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            relative_path TEXT PRIMARY KEY,
            source_size_bytes INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            record_json TEXT NOT NULL,
            indexed_utc TEXT NOT NULL
        )
        """
    )

    fingerprint = _catalog_root_fingerprint(root)
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key = 'root_fingerprint'"
    ).fetchone()
    if existing is not None and str(existing[0]) != fingerprint:
        # Never mix records from two roots when an output folder is reused.
        connection.execute("DELETE FROM runs")
        connection.execute("DELETE FROM artifacts")
    metadata = {
        "schema_version": str(CATALOG_SCHEMA_VERSION),
        "root_fingerprint": fingerprint,
        "root": str(root.resolve()),
    }
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        metadata.items(),
    )
    connection.commit()
    return connection


def _decode_catalog_record(raw: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_hidden_or_index_path(path: Path, output_dir: Path | None = None) -> bool:
    if any(part.startswith(".") for part in path.parts):
        return True
    if output_dir is not None:
        try:
            path.relative_to(output_dir)
            return True
        except ValueError:
            return False
    return False


def _normalise_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_sidecar_metadata(run_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("run_metadata.json", "config.json", "run_manifest.json", "realism_report.json", "safety_report.json"):
        payload = _read_json(run_dir / name)
        if payload:
            metadata[name.removesuffix(".json")] = payload
    return metadata


def _extract_config_metadata(sidecars: dict[str, Any]) -> dict[str, Any]:
    run_metadata = sidecars.get("run_metadata")
    config = run_metadata.get("config") if isinstance(run_metadata, dict) else {}
    if not isinstance(config, dict):
        config = sidecars.get("config") if isinstance(sidecars.get("config"), dict) else {}

    algorithm = config.get("algorithm") if isinstance(config, dict) else {}
    algorithm = algorithm if isinstance(algorithm, dict) else {}
    algorithm_meta_raw = algorithm.get("metadata")
    algorithm_meta = cast(dict[str, Any], algorithm_meta_raw) if isinstance(algorithm_meta_raw, dict) else {}
    scenario = config.get("scenario") if isinstance(config, dict) else {}
    scenario = scenario if isinstance(scenario, dict) else {}
    time_step_config = None
    if isinstance(config, dict):
        time_step_config = config.get("time_step")
        if time_step_config is None:
            time_step_config = config.get("time_step_minutes")

    return {
        "algorithm_name": algorithm_meta.get("name") or algorithm.get("name") or algorithm.get("class"),
        "algorithm_class": algorithm.get("class"),
        "patient_model_type": config.get("patient_model_type") if isinstance(config, dict) else None,
        "duration_config_minutes": config.get("duration_minutes") if isinstance(config, dict) else None,
        "time_step_config_minutes": time_step_config,
        "scenario_name": scenario.get("scenario_name") or scenario.get("name"),
        "physiology_variation_model": config.get("physiology_variation_model") if isinstance(config, dict) else None,
    }


def summarize_results_csv(results_csv: Path, root: Path | None = None) -> dict[str, Any]:
    """Summarize one `results.csv` without loading unrelated artifacts."""

    results_csv = results_csv.resolve()
    root = root.resolve() if root is not None else results_csv.parent
    run_dir = results_csv.parent
    sidecars = _load_sidecar_metadata(run_dir)
    config_metadata = _extract_config_metadata(sidecars)

    record: dict[str, Any] = {
        "run_id": _normalise_relative(run_dir, root).replace("/", "__") or run_dir.name,
        "run_dir": str(run_dir),
        "relative_run_dir": _normalise_relative(run_dir, root),
        "results_csv": str(results_csv),
        "relative_results_csv": _normalise_relative(results_csv, root),
        "size_bytes": results_csv.stat().st_size,
        "sha256": _sha256_file(results_csv),
        "quality_flag": "ok",
        **config_metadata,
    }

    try:
        df = pd.read_csv(results_csv)
    except Exception as exc:
        record.update({"quality_flag": "read_error", "error": str(exc)})
        return record

    record["rows"] = int(len(df))
    if df.empty:
        record.update({"quality_flag": "empty", "error": "results.csv has no rows"})
        return record

    glucose_col = _first_existing_column(df, GLUCOSE_COLUMNS)
    time_col = _first_existing_column(df, TIME_COLUMNS)
    carb_col = _first_existing_column(df, CARB_COLUMNS)
    insulin_col = _first_existing_column(df, INSULIN_COLUMNS)

    if glucose_col is None:
        record.update({"quality_flag": "missing_glucose", "error": "no glucose column found"})
        return record

    glucose = pd.to_numeric(df[glucose_col], errors="coerce").dropna()
    if glucose.empty:
        record.update({"quality_flag": "missing_glucose", "error": "glucose column contains no numeric values"})
        return record

    time_values = pd.to_numeric(df[time_col], errors="coerce") if time_col else pd.Series(np.arange(len(df)))
    valid_time = time_values.dropna()
    if len(valid_time) >= 2:
        duration_minutes = float(valid_time.iloc[-1] - valid_time.iloc[0])
        step_minutes = float(valid_time.diff().dropna().median())
    else:
        duration_minutes = None
        step_minutes = None

    diffs = glucose.diff().abs().dropna()
    if len(diffs) and step_minutes and step_minutes > 0:
        max_rate = float((diffs / step_minutes).max())
    else:
        max_rate = None

    mean_glucose = float(glucose.mean())
    sd_glucose = float(glucose.std(ddof=0))
    cv_pct = float(sd_glucose / mean_glucose * 100.0) if mean_glucose else None
    flat_step_ratio = float((diffs < 0.05).mean()) if len(diffs) else None

    carb_series = df[carb_col] if carb_col else None
    insulin_series = df[insulin_col] if insulin_col else None
    safety_triggered = df["safety_triggered"] if "safety_triggered" in df.columns else None
    fallback_triggered = df["fallback_triggered"] if "fallback_triggered" in df.columns else None

    realism = sidecars.get("realism_report", {})
    safety = sidecars.get("safety_report", {})

    record.update(
        {
            "glucose_column": glucose_col,
            "time_column": time_col,
            "duration_minutes": _safe_float(duration_minutes),
            "time_step_minutes": _safe_float(step_minutes),
            "mean_glucose_mgdl": round(mean_glucose, 3),
            "sd_glucose_mgdl": round(sd_glucose, 3),
            "cv_pct": round(cv_pct, 3) if cv_pct is not None else None,
            "min_glucose_mgdl": round(float(glucose.min()), 3),
            "max_glucose_mgdl": round(float(glucose.max()), 3),
            "glucose_range_mgdl": round(float(glucose.max() - glucose.min()), 3),
            "tir_70_180_pct": round(float(((glucose >= 70) & (glucose <= 180)).mean() * 100.0), 3),
            "time_below_70_pct": round(float((glucose < 70).mean() * 100.0), 3),
            "time_below_54_pct": round(float((glucose < 54).mean() * 100.0), 3),
            "time_above_180_pct": round(float((glucose > 180).mean() * 100.0), 3),
            "time_above_250_pct": round(float((glucose > 250).mean() * 100.0), 3),
            "max_step_delta_mgdl": round(float(diffs.max()), 3) if len(diffs) else 0.0,
            "max_rate_mgdl_per_min": round(max_rate, 3) if max_rate is not None else None,
            "flat_step_ratio": round(flat_step_ratio, 4) if flat_step_ratio is not None else None,
            "meal_event_count": _count_positive(carb_series),
            "total_carbs_grams": round(_sum_numeric(carb_series), 3),
            "insulin_event_count": _count_positive(insulin_series),
            "total_insulin_units": round(_sum_numeric(insulin_series), 5),
            "safety_triggered_count": _count_positive(safety_triggered),
            "fallback_triggered_count": _count_positive(fallback_triggered),
            "realism_verdict": realism.get("verdict"),
            "realism_score": realism.get("realism_score"),
            "safety_total_interventions": safety.get("total_interventions"),
            "safety_bolus_interventions": safety.get("bolus_interventions_count"),
        }
    )

    if record["max_rate_mgdl_per_min"] is not None and record["max_rate_mgdl_per_min"] > 3.05:
        record["quality_flag"] = "review_rate"
    elif record["flat_step_ratio"] is not None and record["flat_step_ratio"] >= 0.70:
        record["quality_flag"] = "review_flatline"
    elif record["realism_verdict"] in {"needs_review", "likely_unrealistic"}:
        record["quality_flag"] = "review_realism"

    return record


def discover_result_csvs(root: Path, output_dir: Path | None = None) -> list[Path]:
    """Find run-level result CSVs beneath a results root."""

    root = root.resolve()
    if not root.exists():
        return []
    candidates: list[Path] = []
    for path in root.rglob("results.csv"):
        if _is_hidden_or_index_path(path, output_dir):
            continue
        candidates.append(path)
    return sorted(candidates)


def build_artifact_inventory(root: Path, output_dir: Path | None = None) -> list[dict[str, Any]]:
    """Create a lightweight file inventory for reports, images, CSVs, and models."""

    root = root.resolve()
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if _is_hidden_or_index_path(path, output_dir):
            continue
        suffix = path.suffix.lower()
        stat = path.stat()
        rows.append(
            {
                "relative_path": _normalise_relative(path, root),
                "path": str(path),
                "artifact_type": ARTIFACT_TYPES.get(suffix, "other"),
                "extension": suffix,
                "size_bytes": int(stat.st_size),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return rows


def _incremental_run_records(
    connection: sqlite3.Connection,
    result_csvs: list[Path],
    root: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    cached = {
        str(row[0]): (int(row[1]), int(row[2]), row[3])
        for row in connection.execute(
            "SELECT relative_results_csv, source_size_bytes, source_mtime_ns, summary_json FROM runs"
        )
    }
    discovered: set[str] = set()
    records: list[dict[str, Any]] = []
    updated = 0
    reused = 0

    for path in result_csvs:
        relative = _normalise_relative(path, root)
        discovered.add(relative)
        stat = path.stat()
        cached_row = cached.get(relative)
        record = None
        if cached_row is not None and cached_row[:2] == (int(stat.st_size), int(stat.st_mtime_ns)):
            record = _decode_catalog_record(cached_row[2])
        if record is not None:
            reused += 1
        else:
            record = summarize_results_csv(path, root=root)
            connection.execute(
                """
                INSERT OR REPLACE INTO runs(
                    relative_results_csv, source_size_bytes, source_mtime_ns,
                    summary_json, indexed_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    relative,
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    json.dumps(record, sort_keys=True),
                    _utc_now(),
                ),
            )
            updated += 1
        records.append(record)

    stale = set(cached) - discovered
    connection.executemany(
        "DELETE FROM runs WHERE relative_results_csv = ?",
        ((relative,) for relative in stale),
    )
    return records, updated, reused


def _incremental_artifact_records(
    connection: sqlite3.Connection,
    root: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    cached = {
        str(row[0]): (int(row[1]), int(row[2]), row[3])
        for row in connection.execute(
            "SELECT relative_path, source_size_bytes, source_mtime_ns, record_json FROM artifacts"
        )
    }
    discovered: set[str] = set()
    records: list[dict[str, Any]] = []
    updated = 0
    reused = 0

    if root.exists():
        paths = sorted(path for path in root.rglob("*") if path.is_file())
    else:
        paths = []
    for path in paths:
        if _is_hidden_or_index_path(path, output_dir):
            continue
        relative = _normalise_relative(path, root)
        discovered.add(relative)
        stat = path.stat()
        cached_row = cached.get(relative)
        record = None
        if cached_row is not None and cached_row[:2] == (int(stat.st_size), int(stat.st_mtime_ns)):
            record = _decode_catalog_record(cached_row[2])
        if record is not None:
            reused += 1
        else:
            suffix = path.suffix.lower()
            record = {
                "relative_path": relative,
                "path": str(path),
                "artifact_type": ARTIFACT_TYPES.get(suffix, "other"),
                "extension": suffix,
                "size_bytes": int(stat.st_size),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
            connection.execute(
                """
                INSERT OR REPLACE INTO artifacts(
                    relative_path, source_size_bytes, source_mtime_ns,
                    record_json, indexed_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    relative,
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    json.dumps(record, sort_keys=True),
                    _utc_now(),
                ),
            )
            updated += 1
        records.append(record)

    stale = set(cached) - discovered
    connection.executemany(
        "DELETE FROM artifacts WHERE relative_path = ?",
        ((relative,) for relative in stale),
    )
    return records, updated, reused


def _write_markdown_report(
    root: Path,
    output_dir: Path,
    run_df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    include_raw: bool,
    *,
    runs_updated: int,
    runs_reused: int,
    artifacts_updated: int,
    artifacts_reused: int,
) -> Path:
    report_path = output_dir / "RESULTS_INDEX.md"

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        safe = frame.copy()
        safe = safe.fillna("")
        headers = [str(col) for col in safe.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in safe.iterrows():
            values = [str(value).replace("|", "\\|") for value in row.tolist()]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines: list[str] = [
        "# IINTS Results Index",
        "",
        "This catalogue is generated from local SDK outputs. It is research/education only and not a medical-device record.",
        "",
        "## Scope",
        "",
        f"- Root: `{root}`",
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Run folders indexed: `{len(run_df)}`",
        f"- Artifacts inventoried: `{len(artifact_df)}`",
        f"- Run summaries recalculated: `{runs_updated}`",
        f"- Run summaries reused from SQLite: `{runs_reused}`",
        f"- Artifact records refreshed: `{artifacts_updated}`",
        f"- Artifact records reused from SQLite: `{artifacts_reused}`",
        f"- Raw long-table export: `{'enabled' if include_raw else 'disabled'}`",
        "",
    ]

    if not run_df.empty:
        quality_counts = run_df["quality_flag"].fillna("unknown").value_counts().to_dict()
        lines.extend(["## Quality Overview", ""])
        for key, value in quality_counts.items():
            lines.append(f"- `{key}`: `{int(value)}`")
        lines.append("")

        summary_cols = [
            "relative_run_dir",
            "quality_flag",
            "realism_verdict",
            "mean_glucose_mgdl",
            "tir_70_180_pct",
            "time_below_70_pct",
            "time_above_180_pct",
            "max_step_delta_mgdl",
            "flat_step_ratio",
        ]
        available_summary_cols = [col for col in summary_cols if col in run_df.columns]
        lines.extend(["## Run Summary", ""])
        lines.append(markdown_table(run_df[available_summary_cols].head(30)))
        lines.append("")

        review_df = run_df[run_df["quality_flag"].fillna("ok") != "ok"]
        if not review_df.empty:
            lines.extend(["## Review Queue", ""])
            review_cols = [
                "relative_run_dir",
                "quality_flag",
                "realism_verdict",
                "max_rate_mgdl_per_min",
                "flat_step_ratio",
                "min_glucose_mgdl",
                "max_glucose_mgdl",
            ]
            review_cols = [col for col in review_cols if col in review_df.columns]
            lines.append(markdown_table(review_df[review_cols].head(50)))
            lines.append("")

        largest = run_df.sort_values("max_step_delta_mgdl", ascending=False).head(15)
        lines.extend(["## Largest Glucose Steps", ""])
        largest_cols = ["relative_run_dir", "max_step_delta_mgdl", "max_rate_mgdl_per_min", "quality_flag"]
        largest_cols = [col for col in largest_cols if col in largest.columns]
        lines.append(markdown_table(largest[largest_cols]))
        lines.append("")

    if not artifact_df.empty:
        artifact_counts = artifact_df["artifact_type"].value_counts().to_dict()
        lines.extend(["## Artifact Types", ""])
        for key, value in artifact_counts.items():
            lines.append(f"- `{key}`: `{int(value)}`")
        lines.append("")

    lines.extend(
        [
            "## Files Written",
            "",
            "- `run_index.csv`: one row per run-level `results.csv`.",
            "- `artifact_inventory.csv`: one row per file under the root.",
            "- `results_catalog.sqlite3`: incremental machine-readable catalogue used to avoid reparsing unchanged runs.",
            "- `result_manager_manifest.json`: paths, counts, and generation metadata.",
            "- `results_index.xlsx`: workbook version when spreadsheet support is available.",
        ]
    )
    if include_raw:
        lines.append("- `all_results_long.csv`: optional combined raw time-step table with `run_id` and source path columns.")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def index_results(
    root: Path,
    output_dir: Path | None = None,
    *,
    include_raw: bool = False,
) -> ResultsIndexBundle:
    """Index all local IINTS result artifacts under `root`.

    Parameters
    ----------
    root:
        Directory containing run folders and generated reports.
    output_dir:
        Directory for index artifacts. Defaults to `<root>/_iints_results_index`.
    include_raw:
        When true, concatenate every discovered `results.csv` into one long CSV.
        This can be large, so the default is metadata-only indexing.
    """

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")
    output_dir = (output_dir.expanduser().resolve() if output_dir else root / "_iints_results_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_csvs = discover_result_csvs(root, output_dir)
    catalog_sqlite = output_dir / "results_catalog.sqlite3"
    connection = _open_catalog(catalog_sqlite, root)
    try:
        run_records, runs_updated, runs_reused = _incremental_run_records(
            connection,
            result_csvs,
            root,
        )
        artifact_records, artifacts_updated, artifacts_reused = _incremental_artifact_records(
            connection,
            root,
            output_dir,
        )
        connection.commit()
    finally:
        connection.close()

    run_df = pd.DataFrame(run_records)
    artifact_df = pd.DataFrame(artifact_records)

    run_index_csv = output_dir / "run_index.csv"
    artifact_inventory_csv = output_dir / "artifact_inventory.csv"
    run_df.to_csv(run_index_csv, index=False)
    artifact_df.to_csv(artifact_inventory_csv, index=False)

    raw_long_csv: Path | None = None
    if include_raw and result_csvs:
        raw_long_csv = output_dir / "all_results_long.csv"
        first = True
        for path in result_csvs:
            df = pd.read_csv(path)
            run_id = _normalise_relative(path.parent, root).replace("/", "__") or path.parent.name
            df.insert(0, "source_results_csv", _normalise_relative(path, root))
            df.insert(0, "run_id", run_id)
            df.to_csv(raw_long_csv, index=False, mode="w" if first else "a", header=first)
            first = False

    workbook_path = output_dir / "results_index.xlsx"
    workbook_xlsx: Path | None = workbook_path
    try:
        with pd.ExcelWriter(workbook_path) as writer:
            run_df.to_excel(writer, sheet_name="runs", index=False)
            artifact_df.to_excel(writer, sheet_name="artifacts", index=False)
            if not run_df.empty:
                overview = (
                    run_df.groupby("quality_flag", dropna=False)
                    .size()
                    .reset_index(name="run_count")
                    .sort_values("run_count", ascending=False)
                )
                overview.to_excel(writer, sheet_name="quality_overview", index=False)
        style_excel_workbook(workbook_path, title="IINTS-AF Research Results Index")
    except Exception:
        workbook_xlsx = None

    report_md = _write_markdown_report(
        root,
        output_dir,
        run_df,
        artifact_df,
        include_raw,
        runs_updated=runs_updated,
        runs_reused=runs_reused,
        artifacts_updated=artifacts_updated,
        artifacts_reused=artifacts_reused,
    )
    manifest_json = output_dir / "result_manager_manifest.json"
    manifest = {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "output_dir": str(output_dir),
        "run_count": int(len(run_df)),
        "artifact_count": int(len(artifact_df)),
        "incremental_index": {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "runs_updated": runs_updated,
            "runs_reused": runs_reused,
            "artifacts_updated": artifacts_updated,
            "artifacts_reused": artifacts_reused,
        },
        "include_raw": include_raw,
        "artifacts": {
            "run_index_csv": str(run_index_csv),
            "artifact_inventory_csv": str(artifact_inventory_csv),
            "catalog_sqlite": str(catalog_sqlite),
            "report_md": str(report_md),
            "workbook_xlsx": str(workbook_xlsx) if workbook_xlsx else None,
            "raw_long_csv": str(raw_long_csv) if raw_long_csv else None,
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))

    return ResultsIndexBundle(
        output_dir=output_dir,
        run_index_csv=run_index_csv,
        artifact_inventory_csv=artifact_inventory_csv,
        report_md=report_md,
        manifest_json=manifest_json,
        catalog_sqlite=catalog_sqlite,
        workbook_xlsx=workbook_xlsx,
        raw_long_csv=raw_long_csv,
        run_count=int(len(run_df)),
        artifact_count=int(len(artifact_df)),
        runs_updated=runs_updated,
        runs_reused=runs_reused,
        artifacts_updated=artifacts_updated,
        artifacts_reused=artifacts_reused,
    )
