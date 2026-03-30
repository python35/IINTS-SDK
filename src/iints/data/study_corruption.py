from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


AVAILABLE_STUDY_CORRUPTIONS = [
    "timestamp_shift",
    "missing_block",
    "duplicate_rows",
    "glucose_spikes",
    "drop_meal_annotations",
    "unit_scale_error",
]


def _normalize_column(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {col: _normalize_column(col) for col in columns}
    candidate_set = {_normalize_column(candidate) for candidate in candidates}
    for column, normalized_name in normalized.items():
        if normalized_name in candidate_set:
            return column
    return None


def _timestamp_column(df: pd.DataFrame) -> str | None:
    return _find_column(
        df.columns,
        ["timestamp", "timestamp_iso", "datetime", "date", "time", "device timestamp", "time_minutes"],
    )


def _glucose_column(df: pd.DataFrame) -> str | None:
    return _find_column(
        df.columns,
        ["glucose", "glucose_actual_mgdl", "glucose_mgdl", "bg", "sgv", "sensor_glucose_mgdl"],
    )


def _carbs_column(df: pd.DataFrame) -> str | None:
    return _find_column(df.columns, ["carbs", "carb", "carbohydrates", "carbgrams"])


def apply_study_corruptions(
    dataframe: pd.DataFrame,
    *,
    modes: list[str],
    seed: int = 42,
    timestamp_shift_minutes: int = 60,
    missing_fraction: float = 0.10,
    duplicate_fraction: float = 0.05,
    spike_fraction: float = 0.03,
    spike_magnitude_mgdl: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    invalid = [mode for mode in modes if mode not in AVAILABLE_STUDY_CORRUPTIONS]
    if invalid:
        allowed = ", ".join(AVAILABLE_STUDY_CORRUPTIONS)
        raise ValueError(f"Unknown corruption mode(s): {', '.join(invalid)}. Use: {allowed}")

    rng = np.random.default_rng(seed)
    corrupted = dataframe.copy()
    manifest_operations: list[dict[str, Any]] = []
    before_rows = int(len(corrupted))
    before_columns = list(corrupted.columns)

    timestamp_column = _timestamp_column(corrupted)
    glucose_column = _glucose_column(corrupted)
    carbs_column = _carbs_column(corrupted)

    for mode in modes:
        if mode == "timestamp_shift":
            if timestamp_column is None:
                manifest_operations.append(
                    {"mode": mode, "applied": False, "reason": "No timestamp-like column found"}
                )
                continue
            if _normalize_column(timestamp_column) == "timeminutes":
                corrupted[timestamp_column] = pd.to_numeric(corrupted[timestamp_column], errors="coerce") + float(timestamp_shift_minutes)
            else:
                parsed = pd.to_datetime(corrupted[timestamp_column], errors="coerce")
                if parsed.notna().any():
                    shifted = parsed + pd.to_timedelta(timestamp_shift_minutes, unit="m")
                    if "timestamp_iso" in timestamp_column.lower() or parsed.dt.tz is not None:
                        corrupted[timestamp_column] = shifted.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    else:
                        corrupted[timestamp_column] = shifted.dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    manifest_operations.append(
                        {"mode": mode, "applied": False, "reason": f"Could not parse {timestamp_column} as datetime"}
                    )
                    continue
            manifest_operations.append(
                {
                    "mode": mode,
                    "applied": True,
                    "column": timestamp_column,
                    "minutes": int(timestamp_shift_minutes),
                    "expected_effect": "Temporal ordering and provenance checks should flag the shift.",
                }
            )

        elif mode == "missing_block":
            if len(corrupted) < 3:
                manifest_operations.append({"mode": mode, "applied": False, "reason": "Dataset too small"})
                continue
            block_size = max(1, int(round(len(corrupted) * missing_fraction)))
            start = int(rng.integers(0, max(len(corrupted) - block_size + 1, 1)))
            corrupted = corrupted.drop(corrupted.index[start : start + block_size]).reset_index(drop=True)
            manifest_operations.append(
                {
                    "mode": mode,
                    "applied": True,
                    "rows_removed": int(block_size),
                    "expected_effect": "Gap handling and coverage checks should worsen.",
                }
            )

        elif mode == "duplicate_rows":
            if len(corrupted) == 0:
                manifest_operations.append({"mode": mode, "applied": False, "reason": "Empty dataset"})
                continue
            duplicate_count = max(1, int(round(len(corrupted) * duplicate_fraction)))
            sample_idx = rng.integers(0, len(corrupted), size=duplicate_count)
            duplicates = corrupted.iloc[sample_idx].copy()
            corrupted = pd.concat([corrupted, duplicates], ignore_index=True)
            manifest_operations.append(
                {
                    "mode": mode,
                    "applied": True,
                    "rows_added": int(duplicate_count),
                    "expected_effect": "Row-count consistency and deduplication checks should trigger.",
                }
            )

        elif mode == "glucose_spikes":
            if glucose_column is None:
                manifest_operations.append(
                    {"mode": mode, "applied": False, "reason": "No glucose-like column found"}
                )
                continue
            numeric = pd.to_numeric(corrupted[glucose_column], errors="coerce")
            valid_idx = np.flatnonzero(numeric.notna().to_numpy())
            if len(valid_idx) == 0:
                manifest_operations.append(
                    {"mode": mode, "applied": False, "reason": f"No numeric values in {glucose_column}"}
                )
                continue
            spike_count = max(1, int(round(len(valid_idx) * spike_fraction)))
            chosen = rng.choice(valid_idx, size=min(spike_count, len(valid_idx)), replace=False)
            deltas = rng.choice([-1.0, 1.0], size=len(chosen)) * float(spike_magnitude_mgdl)
            for row_idx, delta in zip(chosen, deltas):
                corrupted.at[int(row_idx), glucose_column] = float(numeric.iloc[int(row_idx)]) + float(delta)
            manifest_operations.append(
                {
                    "mode": mode,
                    "applied": True,
                    "column": glucose_column,
                    "affected_rows": int(len(chosen)),
                    "magnitude_mgdl": float(spike_magnitude_mgdl),
                    "expected_effect": "Plausibility filters and realism review should mark suspicious spikes.",
                }
            )

        elif mode == "drop_meal_annotations":
            if carbs_column is None:
                manifest_operations.append(
                    {"mode": mode, "applied": False, "reason": "No carbohydrate-like column found"}
                )
                continue
            corrupted[carbs_column] = 0
            manifest_operations.append(
                {
                    "mode": mode,
                    "applied": True,
                    "column": carbs_column,
                    "expected_effect": "Meal-context-dependent evaluation should degrade or change interpretation.",
                }
            )

        elif mode == "unit_scale_error":
            if glucose_column is None:
                manifest_operations.append(
                    {"mode": mode, "applied": False, "reason": "No glucose-like column found"}
                )
                continue
            numeric = pd.to_numeric(corrupted[glucose_column], errors="coerce")
            finite = numeric.dropna()
            if finite.empty:
                manifest_operations.append(
                    {"mode": mode, "applied": False, "reason": f"No numeric values in {glucose_column}"}
                )
                continue
            factor = 0.0555 if float(finite.mean()) > 40.0 else 18.0
            corrupted[glucose_column] = numeric * factor
            manifest_operations.append(
                {
                    "mode": mode,
                    "applied": True,
                    "column": glucose_column,
                    "factor": float(factor),
                    "expected_effect": "Certification range checks should catch the unit mismatch.",
                }
            )

    manifest = {
        "source_rows": before_rows,
        "output_rows": int(len(corrupted)),
        "source_columns": before_columns,
        "applied_modes": modes,
        "operations": manifest_operations,
        "seed": seed,
    }
    return corrupted, manifest


def write_corrupted_study_csv(
    input_csv: str | Path,
    *,
    output_csv: str | Path,
    modes: list[str],
    manifest_output: str | Path | None = None,
    seed: int = 42,
    timestamp_shift_minutes: int = 60,
    missing_fraction: float = 0.10,
    duplicate_fraction: float = 0.05,
    spike_fraction: float = 0.03,
    spike_magnitude_mgdl: float = 60.0,
) -> dict[str, str]:
    source = Path(input_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    target = Path(output_csv).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(source)
    corrupted, manifest = apply_study_corruptions(
        dataframe,
        modes=modes,
        seed=seed,
        timestamp_shift_minutes=timestamp_shift_minutes,
        missing_fraction=missing_fraction,
        duplicate_fraction=duplicate_fraction,
        spike_fraction=spike_fraction,
        spike_magnitude_mgdl=spike_magnitude_mgdl,
    )
    corrupted.to_csv(target, index=False)

    manifest_path = Path(manifest_output).expanduser().resolve() if manifest_output is not None else target.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = dict(manifest)
    manifest_payload["input_csv"] = str(source)
    manifest_payload["output_csv"] = str(target)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return {
        "corrupted_csv": str(target),
        "manifest_json": str(manifest_path),
    }
