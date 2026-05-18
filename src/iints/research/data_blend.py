from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .dataset import load_dataset, save_dataset


PREDICTOR_REQUIRED_COLUMNS = [
    "subject_id",
    "time_minutes",
    "glucose_actual_mgdl",
    "glucose_trend_mgdl_min",
    "patient_iob_units",
    "patient_cob_grams",
    "effective_isf",
    "effective_icr",
    "effective_basal_rate_u_per_hr",
]

PREDICTOR_OPTIONAL_COLUMNS = [
    "segment",
    "steps",
    "calories",
    "heart_rate",
    "sleep_minutes",
    "time_of_day_sin",
    "time_of_day_cos",
    "carb_intake_grams",
    "carb_grams",
    "meal_announcement_grams",
]


def _normalize_source(label: str, path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = load_dataset(path).copy()
    missing_required = [column for column in PREDICTOR_REQUIRED_COLUMNS if column not in df.columns]
    if missing_required:
        raise ValueError(f"{label}: missing required columns {missing_required}")

    original_subjects = df["subject_id"].astype(str)
    df["subject_id"] = label + ":" + original_subjects
    if "segment" not in df.columns:
        df["segment"] = 0
    for column in PREDICTOR_OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
    df["dataset_source"] = label
    ordered_columns = [
        *PREDICTOR_REQUIRED_COLUMNS,
        *[column for column in PREDICTOR_OPTIONAL_COLUMNS if column not in PREDICTOR_REQUIRED_COLUMNS],
        "dataset_source",
    ]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    report = {
        "label": label,
        "path": str(path),
        "rows": int(len(df)),
        "subjects": int(original_subjects.nunique()),
        "missing_optional_columns_filled": [
            column for column in PREDICTOR_OPTIONAL_COLUMNS if column not in load_dataset(path).columns
        ],
    }
    return df[[*ordered_columns, *remaining_columns]], report


def blend_predictor_datasets(
    sources: Iterable[Tuple[str, Path]],
    *,
    output_path: Path,
    manifest_path: Path | None = None,
) -> Dict[str, Any]:
    frames: List[pd.DataFrame] = []
    reports: List[Dict[str, Any]] = []
    for label, path in sources:
        frame, report = _normalize_source(label, path)
        frames.append(frame)
        reports.append(report)
    if not frames:
        raise ValueError("At least one source dataset is required.")

    blended = pd.concat(frames, ignore_index=True)
    save_dataset(blended, output_path)
    manifest = {
        "output_path": str(output_path),
        "rows": int(len(blended)),
        "subjects": int(blended["subject_id"].nunique()),
        "sources": reports,
        "required_columns": PREDICTOR_REQUIRED_COLUMNS,
        "optional_columns": PREDICTOR_OPTIONAL_COLUMNS,
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
