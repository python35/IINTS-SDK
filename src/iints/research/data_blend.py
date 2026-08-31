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


def _infer_isf_icr_fallback_flag(df: pd.DataFrame, column: str) -> pd.Series:
    # Older processed files predate the *_is_fallback provenance columns.
    # Best-effort back-fill: a column that is constant across the whole file
    # is almost certainly an unconditional CLI-default fill (see AZT1D/OhioT1DM
    # prep scripts), not an estimate.
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.Series(df[column].nunique(dropna=True) <= 1, index=df.index)


def _infer_insulin_units_semantics(df: pd.DataFrame) -> pd.Series:
    if "insulin_units" not in df.columns:
        return pd.Series("unknown", index=df.index)
    nonzero_frac = float((df["insulin_units"] != 0).mean())
    label = "dense_expansion" if nonzero_frac >= 0.95 else "sparse_event"
    return pd.Series(label, index=df.index)


def _subject_sample_weights(subject_id: pd.Series) -> pd.Series:
    # Counteracts within-source subject-count skew (e.g. one subject
    # contributing >50% of a cohort's rows) so a single oversized subject
    # doesn't dominate a blended training pool. An "average-sized" subject
    # in this source gets weight ~1; oversized subjects get weight <1.
    counts = subject_id.value_counts()
    mean_count = float(counts.mean())
    per_subject_weight = mean_count / counts
    return subject_id.map(per_subject_weight).astype(float)


PROVENANCE_COLUMNS = [
    "sample_weight",
    "effective_isf_is_fallback",
    "effective_icr_is_fallback",
    "insulin_units_semantics",
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

    if "effective_isf_is_fallback" not in df.columns:
        df["effective_isf_is_fallback"] = _infer_isf_icr_fallback_flag(df, "effective_isf")
    if "effective_icr_is_fallback" not in df.columns:
        df["effective_icr_is_fallback"] = _infer_isf_icr_fallback_flag(df, "effective_icr")
    if "insulin_units_semantics" not in df.columns:
        df["insulin_units_semantics"] = _infer_insulin_units_semantics(df)
    df["sample_weight"] = _subject_sample_weights(original_subjects)

    df["dataset_source"] = label
    ordered_columns = [
        *PREDICTOR_REQUIRED_COLUMNS,
        *[column for column in PREDICTOR_OPTIONAL_COLUMNS if column not in PREDICTOR_REQUIRED_COLUMNS],
        *PROVENANCE_COLUMNS,
        "dataset_source",
    ]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    subject_counts = original_subjects.value_counts()
    top_subject_share = float(subject_counts.max() / len(df)) if len(df) else 0.0
    report = {
        "label": label,
        "path": str(path),
        "rows": int(len(df)),
        "subjects": int(original_subjects.nunique()),
        "missing_optional_columns_filled": [
            column for column in PREDICTOR_OPTIONAL_COLUMNS if column not in load_dataset(path).columns
        ],
        "top_subject_share": top_subject_share,
        "effective_isf_fallback_pct": float(df["effective_isf_is_fallback"].mean() * 100.0),
        "effective_icr_fallback_pct": float(df["effective_icr_is_fallback"].mean() * 100.0),
        "insulin_units_semantics": sorted(df["insulin_units_semantics"].astype(str).unique().tolist()),
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
    warnings = [
        f"{report['label']}: one subject is {report['top_subject_share'] * 100:.1f}% of this source's rows"
        for report in reports
        if report["top_subject_share"] > 0.3
    ]
    manifest = {
        "output_path": str(output_path),
        "rows": int(len(blended)),
        "subjects": int(blended["subject_id"].nunique()),
        "sources": reports,
        "warnings": warnings,
        "required_columns": PREDICTOR_REQUIRED_COLUMNS,
        "optional_columns": PREDICTOR_OPTIONAL_COLUMNS,
        "provenance_columns": PROVENANCE_COLUMNS,
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
