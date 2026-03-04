from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pandas as pd

from iints.research.dataset import subject_split


def _window_ids(
    df: pd.DataFrame,
    *,
    history_steps: int,
    horizon_steps: int,
    subject_column: str,
    segment_column: Optional[str],
) -> Set[str]:
    total = len(df)
    if total < (history_steps + horizon_steps):
        return set()

    boundary = pd.Series(False, index=df.index)
    if subject_column in df.columns:
        boundary |= df[subject_column] != df[subject_column].shift(1)
    if segment_column and segment_column in df.columns:
        boundary |= df[segment_column] != df[segment_column].shift(1)
    if len(boundary) > 0:
        boundary.iloc[0] = True
    boundary_arr = boundary.to_numpy(dtype=bool)

    ids: Set[str] = set()
    end_index = total - history_steps - horizon_steps + 1
    for idx in range(end_index):
        window_end = idx + history_steps
        horizon_end = window_end + horizon_steps
        if boundary_arr[idx + 1 : horizon_end].any():
            continue

        subject = str(df.iloc[idx][subject_column]) if subject_column in df.columns else "unknown"
        segment = str(df.iloc[idx][segment_column]) if segment_column and segment_column in df.columns else "seg"
        start_time = df.iloc[idx]["time_minutes"] if "time_minutes" in df.columns else idx
        ids.add(f"{subject}|{segment}|{start_time}|{horizon_end}")
    return ids


def audit_subject_split_and_leakage(
    df: pd.DataFrame,
    *,
    history_steps: int,
    horizon_steps: int,
    feature_columns: List[str],
    target_column: str,
    subject_column: str = "subject_id",
    segment_column: Optional[str] = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    if subject_column not in df.columns:
        raise ValueError(f"Missing subject column: {subject_column}")

    train_df, val_df, test_df = subject_split(
        df,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        subject_column=subject_column,
        seed=seed,
    )

    train_subjects = set(train_df[subject_column].unique())
    val_subjects = set(val_df[subject_column].unique())
    test_subjects = set(test_df[subject_column].unique())

    overlaps = {
        "train_val": sorted(train_subjects.intersection(val_subjects)),
        "train_test": sorted(train_subjects.intersection(test_subjects)),
        "val_test": sorted(val_subjects.intersection(test_subjects)),
    }

    ids_train = _window_ids(
        train_df,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        subject_column=subject_column,
        segment_column=segment_column,
    )
    ids_val = _window_ids(
        val_df,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        subject_column=subject_column,
        segment_column=segment_column,
    )
    ids_test = _window_ids(
        test_df,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        subject_column=subject_column,
        segment_column=segment_column,
    )

    seq_overlap = {
        "train_val": len(ids_train.intersection(ids_val)),
        "train_test": len(ids_train.intersection(ids_test)),
        "val_test": len(ids_val.intersection(ids_test)),
    }

    leakage_free = (
        len(overlaps["train_val"]) == 0
        and len(overlaps["train_test"]) == 0
        and len(overlaps["val_test"]) == 0
        and seq_overlap["train_val"] == 0
        and seq_overlap["train_test"] == 0
        and seq_overlap["val_test"] == 0
    )

    return {
        "leakage_free": leakage_free,
        "subject_counts": {
            "train": len(train_subjects),
            "val": len(val_subjects),
            "test": len(test_subjects),
        },
        "subject_overlap_ids": overlaps,
        "sequence_counts": {
            "train": len(ids_train),
            "val": len(ids_val),
            "test": len(ids_test),
        },
        "sequence_overlap_counts": seq_overlap,
    }
