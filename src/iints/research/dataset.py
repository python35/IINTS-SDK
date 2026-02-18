from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    history_steps: int,
    horizon_steps: int,
    feature_columns: List[str],
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a time-series dataframe into (X, y) sequences.
    X shape: [N, history_steps, num_features]
    y shape: [N, horizon_steps]
    """
    if history_steps <= 0 or horizon_steps <= 0:
        raise ValueError("history_steps and horizon_steps must be > 0")
    missing = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    values = df[feature_columns].to_numpy(dtype=np.float32)
    target = df[target_column].to_numpy(dtype=np.float32)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    total = len(df)
    end_index = total - history_steps - horizon_steps + 1
    for idx in range(end_index):
        X_list.append(values[idx : idx + history_steps])
        y_list.append(target[idx + history_steps : idx + history_steps + horizon_steps])
    if not X_list:
        raise ValueError("Not enough rows to build sequences with current window sizes.")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Parquet support requires pyarrow. Install with `pip install iints-sdk-python35[research]`."
        ) from exc


def load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(
            "Parquet support requires pyarrow. Install with `pip install iints-sdk-python35[research]`."
        ) from exc


def concat_runs(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True)


def basic_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for col in columns:
        if col in df.columns:
            stats[f"{col}_mean"] = float(df[col].mean())
            stats[f"{col}_std"] = float(df[col].std())
    return stats
