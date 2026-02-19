from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    history_steps: int,
    horizon_steps: int,
    feature_columns: List[str],
    target_column: str,
    subject_column: Optional[str] = "subject_id",
    segment_column: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a time-series dataframe into (X, y) sequences.

    X shape: [N, history_steps, num_features]
    y shape: [N, horizon_steps]

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, sorted by time within each subject/segment.
    history_steps : int
        Number of past steps used as input features.
    horizon_steps : int
        Number of future steps to predict.
    feature_columns : list of str
        Columns to use as input features.
    target_column : str
        Column to predict.
    subject_column : str or None
        If provided, sequences will not cross subject boundaries.
        Defaults to "subject_id".
    segment_column : str or None
        If provided, sequences will additionally not cross segment boundaries
        (e.g. gaps in CGM data).  Defaults to None.

    Returns
    -------
    X : np.ndarray of shape [N, history_steps, num_features]
    y : np.ndarray of shape [N, horizon_steps]
    """
    if history_steps <= 0 or horizon_steps <= 0:
        raise ValueError("history_steps and horizon_steps must be > 0")
    missing = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build a boundary mask: True at every row that starts a new group
    # (subject or segment), so we never build a window that crosses the boundary.
    boundary = pd.Series(False, index=df.index)
    if subject_column and subject_column in df.columns:
        boundary |= df[subject_column] != df[subject_column].shift(1)
    if segment_column and segment_column in df.columns:
        boundary |= df[segment_column] != df[segment_column].shift(1)
    # First row is always a boundary
    if len(boundary) > 0:
        boundary.iloc[0] = True

    values = df[feature_columns].to_numpy(dtype=np.float32)
    target = df[target_column].to_numpy(dtype=np.float32)
    boundary_arr = boundary.to_numpy(dtype=bool)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    total = len(df)
    end_index = total - history_steps - horizon_steps + 1

    for idx in range(end_index):
        window_end = idx + history_steps      # exclusive, first prediction step
        horizon_end = window_end + horizon_steps

        # Reject window if any boundary falls *inside* the history window
        # (idx+1 .. window_end) or at the start of the horizon (window_end).
        # idx itself may be a boundary (start of a new subject), which is fine.
        if boundary_arr[idx + 1 : horizon_end].any():
            continue

        X_list.append(values[idx:window_end])
        y_list.append(target[window_end:horizon_end])

    if not X_list:
        raise ValueError("Not enough rows to build sequences with current window sizes.")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


# ---------------------------------------------------------------------------
# P0-2: Subject-level train / val / test split
# ---------------------------------------------------------------------------

def subject_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    subject_column: str = "subject_id",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train / val / test sets by subject ID.

    All rows belonging to one subject stay in the same split, which prevents
    data leakage between sets (a subject's glucose patterns are unique and
    would otherwise trivially inflate validation metrics).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a subject identifier column.
    val_fraction : float
        Fraction of *subjects* assigned to the validation set.
    test_fraction : float
        Fraction of *subjects* assigned to the held-out test set.
    subject_column : str
        Name of the subject identifier column.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Three non-overlapping subsets.
    """
    if subject_column not in df.columns:
        raise ValueError(
            f"Column '{subject_column}' not found in dataframe. "
            "Cannot perform subject-level split."
        )
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(df[subject_column].unique()))
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, round(n * test_fraction))
    n_val = max(1, round(n * val_fraction))
    # Ensure we have at least one training subject
    if n - n_val - n_test < 1:
        raise ValueError(
            f"Not enough subjects ({n}) for the requested val/test fractions. "
            "Reduce val_fraction or test_fraction."
        )

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test: n_test + n_val])
    train_subjects = set(subjects[n_test + n_val:])

    train_df = df[df[subject_column].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df[subject_column].isin(val_subjects)].reset_index(drop=True)
    test_df = df[df[subject_column].isin(test_subjects)].reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# P3-10: Feature normalisation
# ---------------------------------------------------------------------------

class FeatureScaler:
    """
    Fit-transform scaler for LSTM feature arrays.

    Two strategies are supported:

    ``"zscore"``
        Standard z-score normalisation: (x - mean) / std.
        Sensitive to outliers but widely used and interpretable.

    ``"robust"``
        Robust scaling: (x - median) / IQR.
        Less sensitive to outlier glucose spikes and extreme bolus values.

    ``"none"``
        Pass-through (no scaling).

    The scaler is fitted on training data only and the same parameters
    are applied to val/test splits to avoid leakage.

    Parameters
    ----------
    strategy : str
        One of ``"zscore"``, ``"robust"``, or ``"none"``.
    """

    def __init__(self, strategy: str = "zscore") -> None:
        if strategy not in {"zscore", "robust", "none"}:
            raise ValueError(f"Unknown normalization strategy: {strategy!r}. "
                             "Choose from 'zscore', 'robust', 'none'.")
        self.strategy = strategy
        self._center: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Fit scaler on X of shape [N, T, F] or [N, F].

        Parameters computed per feature (last axis).
        """
        flat = X.reshape(-1, X.shape[-1])
        if self.strategy == "zscore":
            self._center = flat.mean(axis=0)
            std = flat.std(axis=0)
            # Replace near-zero std with 1 to avoid division by zero
            self._scale = np.where(std < 1e-8, 1.0, std)
        elif self.strategy == "robust":
            self._center = np.median(flat, axis=0)
            q75 = np.percentile(flat, 75, axis=0)
            q25 = np.percentile(flat, 25, axis=0)
            iqr = q75 - q25
            self._scale = np.where(iqr < 1e-8, 1.0, iqr)
        else:
            # none – identity
            self._center = np.zeros(X.shape[-1], dtype=np.float32)
            self._scale = np.ones(X.shape[-1], dtype=np.float32)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling. X shape: [N, T, F] or [N, F]."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform().")
        return ((X - self._center) / self._scale).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo scaling."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before inverse_transform().")
        return (X * self._scale + self._center).astype(np.float32)

    def to_dict(self) -> dict:
        """Serialise scaler parameters for storage in model checkpoint."""
        return {
            "strategy": self.strategy,
            "center": self._center.tolist() if self._center is not None else None,
            "scale": self._scale.tolist() if self._scale is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        """Restore a scaler from a serialised dict."""
        scaler = cls(strategy=d["strategy"])
        if d.get("center") is not None:
            scaler._center = np.array(d["center"], dtype=np.float32)
            scaler._scale = np.array(d["scale"], dtype=np.float32)
            scaler._fitted = True
        return scaler


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Parquet support requires pyarrow. Install with `pip install iints-sdk-python35[research]`."
        ) from exc


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save dataset as parquet when available, or CSV as a fallback."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        save_parquet(df, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(
            "Parquet support requires pyarrow. Install with `pip install iints-sdk-python35[research]`."
        ) from exc


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset from parquet or CSV."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        return load_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def concat_runs(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True)


def basic_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for col in columns:
        if col in df.columns:
            stats[f"{col}_mean"] = float(df[col].mean())
            stats[f"{col}_std"] = float(df[col].std())
    return stats
