from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


_DANGEROUS_PREFIXES = ("=", "+", "-", "@")


def sanitize_csv_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if stripped and stripped[0] in _DANGEROUS_PREFIXES:
        return "'" + value
    return value


def sanitize_csv_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: sanitize_csv_cell(value) for key, value in row.items()}


def sanitize_csv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sanitized = df.copy()
    for column in sanitized.columns:
        if pd.api.types.is_object_dtype(sanitized[column]) or pd.api.types.is_string_dtype(sanitized[column]):
            sanitized[column] = sanitized[column].map(sanitize_csv_cell)
    return sanitized
