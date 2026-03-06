from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .contracts import ModelReadyContract, parse_contract
from .runner import ContractRunner, ValidationResult


ContractInput = Union[ModelReadyContract, Mapping[str, Any]]


@dataclass(frozen=True)
class SyntheticMirrorArtifact:
    summary: Dict[str, Any]
    validation: ValidationResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "validation": self.validation.to_dict(),
        }


def _resolve_contract(contract: ContractInput) -> ModelReadyContract:
    if isinstance(contract, ModelReadyContract):
        return contract
    return parse_contract(dict(contract))


def _collect_required_columns(contract: ModelReadyContract) -> List[str]:
    cols: set[str] = set()
    for stream in contract.streams:
        required = stream.metadata.get("required_columns", [])
        if isinstance(required, list):
            for column in required:
                name = str(column).strip()
                if name:
                    cols.add(name)
    for process in contract.processes:
        token = process.input_stream.split(".")[-1].strip()
        if token:
            cols.add(token)
    return sorted(cols)


def _collect_contract_ranges(contract: ModelReadyContract) -> Dict[str, Dict[str, float]]:
    ranges: Dict[str, Dict[str, float]] = {}
    for stream in contract.streams:
        raw_ranges = stream.metadata.get("ranges", {})
        if not isinstance(raw_ranges, dict):
            continue
        for column, bounds in raw_ranges.items():
            if not isinstance(bounds, dict):
                continue
            parsed: Dict[str, float] = {}
            if "min" in bounds:
                parsed["min"] = float(bounds["min"])
            if "max" in bounds:
                parsed["max"] = float(bounds["max"])
            if parsed:
                ranges[str(column).strip()] = parsed
    return ranges


def _collect_contract_types(contract: ModelReadyContract) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for stream in contract.streams:
        raw_types = stream.metadata.get("column_types", {})
        if not isinstance(raw_types, dict):
            continue
        for column, expected in raw_types.items():
            types[str(column).strip()] = str(expected).strip().lower()
    return types


def _is_integer_type(expected: str) -> bool:
    return expected in {"int", "integer"}


def _is_numeric_type(expected: str) -> bool:
    return expected in {"float", "number", "numeric", "int", "integer"}


def _is_datetime_type(expected: str) -> bool:
    return expected in {"datetime", "timestamp"}


def _build_timeline(series: pd.Series, n_rows: int) -> List[str]:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    valid = parsed.dropna()
    if len(valid) < 2:
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        step = pd.Timedelta(minutes=5)
    else:
        ordered = valid.sort_values()
        deltas = ordered.diff().dropna()
        median_delta = deltas.median() if not deltas.empty else pd.Timedelta(minutes=5)
        if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
            median_delta = pd.Timedelta(minutes=5)
        start = ordered.iloc[0]
        step = median_delta
    return [(start + idx * step).strftime("%Y-%m-%dT%H:%M:%SZ") for idx in range(n_rows)]


def _apply_contract_ranges(df: pd.DataFrame, ranges: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for column, bounds in ranges.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if "min" in bounds:
            values = values.clip(lower=bounds["min"])
        if "max" in bounds:
            values = values.clip(upper=bounds["max"])
        out[column] = values
    return out


def _inject_missing_columns(
    df: pd.DataFrame,
    *,
    required_columns: List[str],
    contract_types: Dict[str, str],
    contract_ranges: Dict[str, Dict[str, float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    n_rows = len(out)
    for column in required_columns:
        if column in out.columns:
            continue
        expected = contract_types.get(column, "float")
        bounds = contract_ranges.get(column, {})
        lower = bounds.get("min", 0.0)
        upper = bounds.get("max", max(lower + 1.0, 100.0))

        if _is_datetime_type(expected) or column.lower() == "timestamp":
            out[column] = _build_timeline(pd.Series(dtype="datetime64[ns, UTC]"), n_rows)
            continue

        if _is_numeric_type(expected):
            values = rng.uniform(lower, upper, size=n_rows)
            if _is_integer_type(expected):
                out[column] = np.rint(values).astype(int)
            else:
                out[column] = values
            continue

        out[column] = "synthetic"
    return out


def _build_summary(
    source_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    seed: int,
    noise_scale: float,
    timestamp_column: Optional[str],
) -> Dict[str, Any]:
    shared_numeric: List[str] = []
    for column in source_df.columns:
        if column in synthetic_df.columns and pd.api.types.is_numeric_dtype(source_df[column]):
            shared_numeric.append(column)

    numeric_stats: Dict[str, Dict[str, float]] = {}
    for column in shared_numeric:
        src = pd.to_numeric(source_df[column], errors="coerce")
        syn = pd.to_numeric(synthetic_df[column], errors="coerce")
        src_mean = float(np.nanmean(src)) if np.isfinite(np.nanmean(src)) else 0.0
        syn_mean = float(np.nanmean(syn)) if np.isfinite(np.nanmean(syn)) else 0.0
        src_std = float(np.nanstd(src)) if np.isfinite(np.nanstd(src)) else 0.0
        syn_std = float(np.nanstd(syn)) if np.isfinite(np.nanstd(syn)) else 0.0
        numeric_stats[column] = {
            "source_mean": round(src_mean, 4),
            "synthetic_mean": round(syn_mean, 4),
            "source_std": round(src_std, 4),
            "synthetic_std": round(syn_std, 4),
            "delta_mean": round(syn_mean - src_mean, 4),
        }

    return {
        "source_rows": int(len(source_df)),
        "synthetic_rows": int(len(synthetic_df)),
        "seed": int(seed),
        "noise_scale": float(noise_scale),
        "timestamp_column": timestamp_column,
        "numeric_columns_compared": shared_numeric,
        "numeric_stats": numeric_stats,
    }


def generate_synthetic_mirror(
    source_df: pd.DataFrame,
    contract: ContractInput,
    *,
    rows: Optional[int] = None,
    seed: int = 42,
    noise_scale: float = 0.05,
    timestamp_column: str = "timestamp",
) -> Tuple[pd.DataFrame, SyntheticMirrorArtifact]:
    """
    Generate a privacy-safe synthetic mirror dataset that preserves schema and broad statistics.
    """
    if source_df.empty:
        raise ValueError("source_df must not be empty")
    if noise_scale < 0:
        raise ValueError("noise_scale must be >= 0")

    contract_obj = _resolve_contract(contract)
    n_rows = int(rows) if rows is not None else int(len(source_df))
    if n_rows <= 0:
        raise ValueError("rows must be > 0")

    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, len(source_df), size=n_rows)
    synthetic = source_df.iloc[sampled_idx].reset_index(drop=True).copy(deep=True)

    for column in synthetic.columns:
        if not pd.api.types.is_numeric_dtype(source_df[column]):
            continue
        source_values = pd.to_numeric(source_df[column], errors="coerce")
        synth_values = pd.to_numeric(synthetic[column], errors="coerce").astype(float)
        std = float(np.nanstd(source_values))
        if np.isfinite(std) and std > 0 and noise_scale > 0:
            synth_values = synth_values + rng.normal(0.0, std * noise_scale, size=n_rows)
        if pd.api.types.is_integer_dtype(source_df[column]):
            synthetic[column] = np.rint(synth_values).astype(int)
        else:
            synthetic[column] = synth_values

    inferred_timestamp: Optional[str] = None
    if timestamp_column in synthetic.columns:
        synthetic[timestamp_column] = _build_timeline(source_df[timestamp_column], n_rows)
        inferred_timestamp = timestamp_column

    required_columns = _collect_required_columns(contract_obj)
    contract_ranges = _collect_contract_ranges(contract_obj)
    contract_types = _collect_contract_types(contract_obj)

    synthetic = _inject_missing_columns(
        synthetic,
        required_columns=required_columns,
        contract_types=contract_types,
        contract_ranges=contract_ranges,
        rng=rng,
    )
    synthetic = _apply_contract_ranges(synthetic, contract_ranges)

    if inferred_timestamp is None and "timestamp" in synthetic.columns:
        synthetic["timestamp"] = _build_timeline(pd.Series(dtype="datetime64[ns, UTC]"), n_rows)
        inferred_timestamp = "timestamp"

    validation = ContractRunner(contract_obj).run(synthetic, apply_builtin_transforms=False)
    summary = _build_summary(
        source_df=source_df,
        synthetic_df=synthetic,
        seed=seed,
        noise_scale=noise_scale,
        timestamp_column=inferred_timestamp,
    )
    return synthetic, SyntheticMirrorArtifact(summary=summary, validation=validation)
