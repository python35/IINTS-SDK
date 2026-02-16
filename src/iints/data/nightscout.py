from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import asyncio

import pandas as pd

from iints.data.importer import ImportResult, import_cgm_dataframe, scenario_from_dataframe


@dataclass
class NightscoutConfig:
    url: str
    api_secret: Optional[str] = None
    token: Optional[str] = None
    start: Optional[str] = None  # ISO string or date-like
    end: Optional[str] = None
    limit: Optional[int] = None


def _require_nightscout():
    try:
        import py_nightscout as nightscout  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "py-nightscout is required. Install with `pip install iints-sdk-python35[nightscout]`."
        ) from exc
    return nightscout


async def _fetch_entries_async(config: NightscoutConfig) -> List[Any]:
    nightscout = _require_nightscout()
    api = nightscout.Api(
        config.url,
        api_secret=config.api_secret,
        token=config.token,
    )
    entries = await api.get_sgvs()
    return list(entries or [])


def _entry_get(entry: Any, key: str) -> Any:
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def _entries_to_dataframe(entries: Iterable[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        glucose = _entry_get(entry, "sgv") or _entry_get(entry, "glucose")
        if glucose is None:
            continue
        ts_raw = (
            _entry_get(entry, "date")
            or _entry_get(entry, "dateString")
            or _entry_get(entry, "timestamp")
        )
        if ts_raw is None:
            continue
        if isinstance(ts_raw, (int, float)):
            timestamp = pd.to_datetime(ts_raw, unit="ms", errors="coerce")
        else:
            timestamp = pd.to_datetime(ts_raw, errors="coerce")
        if pd.isna(timestamp):
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "glucose": float(glucose),
                "carbs": 0.0,
                "insulin": 0.0,
            }
        )
    return pd.DataFrame(rows)


def fetch_nightscout_dataframe(config: NightscoutConfig) -> pd.DataFrame:
    entries = asyncio.run(_fetch_entries_async(config))
    df = _entries_to_dataframe(entries)
    if df.empty:
        return df

    if config.start:
        start_ts = pd.to_datetime(config.start, errors="coerce")
        if pd.notna(start_ts):
            df = df[df["timestamp"] >= start_ts]
    if config.end:
        end_ts = pd.to_datetime(config.end, errors="coerce")
        if pd.notna(end_ts):
            df = df[df["timestamp"] <= end_ts]
    if config.limit:
        df = df.head(int(config.limit))
    return df.reset_index(drop=True)


def import_nightscout(
    config: NightscoutConfig,
    scenario_name: str = "Nightscout Import",
    scenario_version: str = "1.0",
    carb_threshold: float = 0.1,
) -> ImportResult:
    df = fetch_nightscout_dataframe(config)
    if df.empty:
        raise ValueError("No Nightscout CGM entries found for the given parameters.")

    standard_df = import_cgm_dataframe(
        df,
        data_format="generic",
        time_unit="minutes",
        source="nightscout",
    )
    scenario = scenario_from_dataframe(
        standard_df,
        scenario_name=scenario_name,
        scenario_version=scenario_version,
        carb_threshold=carb_threshold,
    )
    return ImportResult(dataframe=standard_df, scenario=scenario)
