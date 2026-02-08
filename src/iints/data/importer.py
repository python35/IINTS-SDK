from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

import pandas as pd

from iints.data.ingestor import DataIngestor


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {col: _normalize_column(col) for col in columns}
    candidate_set = {_normalize_column(c) for c in candidates}
    for col, norm in normalized.items():
        if norm in candidate_set:
            return col
    return None


DEFAULT_MAPPINGS: Dict[str, Dict[str, List[str]]] = {
    "generic": {
        "timestamp": ["timestamp", "time", "datetime", "date", "eventtime", "device timestamp"],
        "glucose": ["glucose", "bg", "sgv", "sensorglucose", "glucosemgdl", "glucosevalue"],
        "carbs": ["carbs", "carb", "carbohydrates", "carbsg", "carbgrams"],
        "insulin": ["insulin", "insulinunits", "bolus", "basal", "totalinsulin"],
    },
    "dexcom": {
        "timestamp": ["timestamp", "eventtime", "device timestamp"],
        "glucose": ["glucose", "glucosevalue", "sgv", "sensorglucose"],
        "carbs": ["carbs", "carb", "carbohydrates"],
        "insulin": ["insulin", "insulinunits", "bolus", "basal"],
    },
    "libre": {
        "timestamp": ["timestamp", "device timestamp", "datetime", "time"],
        "glucose": ["glucose", "glucosevalue", "sensorglucose", "sgv"],
        "carbs": ["carbs", "carb", "carbohydrates"],
        "insulin": ["insulin", "insulinunits", "bolus", "basal"],
    },
}


@dataclass
class ImportResult:
    dataframe: pd.DataFrame
    scenario: Dict[str, Any]


def import_cgm_csv(
    path: str | Path,
    data_format: str = "generic",
    column_map: Optional[Dict[str, str]] = None,
    time_unit: str = "minutes",
    source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Import CGM data from CSV into the universal IINTS schema.
    """
    df = pd.read_csv(path)
    columns = list(df.columns)
    mapping = column_map or {}
    mapping = {k: v for k, v in mapping.items() if v}

    candidates = DEFAULT_MAPPINGS.get(data_format, DEFAULT_MAPPINGS["generic"])

    def resolve(key: str, required: bool = True) -> Optional[str]:
        if key in mapping:
            return mapping[key]
        col = _find_column(columns, candidates.get(key, []))
        if required and col is None:
            raise ValueError(f"Missing required column for '{key}'. Columns: {columns}")
        return col

    ts_col = resolve("timestamp", required=True)
    glucose_col = resolve("glucose", required=True)
    carbs_col = resolve("carbs", required=False)
    insulin_col = resolve("insulin", required=False)

    df = df.rename(
        columns={
            ts_col: "timestamp",
            glucose_col: "glucose",
            carbs_col: "carbs",
            insulin_col: "insulin",
        }
    )

    if "carbs" not in df.columns:
        df["carbs"] = 0.0
    if "insulin" not in df.columns:
        df["insulin"] = 0.0

    # Parse timestamps
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        ts = df["timestamp"]
    else:
        # Try datetime parsing, fallback to numeric
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().all():
            ts = None
        else:
            df["timestamp"] = ts

    if ts is not None:
        df["timestamp"] = (ts - ts.iloc[0]).dt.total_seconds() / 60.0
    else:
        # Assume numeric
        if time_unit == "seconds":
            df["timestamp"] = df["timestamp"].astype(float) / 60.0
        else:
            df["timestamp"] = df["timestamp"].astype(float)

    df["source"] = source or data_format
    ingestor = DataIngestor()
    ingestor._validate_schema(df, ingestor.UNIVERSAL_SCHEMA)
    return df[list(ingestor.UNIVERSAL_SCHEMA.keys())]


def scenario_from_dataframe(
    df: pd.DataFrame,
    scenario_name: str,
    scenario_version: str = "1.0",
    description: str = "Imported CGM scenario",
    carb_threshold: float = 0.1,
    absorption_delay_minutes: int = 10,
    duration_minutes: int = 60,
) -> Dict[str, Any]:
    stress_events = []
    if "carbs" in df.columns:
        for _, row in df[df["carbs"] > carb_threshold].iterrows():
            stress_events.append(
                {
                    "start_time": int(row["timestamp"]),
                    "event_type": "meal",
                    "value": float(row["carbs"]),
                    "absorption_delay_minutes": absorption_delay_minutes,
                    "duration": duration_minutes,
                }
            )

    return {
        "scenario_name": scenario_name,
        "scenario_version": scenario_version,
        "description": description,
        "stress_events": stress_events,
    }


def scenario_from_csv(
    path: str | Path,
    scenario_name: str = "Imported CGM Scenario",
    scenario_version: str = "1.0",
    data_format: str = "generic",
    column_map: Optional[Dict[str, str]] = None,
    time_unit: str = "minutes",
    carb_threshold: float = 0.1,
) -> ImportResult:
    df = import_cgm_csv(
        path,
        data_format=data_format,
        column_map=column_map,
        time_unit=time_unit,
    )
    scenario = scenario_from_dataframe(
        df,
        scenario_name=scenario_name,
        scenario_version=scenario_version,
        carb_threshold=carb_threshold,
    )
    return ImportResult(dataframe=df, scenario=scenario)


def export_standard_csv(df: pd.DataFrame, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path)
