from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import io
import re
import sys
import csv

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
    "carelink": {
        "timestamp": ["date", "time"],
        "glucose": ["sensor glucose (mg/dl)", "bg reading (mg/dl)"],
        "carbs": ["bwz carb input (grams)"],
        "insulin": ["bolus volume delivered (u)", "basal rate (u/h)"],
    },
}

IMPORT_FORMAT_SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "generic": {
        "required": ["timestamp", "glucose"],
        "optional": ["carbs", "insulin"],
    },
    "dexcom": {
        "required": ["timestamp", "glucose"],
        "optional": ["carbs", "insulin"],
    },
    "libre": {
        "required": ["timestamp", "glucose"],
        "optional": ["carbs", "insulin"],
    },
    "carelink": {
        "required": ["timestamp", "glucose"],
        "optional": ["carbs", "insulin"],
    },
}


@dataclass
class ImportResult:
    dataframe: pd.DataFrame
    scenario: Dict[str, Any]


def _parse_decimal_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.isin(["", "nan", "None"]))
    cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _find_carelink_header_index(lines: List[str]) -> int:
    for idx, line in enumerate(lines):
        if line.startswith("Index;Date;Time;"):
            return idx
    raise ValueError("Could not find the CareLink event table header (Index;Date;Time;...).")


def _parse_carelink_metadata(lines: List[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"raw_preamble_lines": [line for line in lines if line.strip()]}
    parsed_lines: List[List[str]] = []
    for line in lines:
        if not line.strip() or line.startswith("-------"):
            continue
        parts = next(csv.reader([line], delimiter=";"))
        cleaned = [part.strip().strip('"') for part in parts]
        parsed_lines.append(cleaned)
        for i in range(6, len(cleaned) - 1, 2):
            key = cleaned[i]
            value = cleaned[i + 1]
            if not key:
                continue
            metadata[key] = value

    if len(parsed_lines) >= 2:
        line0 = parsed_lines[0]
        line1 = parsed_lines[1]
        for idx in range(min(6, len(line0), len(line1))):
            key = line0[idx]
            value = line1[idx]
            if key:
                metadata[key] = value

    if parsed_lines:
        line0 = parsed_lines[0]
        if len(line0) >= 2 and line0[0] and line0[1] and "First Name" not in metadata:
            metadata[line0[0]] = line0[1]

    patient_name = " ".join(
        part for part in [metadata.get("First Name", ""), metadata.get("Last Name", "")] if part
    ).strip()
    if patient_name:
        metadata["patient_name"] = patient_name
    return metadata


def _read_carelink_event_frame(path: Union[str, Path]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header_idx = _find_carelink_header_index(lines)
    metadata = _parse_carelink_metadata(lines[:header_idx])

    df = pd.read_csv(
        file_path,
        sep=";",
        skiprows=header_idx,
        encoding="utf-8-sig",
        dtype=str,
    )
    df.columns = [str(col).strip() for col in df.columns]
    for column in df.columns:
        df[column] = df[column].astype("string").str.strip()

    if {"Date", "Time"} - set(df.columns):
        raise ValueError("CareLink export is missing the expected Date/Time columns.")

    df = df[df["Date"].notna() & df["Time"].notna()].copy()
    dt = pd.to_datetime(
        df["Date"].fillna("") + " " + df["Time"].fillna(""),
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
    )
    df = df[dt.notna()].copy()
    df["timestamp_dt"] = dt[dt.notna()]
    df = df.sort_values("timestamp_dt").reset_index(drop=True)
    return df, metadata


def _attach_events_to_glucose_timeline(
    glucose_df: pd.DataFrame,
    event_df: pd.DataFrame,
    *,
    value_column: str,
    output_column: str,
    tolerance_minutes: float,
) -> pd.DataFrame:
    if event_df.empty:
        glucose_df[output_column] = glucose_df.get(output_column, 0.0)
        return glucose_df

    glucose_df = glucose_df.copy()
    glucose_df[output_column] = glucose_df.get(output_column, 0.0)
    timestamps = list(glucose_df["timestamp_dt"])
    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    for row in event_df.itertuples(index=False):
        event_time = getattr(row, "timestamp_dt")
        amount = getattr(row, value_column)
        if pd.isna(amount):
            continue

        insert_at = int(glucose_df["timestamp_dt"].searchsorted(event_time))
        candidates: List[int] = []
        if insert_at < len(timestamps):
            candidates.append(insert_at)
        if insert_at > 0:
            candidates.append(insert_at - 1)
        if not candidates:
            continue

        nearest_idx = min(candidates, key=lambda idx: abs(timestamps[idx] - event_time))
        if abs(timestamps[nearest_idx] - event_time) <= tolerance:
            current_value = pd.to_numeric(
                pd.Series([glucose_df.at[nearest_idx, output_column]]),
                errors="coerce",
            ).fillna(0.0).iloc[0]
            glucose_df.at[nearest_idx, output_column] = float(current_value) + float(amount)

    return glucose_df


def summarize_carelink_csv(path: Union[str, Path]) -> Dict[str, Any]:
    raw_df, metadata = _read_carelink_event_frame(path)
    summary = {
        "patient_name": metadata.get("patient_name", ""),
        "start_date": metadata.get("Start Date", ""),
        "end_date": metadata.get("End Date", ""),
        "device": metadata.get("Device", ""),
        "cgm": metadata.get("CGM", ""),
        "raw_event_rows": int(len(raw_df)),
        "sensor_glucose_rows": int(_parse_decimal_series(raw_df.get("Sensor Glucose (mg/dL)", pd.Series(dtype="string"))).notna().sum()),
        "bg_reading_rows": int(_parse_decimal_series(raw_df.get("BG Reading (mg/dL)", pd.Series(dtype="string"))).notna().sum()),
        "bolus_rows": int(_parse_decimal_series(raw_df.get("Bolus Volume Delivered (U)", pd.Series(dtype="string"))).fillna(0).gt(0).sum()),
        "meal_rows": int(_parse_decimal_series(raw_df.get("BWZ Carb Input (grams)", pd.Series(dtype="string"))).fillna(0).gt(0).sum()),
        "alert_rows": int(raw_df.get("Alert", pd.Series(dtype="string")).fillna("").astype(str).str.len().gt(0).sum()),
        "sensor_exception_rows": int(raw_df.get("Sensor Exception", pd.Series(dtype="string")).fillna("").astype(str).str.len().gt(0).sum()),
    }
    return summary


def import_carelink_csv(
    path: Union[str, Path],
    *,
    source: Optional[str] = None,
    event_tolerance_minutes: float = 7.5,
) -> pd.DataFrame:
    """
    Import a Medtronic CareLink / MiniMed export into the universal IINTS schema.

    The CareLink export is an event log, not a simple one-row-per-reading CSV.
    This parser:
    - skips the CareLink metadata preamble
    - extracts glucose values from sensor glucose and SMBG rows
    - aligns carb and bolus events to the nearest CGM timestamp
    - estimates basal insulin between glucose samples from the reported basal rate
    """
    raw_df, _metadata = _read_carelink_event_frame(path)

    sensor_glucose = _parse_decimal_series(raw_df.get("Sensor Glucose (mg/dL)", pd.Series(dtype="string")))
    bg_reading = _parse_decimal_series(raw_df.get("BG Reading (mg/dL)", pd.Series(dtype="string")))
    raw_df["effective_glucose"] = sensor_glucose.combine_first(bg_reading)

    glucose_df = (
        raw_df.loc[raw_df["effective_glucose"].notna(), ["timestamp_dt", "effective_glucose"]]
        .groupby("timestamp_dt", as_index=False)
        .last()
        .rename(columns={"effective_glucose": "glucose"})
    )
    if glucose_df.empty:
        raise ValueError("No glucose readings were found in the CareLink export.")

    carbs_df = raw_df.loc[:, ["timestamp_dt"]].copy()
    carbs_df["carbs"] = _parse_decimal_series(raw_df.get("BWZ Carb Input (grams)", pd.Series(dtype="string")))
    carbs_df = carbs_df[carbs_df["carbs"].notna() & (carbs_df["carbs"] > 0)].groupby("timestamp_dt", as_index=False).sum()

    bolus_df = raw_df.loc[:, ["timestamp_dt"]].copy()
    bolus_df["bolus_units"] = _parse_decimal_series(raw_df.get("Bolus Volume Delivered (U)", pd.Series(dtype="string")))
    bolus_df["bolus_number"] = raw_df.get("Bolus Number", pd.Series(dtype="string")).astype("string").fillna("")
    bolus_df["bolus_source"] = raw_df.get("Bolus Source", pd.Series(dtype="string")).astype("string").fillna("")
    bolus_df = bolus_df[
        bolus_df["bolus_units"].notna()
        & (bolus_df["bolus_units"] > 0)
        & (bolus_df["bolus_source"] != "CLOSED_LOOP_AUTO_INSULIN")
    ].copy()
    bolus_df["bolus_key"] = bolus_df["bolus_number"].where(bolus_df["bolus_number"].str.len() > 0, bolus_df["timestamp_dt"].astype(str))
    bolus_df = (
        bolus_df.groupby(["timestamp_dt", "bolus_key"], as_index=False)["bolus_units"].max()
        .groupby("timestamp_dt", as_index=False)["bolus_units"]
        .sum()
    )

    basal_df = raw_df.loc[:, ["timestamp_dt"]].copy()
    basal_df["basal_rate_u_per_hr"] = _parse_decimal_series(raw_df.get("Basal Rate (U/h)", pd.Series(dtype="string")))
    basal_df = (
        basal_df[basal_df["basal_rate_u_per_hr"].notna()]
        .groupby("timestamp_dt", as_index=False)
        .last()
        .sort_values("timestamp_dt")
    )

    glucose_df = glucose_df.sort_values("timestamp_dt").reset_index(drop=True)
    glucose_df = _attach_events_to_glucose_timeline(
        glucose_df,
        carbs_df,
        value_column="carbs",
        output_column="carbs",
        tolerance_minutes=event_tolerance_minutes,
    )
    glucose_df = _attach_events_to_glucose_timeline(
        glucose_df,
        bolus_df,
        value_column="bolus_units",
        output_column="bolus_units",
        tolerance_minutes=event_tolerance_minutes,
    )

    if not basal_df.empty:
        glucose_df = pd.merge_asof(
            glucose_df.sort_values("timestamp_dt"),
            basal_df,
            on="timestamp_dt",
            direction="backward",
        )
        glucose_df["basal_rate_u_per_hr"] = glucose_df["basal_rate_u_per_hr"].ffill().fillna(0.0)
    else:
        glucose_df["basal_rate_u_per_hr"] = 0.0

    timestamp_series = pd.to_datetime(glucose_df["timestamp_dt"], errors="coerce")
    raw_step_minutes = timestamp_series.diff().dt.total_seconds().div(60.0)
    typical_step = raw_step_minutes[(raw_step_minutes > 0) & (raw_step_minutes <= 15)].median()
    if pd.isna(typical_step):
        typical_step = 5.0
    capped_step_minutes = raw_step_minutes.fillna(float(typical_step)).clip(lower=0.0, upper=max(float(typical_step) * 1.5, 15.0))
    glucose_df["basal_units"] = glucose_df["basal_rate_u_per_hr"] * capped_step_minutes.div(60.0)
    bolus_series = glucose_df["bolus_units"] if "bolus_units" in glucose_df.columns else pd.Series(0.0, index=glucose_df.index)
    glucose_df["insulin"] = bolus_series.fillna(0.0) + glucose_df["basal_units"].fillna(0.0)
    glucose_df["timestamp"] = (
        glucose_df["timestamp_dt"] - glucose_df["timestamp_dt"].iloc[0]
    ).dt.total_seconds() / 60.0
    glucose_df["source"] = source or "carelink_minimed"

    standard = glucose_df[["timestamp", "glucose", "carbs", "insulin", "source"]].copy()
    standard["carbs"] = standard["carbs"].fillna(0.0)
    standard["insulin"] = standard["insulin"].fillna(0.0)

    ingestor = DataIngestor()
    ingestor._validate_schema(standard, ingestor.UNIVERSAL_SCHEMA)
    return standard


def guess_column_mapping(columns: Iterable[str], data_format: str = "generic") -> Dict[str, Optional[str]]:
    candidates = DEFAULT_MAPPINGS.get(data_format, DEFAULT_MAPPINGS["generic"])
    return {
        "timestamp": _find_column(columns, candidates.get("timestamp", [])),
        "glucose": _find_column(columns, candidates.get("glucose", [])),
        "carbs": _find_column(columns, candidates.get("carbs", [])),
        "insulin": _find_column(columns, candidates.get("insulin", [])),
    }


def validate_import_schema(
    columns: Iterable[str],
    data_format: str,
    column_map: Optional[Dict[str, str]] = None,
) -> None:
    schema = IMPORT_FORMAT_SCHEMAS.get(data_format, IMPORT_FORMAT_SCHEMAS["generic"])
    candidates = DEFAULT_MAPPINGS.get(data_format, DEFAULT_MAPPINGS["generic"])
    mapping = column_map or {}

    missing: List[str] = []
    for key in schema["required"]:
        if key in mapping:
            continue
        found = _find_column(columns, candidates.get(key, []))
        if found is None:
            missing.append(key)

    if missing:
        raise ValueError(
            f"Missing required columns for format '{data_format}': {', '.join(missing)}. "
            f"Columns: {list(columns)}"
        )


def import_cgm_dataframe(
    df: pd.DataFrame,
    data_format: str = "generic",
    column_map: Optional[Dict[str, str]] = None,
    time_unit: str = "minutes",
    source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Import CGM data from an in-memory DataFrame into the universal IINTS schema.
    """
    columns = list(df.columns)
    mapping = column_map or {}
    mapping = {k: v for k, v in mapping.items() if v}

    candidates = DEFAULT_MAPPINGS.get(data_format, DEFAULT_MAPPINGS["generic"])

    validate_import_schema(columns, data_format=data_format, column_map=mapping)

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
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        # Assume numeric (minutes or seconds)
        if time_unit == "seconds":
            df["timestamp"] = df["timestamp"].astype(float) / 60.0
        else:
            df["timestamp"] = df["timestamp"].astype(float)
    elif pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        ts = df["timestamp"]
        df["timestamp"] = (ts - ts.iloc[0]).dt.total_seconds() / 60.0
    else:
        # Try datetime parsing, fallback to numeric
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().all():
            if time_unit == "seconds":
                df["timestamp"] = df["timestamp"].astype(float) / 60.0
            else:
                df["timestamp"] = df["timestamp"].astype(float)
        else:
            df["timestamp"] = (ts - ts.iloc[0]).dt.total_seconds() / 60.0

    df["source"] = source or data_format
    ingestor = DataIngestor()
    ingestor._validate_schema(df, ingestor.UNIVERSAL_SCHEMA)
    return df[list(ingestor.UNIVERSAL_SCHEMA.keys())]


def import_cgm_csv(
    path: Union[str, Path],
    data_format: str = "generic",
    column_map: Optional[Dict[str, str]] = None,
    time_unit: str = "minutes",
    source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Import CGM data from CSV into the universal IINTS schema.
    """
    if data_format == "carelink":
        return import_carelink_csv(path, source=source)
    df = pd.read_csv(path)
    return import_cgm_dataframe(
        df,
        data_format=data_format,
        column_map=column_map,
        time_unit=time_unit,
        source=source,
    )


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
    path: Union[str, Path],
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


def export_standard_csv(df: pd.DataFrame, output_path: Union[str, Path]) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path)


def _read_demo_csv_text() -> str:
    if sys.version_info >= (3, 9):
        from importlib.resources import files
        return files("iints.data.demo").joinpath("demo_cgm.csv").read_text()
    from importlib import resources
    return resources.read_text("iints.data.demo", "demo_cgm.csv")


def load_demo_dataframe() -> pd.DataFrame:
    text = _read_demo_csv_text()
    return pd.read_csv(io.StringIO(text))


def export_demo_csv(output_path: Union[str, Path]) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_read_demo_csv_text())
    return str(output_path)
