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


def _parse_single_decimal(raw: Any) -> float:
    if raw is None or pd.isna(raw):
        return float("nan")
    text = str(raw).strip()
    if text in ("", "nan", "None"):
        return float("nan")
    has_comma = "," in text
    has_dot = "." in text
    if has_comma and has_dot:
        # Whichever separator appears last is the decimal mark.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        text = text.replace(",", ".")
    # If only '.' is present (e.g. "1.25" U or "0.85" U/h), preserve it as decimal.
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_decimal_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float, index=series.index)
    return series.map(_parse_single_decimal).astype(float)


def _find_carelink_header_index(lines: List[str]) -> tuple[int, str]:
    for idx, line in enumerate(lines):
        stripped = line.lstrip("\ufeff")
        if stripped.startswith("Index;Date;Time;"):
            return idx, ";"
        if stripped.startswith("Index,Date,Time,"):
            return idx, ","
    raise ValueError(
        "Could not find the CareLink event table header (Index;Date;Time;... or Index,Date,Time,...)."
    )


def _parse_carelink_metadata(lines: List[str], *, delimiter: str = ";") -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"raw_preamble_lines": [line for line in lines if line.strip()]}
    parsed_lines: List[List[str]] = []
    for line in lines:
        if not line.strip() or line.startswith("-------"):
            continue
        parts = next(csv.reader([line], delimiter=delimiter))
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


def _parse_carelink_timestamps(date_col: pd.Series, time_col: pd.Series) -> pd.Series:
    combined = date_col.fillna("") + " " + time_col.fillna("")
    for fmt in ("%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
        parsed = pd.to_datetime(combined, format=fmt, errors="coerce")
        if parsed.notna().any():
            return parsed
    return pd.to_datetime(combined, errors="coerce", dayfirst=True)


def _classify_carelink_section(banner_line: str, delimiter: str) -> str:
    lower = banner_line.lower()
    if "aggregated auto insulin" in lower:
        return "aggregated_auto_insulin"
    if f"{delimiter}sensor{delimiter}" in lower or lower.endswith(f"{delimiter}sensor"):
        return "sensor"
    return "pump"


def _read_carelink_event_frame(path: Union[str, Path]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    first_header_idx, delimiter = _find_carelink_header_index(lines)
    metadata = _parse_carelink_metadata(lines[:first_header_idx], delimiter=delimiter)

    # Split into explicit CareLink sections (Pump, Aggregated Auto Insulin Data, Sensor)
    # so daily aggregated totals are never mixed with 5-minute pump/sensor events.
    header_prefix = f"Index{delimiter}Date{delimiter}Time{delimiter}"
    section_blocks: List[tuple[str, List[str]]] = []
    current_section = "pump"
    active_lines: Optional[List[str]] = None

    for line in lines:
        stripped = line.lstrip("\ufeff")
        if stripped.startswith("-------"):
            if active_lines is not None and len(active_lines) > 1:
                section_blocks.append((current_section, active_lines))
                active_lines = None
            current_section = _classify_carelink_section(stripped, delimiter)
            continue
        if stripped.startswith(header_prefix):
            if active_lines is not None and len(active_lines) > 1:
                section_blocks.append((current_section, active_lines))
            active_lines = [stripped]
            continue
        if active_lines is not None and stripped.strip():
            active_lines.append(stripped)

    if active_lines is not None and len(active_lines) > 1:
        section_blocks.append((current_section, active_lines))

    if not section_blocks:
        raise ValueError("CareLink export contains no event rows after the header.")

    frames: List[pd.DataFrame] = []
    for section_name, block_lines in section_blocks:
        sec_df = pd.read_csv(
            io.StringIO("\n".join(block_lines)),
            sep=delimiter,
            dtype=str,
        )
        sec_df.columns = [str(col).strip() for col in sec_df.columns]
        sec_df["carelink_section"] = section_name
        frames.append(sec_df)

    df = pd.concat(frames, ignore_index=True, sort=False)
    for column in df.columns:
        if column != "carelink_section":
            df[column] = df[column].astype("string").str.strip()

    if {"Date", "Time"} - set(df.columns):
        raise ValueError("CareLink export is missing the expected Date/Time columns.")

    df = df[df["Date"].notna() & df["Time"].notna()].copy()
    dt = _parse_carelink_timestamps(df["Date"], df["Time"])
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
    tolerance = pd.to_timedelta(float(tolerance_minutes), unit="m")

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


def load_carelink_event_log(path: Union[str, Path]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load the raw Medtronic CareLink / MiniMed event log plus parsed metadata.

    Returns:
        A tuple of `(event_dataframe, metadata)` where the event dataframe contains
        the raw CareLink event table with a parsed `timestamp_dt` column.
    """
    return _read_carelink_event_frame(path)


def import_carelink_timeline(
    path: Union[str, Path],
    *,
    source: Optional[str] = None,
    event_tolerance_minutes: float = 7.5,
    fill_gaps: bool = False,
    grid_step_minutes: float = 5.0,
) -> pd.DataFrame:
    """
    Import a CareLink export into an enriched glucose timeline with real datetimes.

    Section-aware accounting:
    - Separates ``Aggregated Auto Insulin Data`` daily totals from 5-minute
      ``Pump`` / ``Sensor`` rows so daily totals are never booked as 5-minute
      bolus spikes and never double-counted with the programmed fallback
      ``Basal Rate (U/h)``.
    - When ``fill_gaps=True``, builds a continuous ``grid_step_minutes`` grid
      where sensor dropouts have ``glucose = NaN`` while every bolus and carb
      event is conserved (exact 0.0000 U bolus/carb conservation).

    Returns:
        DataFrame with columns:
        `timestamp_dt`, `timestamp`, `glucose`, `carbs`, `insulin`,
        `bolus_units`, `basal_units`, `basal_rate_u_per_hr`, `isig`,
        `bwz_active_insulin`, `source`
    """
    raw_df, _metadata = _read_carelink_event_frame(path)

    section_col = raw_df.get("carelink_section", pd.Series("pump", index=raw_df.index))
    agg_df = raw_df[section_col == "aggregated_auto_insulin"].copy()
    event_df = raw_df[section_col != "aggregated_auto_insulin"].copy()

    # Extract daily aggregated SmartGuard auto-insulin totals (U/day) by calendar date.
    daily_auto_by_date: Dict[Any, float] = {}
    if not agg_df.empty:
        agg_df["agg_units"] = _parse_decimal_series(
            agg_df.get("Bolus Volume Delivered (U)", pd.Series(dtype="string"))
        )
        valid_agg = agg_df[agg_df["agg_units"].notna() & (agg_df["agg_units"] > 0)].copy()
        if not valid_agg.empty:
            valid_agg["cal_date"] = valid_agg["timestamp_dt"].dt.date
            daily_auto_by_date = {
                d: float(v) for d, v in valid_agg.groupby("cal_date")["agg_units"].sum().items()
            }

    MGDL_PER_MMOLL = 18.0182
    sensor_glucose = _parse_decimal_series(event_df.get("Sensor Glucose (mg/dL)", pd.Series(dtype="string")))
    if sensor_glucose.notna().sum() == 0 and "Sensor Glucose (mmol/L)" in event_df.columns:
        sensor_glucose = _parse_decimal_series(event_df["Sensor Glucose (mmol/L)"]) * MGDL_PER_MMOLL
    bg_reading = _parse_decimal_series(event_df.get("BG Reading (mg/dL)", pd.Series(dtype="string")))
    if bg_reading.notna().sum() == 0 and "BG Reading (mmol/L)" in event_df.columns:
        bg_reading = _parse_decimal_series(event_df["BG Reading (mmol/L)"]) * MGDL_PER_MMOLL
    event_df["effective_glucose"] = sensor_glucose.combine_first(bg_reading)
    event_df["isig_val"] = _parse_decimal_series(event_df.get("ISIG Value", pd.Series(dtype="string")))
    event_df["bwz_iob_val"] = _parse_decimal_series(
        event_df.get("BWZ Active Insulin (U)", pd.Series(dtype="string"))
    )

    glucose_obs_df = (
        event_df.loc[event_df["effective_glucose"].notna(), ["timestamp_dt", "effective_glucose", "isig_val"]]
        .groupby("timestamp_dt", as_index=False)
        .last()
        .rename(columns={"effective_glucose": "glucose", "isig_val": "isig"})
        .sort_values("timestamp_dt")
        .reset_index(drop=True)
    )
    if glucose_obs_df.empty:
        raise ValueError("No glucose readings were found in the CareLink export.")

    carbs_df = event_df.loc[:, ["timestamp_dt"]].copy()
    carbs_df["carbs"] = _parse_decimal_series(event_df.get("BWZ Carb Input (grams)", pd.Series(dtype="string")))
    carbs_df = carbs_df[carbs_df["carbs"].notna() & (carbs_df["carbs"] > 0)].groupby("timestamp_dt", as_index=False).sum()

    all_bolus_df = event_df.loc[:, ["timestamp_dt"]].copy()
    all_bolus_df["bolus_units"] = _parse_decimal_series(
        event_df.get("Bolus Volume Delivered (U)", pd.Series(dtype="string"))
    )
    all_bolus_df["bolus_number"] = event_df.get("Bolus Number", pd.Series(dtype="string")).astype("string").fillna("")
    all_bolus_df["bolus_source"] = event_df.get("Bolus Source", pd.Series(dtype="string")).astype("string").fillna("")

    # Manual boluses + SmartGuard food/correction boluses (exclude per-5-min auto-basal here)
    bolus_df = all_bolus_df[
        all_bolus_df["bolus_units"].notna()
        & (all_bolus_df["bolus_units"] > 0)
        & (all_bolus_df["bolus_source"] != "CLOSED_LOOP_AUTO_INSULIN")
    ].copy()
    bolus_df["bolus_key"] = bolus_df["bolus_number"].where(
        bolus_df["bolus_number"].str.len() > 0,
        bolus_df["timestamp_dt"].astype(str),
    )
    bolus_df = (
        bolus_df.groupby(["timestamp_dt", "bolus_key"], as_index=False)["bolus_units"].max()
        .groupby("timestamp_dt", as_index=False)["bolus_units"]
        .sum()
    )

    # Per-5-min CLOSED_LOOP_AUTO_INSULIN rows inside Pump section (when no aggregated daily section exists)
    auto_basal_df = all_bolus_df[
        all_bolus_df["bolus_units"].notna()
        & (all_bolus_df["bolus_units"] > 0)
        & (all_bolus_df["bolus_source"] == "CLOSED_LOOP_AUTO_INSULIN")
    ].copy()
    auto_basal_df["auto_basal_units"] = auto_basal_df["bolus_units"]
    auto_basal_df = (
        auto_basal_df.groupby("timestamp_dt", as_index=False).agg({"auto_basal_units": "sum"})
        if not auto_basal_df.empty
        else pd.DataFrame(columns=["timestamp_dt", "auto_basal_units"])
    )

    bwz_iob_df = (
        event_df.loc[event_df["bwz_iob_val"].notna(), ["timestamp_dt", "bwz_iob_val"]]
        .groupby("timestamp_dt", as_index=False)
        .last()
        .rename(columns={"bwz_iob_val": "bwz_active_insulin"})
    )

    basal_df = event_df.loc[:, ["timestamp_dt"]].copy()
    basal_df["basal_rate_u_per_hr"] = _parse_decimal_series(event_df.get("Basal Rate (U/h)", pd.Series(dtype="string")))
    basal_df = (
        basal_df[basal_df["basal_rate_u_per_hr"].notna()]
        .groupby("timestamp_dt", as_index=False)
        .last()
        .sort_values("timestamp_dt")
    )

    if fill_gaps:
        # Build a continuous grid_step_minutes grid covering all glucose, bolus, and carb events
        all_ts_min = [glucose_obs_df["timestamp_dt"].iloc[0]]
        all_ts_max = [glucose_obs_df["timestamp_dt"].iloc[-1]]
        for extra_df in (bolus_df, carbs_df):
            if not extra_df.empty:
                all_ts_min.append(extra_df["timestamp_dt"].min())
                all_ts_max.append(extra_df["timestamp_dt"].max())
        grid_start = min(all_ts_min).floor(f"{int(round(grid_step_minutes))}min")
        grid_end = max(all_ts_max).ceil(f"{int(round(grid_step_minutes))}min")
        grid_times = pd.date_range(start=grid_start, end=grid_end, freq=f"{int(round(grid_step_minutes))}min")
        glucose_df = pd.DataFrame({"timestamp_dt": grid_times})
        # Nearest-align observed CGM readings onto the 5-min grid (within half a step + small margin)
        tol_grid = pd.to_timedelta(float(grid_step_minutes) * 0.55, unit="m")
        glucose_df = pd.merge_asof(
            glucose_df,
            glucose_obs_df,
            on="timestamp_dt",
            direction="nearest",
            tolerance=tol_grid,
        )
        attach_tol = max(float(event_tolerance_minutes), float(grid_step_minutes))
    else:
        glucose_df = glucose_obs_df.copy()
        attach_tol = float(event_tolerance_minutes)

    glucose_df = glucose_df.sort_values("timestamp_dt").reset_index(drop=True)
    glucose_df = _attach_events_to_glucose_timeline(
        glucose_df,
        carbs_df,
        value_column="carbs",
        output_column="carbs",
        tolerance_minutes=attach_tol,
    )
    glucose_df = _attach_events_to_glucose_timeline(
        glucose_df,
        bolus_df,
        value_column="bolus_units",
        output_column="bolus_units",
        tolerance_minutes=attach_tol,
    )
    glucose_df = _attach_events_to_glucose_timeline(
        glucose_df,
        auto_basal_df,
        value_column="auto_basal_units",
        output_column="auto_basal_units",
        tolerance_minutes=attach_tol,
    )
    if not bwz_iob_df.empty:
        glucose_df = pd.merge_asof(
            glucose_df,
            bwz_iob_df.sort_values("timestamp_dt"),
            on="timestamp_dt",
            direction="nearest",
            tolerance=pd.to_timedelta(float(attach_tol), unit="m"),
        )
    else:
        glucose_df["bwz_active_insulin"] = float("nan")

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
    if fill_gaps:
        capped_step_minutes = pd.Series(float(grid_step_minutes), index=glucose_df.index)
    else:
        typical_step = raw_step_minutes[(raw_step_minutes > 0) & (raw_step_minutes <= 15)].median()
        if pd.isna(typical_step):
            typical_step = 5.0
        capped_step_minutes = raw_step_minutes.fillna(float(typical_step)).clip(
            lower=0.0,
            upper=max(float(typical_step) * 1.5, 15.0),
        )

    glucose_df["carbs"] = glucose_df.get("carbs", 0.0)
    glucose_df["bolus_units"] = glucose_df.get("bolus_units", 0.0)
    auto_basal_series = glucose_df.get("auto_basal_units", pd.Series(0.0, index=glucose_df.index)).fillna(0.0)

    # Prefer section-separated daily Aggregated Auto Insulin rate over the programmed
    # fallback Basal Rate (U/h) whenever Aggregated Auto Insulin Data is present.
    if daily_auto_by_date:
        cal_dates = timestamp_series.dt.date
        agg_daily_rate = cal_dates.map(lambda d: daily_auto_by_date.get(d, float("nan")) / 24.0)
        has_agg = agg_daily_rate.notna()
        glucose_df["basal_rate_u_per_hr"] = glucose_df["basal_rate_u_per_hr"].where(~has_agg, agg_daily_rate)

    manual_basal_units = glucose_df["basal_rate_u_per_hr"] * capped_step_minutes.div(60.0)
    has_auto_basal = auto_basal_series > 0.0
    glucose_df["basal_units"] = manual_basal_units.where(~has_auto_basal, auto_basal_series)
    safe_step_hr = capped_step_minutes.clip(lower=1e-6).div(60.0)
    glucose_df["basal_rate_u_per_hr"] = glucose_df["basal_rate_u_per_hr"].where(
        ~has_auto_basal, auto_basal_series.div(safe_step_hr)
    )
    glucose_df["insulin"] = glucose_df["bolus_units"].fillna(0.0) + glucose_df["basal_units"].fillna(0.0)
    glucose_df["timestamp"] = (
        glucose_df["timestamp_dt"] - glucose_df["timestamp_dt"].iloc[0]
    ).dt.total_seconds() / 60.0
    glucose_df["source"] = source or "carelink_minimed"

    enriched = glucose_df[
        [
            "timestamp_dt",
            "timestamp",
            "glucose",
            "carbs",
            "insulin",
            "bolus_units",
            "basal_units",
            "basal_rate_u_per_hr",
            "isig",
            "bwz_active_insulin",
            "source",
        ]
    ].copy()
    enriched["carbs"] = enriched["carbs"].fillna(0.0)
    enriched["insulin"] = enriched["insulin"].fillna(0.0)
    enriched["bolus_units"] = enriched["bolus_units"].fillna(0.0)
    enriched["basal_units"] = enriched["basal_units"].fillna(0.0)
    return enriched


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
    enriched = import_carelink_timeline(
        path,
        source=source,
        event_tolerance_minutes=event_tolerance_minutes,
    )
    standard = enriched[["timestamp", "glucose", "carbs", "insulin", "source"]].copy()
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
