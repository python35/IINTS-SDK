"""Section-aware Medtronic CareLink (MiniMed 780G + Simplera) CSV parser.

Splits the three CareLink blocks (``Pump``, ``Aggregated Auto Insulin Data``,
and ``Sensor``), reconciles daily SmartGuard auto-insulin without double-counting
the programmed fallback basal profile, and inserts 5-minute placeholder rows
across CGM gaps so boluses and carbs delivered during sensor swaps are conserved
exactly (delta = 0.000 U).

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

__all__ = [
    "CareLinkSections",
    "SECTION_PUMP",
    "SECTION_AGGREGATED",
    "SECTION_SENSOR",
    "read_sections",
    "build_timeline",
    "reconcile_report",
]

SECTION_PUMP = "pump"
SECTION_AGGREGATED = "aggregated_auto_insulin"
SECTION_SENSOR = "sensor"

_GRID_MINUTES = 5.0
_ALIGN_TOLERANCE_MIN = 5.0


@dataclass(frozen=True)
class CareLinkSections:
    """The raw blocks of one CareLink export, parsed but not yet reconciled."""

    pump: pd.DataFrame
    aggregated: pd.DataFrame
    sensor: pd.DataFrame
    device_header: str

    def summary(self) -> Dict[str, int]:
        return {
            "pump_rows": len(self.pump),
            "aggregated_days": len(self.aggregated),
            "sensor_rows": len(self.sensor),
        }


def _classify(label: str) -> str:
    low = label.lower()
    if "aggregated auto insulin" in low:
        return SECTION_AGGREGATED
    if ";sensor;" in low or ",sensor," in low:
        return SECTION_SENSOR
    return SECTION_PUMP


def _timestamps(frame: pd.DataFrame) -> pd.Series:
    raw = frame["Date"].astype(str).str.strip() + " " + frame["Time"].astype(str).str.strip()
    ts = pd.to_datetime(raw, format="%Y/%m/%d %H:%M:%S", errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(raw, errors="coerce")
    return ts


def read_sections(path: str | Path) -> CareLinkSections:
    """Split a CareLink export into its blocks and parse each one independently."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    header_rows = [i for i, line in enumerate(lines) if line.startswith("Index;") or line.startswith("Index,")]
    if not header_rows:
        raise ValueError(f"{path} has no 'Index;' or 'Index,' section header - not a CareLink export")

    sep = ";" if lines[header_rows[0]].startswith("Index;") else ","
    decimal = "," if sep == ";" else "."

    blocks: Dict[str, pd.DataFrame] = {}
    for position, start in enumerate(header_rows):
        label = lines[start - 1] if start else ""
        kind = _classify(label)
        end = header_rows[position + 1] - 1 if position + 1 < len(header_rows) else len(lines)
        n_rows = max(0, end - start - 2)
        frame = pd.read_csv(
            path,
            sep=sep,
            decimal=decimal,
            skiprows=start,
            nrows=n_rows or None,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
        frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
        frame["timestamp_dt"] = _timestamps(frame)
        frame = frame[frame["timestamp_dt"].notna()].reset_index(drop=True)
        blocks[kind] = pd.concat([blocks[kind], frame], ignore_index=True) if kind in blocks else frame

    empty = pd.DataFrame(columns=["timestamp_dt"])
    return CareLinkSections(
        pump=blocks.get(SECTION_PUMP, empty).copy(),
        aggregated=blocks.get(SECTION_AGGREGATED, empty).copy(),
        sensor=blocks.get(SECTION_SENSOR, empty).copy(),
        device_header=lines[0].strip() if lines else "",
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _deduplicated_boluses(pump: pd.DataFrame) -> pd.DataFrame:
    """One row per physically delivered bolus."""
    units = _numeric(pump, "Bolus Volume Delivered (U)")
    mask = units.notna() & (units > 0)
    if not mask.any():
        return pd.DataFrame(columns=["timestamp_dt", "bolus_u", "bolus_source"])
    frame = pump.loc[mask, ["timestamp_dt"]].copy()
    frame["bolus_u"] = units[mask]
    if "Bolus Source" in pump:
        frame["bolus_source"] = pump.loc[mask, "Bolus Source"].astype("string").fillna("")
    else:
        frame["bolus_source"] = ""
    if "Bolus Number" in pump:
        number = pump.loc[mask, "Bolus Number"].astype("string").fillna("")
    else:
        number = pd.Series("", index=frame.index, dtype="string")
    frame["key"] = number.where(number.str.len() > 0, frame["timestamp_dt"].astype(str))
    frame = frame.groupby(["timestamp_dt", "key"], as_index=False).agg(
        bolus_u=("bolus_u", "max"),
        bolus_source=("bolus_source", "first"),
    )
    return frame.drop(columns=[c for c in ("key",) if c in frame])


def _programmed_basal(pump: pd.DataFrame) -> pd.DataFrame:
    rate = _numeric(pump, "Basal Rate (U/h)")
    frame = pump.loc[rate.notna(), ["timestamp_dt"]].copy()
    frame["prog_basal_u_per_h"] = rate[rate.notna()]
    return frame.sort_values("timestamp_dt").drop_duplicates("timestamp_dt").reset_index(drop=True)


def _daily_auto_insulin(aggregated: pd.DataFrame) -> pd.Series:
    """Delivered SmartGuard auto-insulin per calendar day (U)."""
    if aggregated.empty:
        return pd.Series(dtype=float)
    units = _numeric(aggregated, "Bolus Volume Delivered (U)")
    frame = aggregated.loc[units.notna(), ["timestamp_dt"]].copy()
    frame["u"] = units[units.notna()]
    frame["day"] = frame["timestamp_dt"].dt.normalize()
    return frame.groupby("day")["u"].sum()


def _align_to_grid(
    events: pd.DataFrame,
    grid: pd.Series,
    value: str,
    tolerance_min: float = _ALIGN_TOLERANCE_MIN,
) -> np.ndarray:
    """Sum an event column onto the nearest grid sample within ``tolerance_min``."""
    out = np.zeros(len(grid), dtype=float)
    if events.empty or len(grid) == 0:
        return out
    grid_vals = grid.to_numpy()
    idx = np.searchsorted(grid_vals, events["timestamp_dt"].to_numpy())
    for position, raw_slot in enumerate(idx):
        candidates = [c for c in (raw_slot - 1, raw_slot) if 0 <= c < len(grid_vals)]
        if not candidates:
            continue
        event_time = np.datetime64(events["timestamp_dt"].iloc[position])
        best = min(candidates, key=lambda c: abs((grid_vals[c] - event_time) / np.timedelta64(1, "m")))
        if abs((grid_vals[best] - event_time) / np.timedelta64(1, "m")) <= tolerance_min:
            out[best] += float(events[value].iloc[position])
    return out


def _fill_cgm_gaps(grid: pd.DataFrame, max_step_min: float = 7.5) -> pd.DataFrame:
    """Insert 5-minute placeholder rows (glucose=NaN) across CGM dropouts."""
    times = grid["timestamp_dt"].to_numpy()
    extra: List[pd.Timestamp] = []
    steps = np.diff(times) / np.timedelta64(1, "m")
    for position, step in enumerate(steps):
        if step <= max_step_min:
            continue
        n_insert = int(np.floor(step / _GRID_MINUTES)) - 1
        start = pd.Timestamp(times[position])
        extra.extend(start + pd.to_timedelta(_GRID_MINUTES * (k + 1), unit="m") for k in range(n_insert))
    if not extra:
        grid["is_filled"] = False
        return grid
    filler = pd.DataFrame({"timestamp_dt": extra})
    filler["glucose"] = np.nan
    filler["isig"] = np.nan
    filler["sensor_exception"] = ""
    filler["is_filled"] = True
    grid["is_filled"] = False
    out = pd.concat([grid, filler], ignore_index=True).sort_values("timestamp_dt")
    return out.drop_duplicates("timestamp_dt").reset_index(drop=True)


def build_timeline(
    sections: CareLinkSections,
    *,
    auto_basal_mode: str = "profile",
    fill_gaps: bool = True,
) -> pd.DataFrame:
    """Reconcile CareLink sections into a continuous 5-minute timeline."""
    if auto_basal_mode not in {"uniform", "profile", "none"}:
        raise ValueError("auto_basal_mode must be 'uniform', 'profile' or 'none'")

    sensor = sections.sensor
    glucose = _numeric(sensor, "Sensor Glucose (mg/dL)")
    keep = glucose.notna()
    if "Sensor Exception" in sensor:
        exception = sensor["Sensor Exception"].astype("string").fillna("")
    else:
        exception = pd.Series("", index=sensor.index, dtype="string")

    grid = (
        sensor.loc[keep, ["timestamp_dt"]]
        .assign(
            glucose=glucose[keep],
            isig=_numeric(sensor, "ISIG Value")[keep],
            sensor_exception=exception[keep],
        )
        .sort_values("timestamp_dt")
        .drop_duplicates("timestamp_dt")
        .reset_index(drop=True)
    )

    if fill_gaps and len(grid) > 1:
        grid = _fill_cgm_gaps(grid)
    else:
        grid["is_filled"] = False

    times = grid["timestamp_dt"]
    grid["t_min"] = (times - times.iloc[0]).dt.total_seconds() / 60.0
    grid["day"] = times.dt.normalize()
    grid["minute_of_day"] = times.dt.hour * 60 + times.dt.minute

    pump = sections.pump
    boluses = _deduplicated_boluses(pump)
    grid["bolus_u"] = _align_to_grid(boluses, times, "bolus_u")

    carbs = _numeric(pump, "BWZ Carb Input (grams)")
    carb_events = pump.loc[carbs.notna() & (carbs > 0), ["timestamp_dt"]].copy()
    carb_events["carbs_g"] = carbs[carbs.notna() & (carbs > 0)]
    carb_events = carb_events.groupby("timestamp_dt", as_index=False).agg({"carbs_g": "max"})
    grid["carbs_g"] = _align_to_grid(carb_events, times, "carbs_g")

    programmed = _programmed_basal(pump)
    if programmed.empty:
        grid["prog_basal_u_per_h"] = 0.0
    else:
        merged = pd.merge_asof(
            grid[["timestamp_dt"]],
            programmed,
            on="timestamp_dt",
            direction="backward",
        )
        grid["prog_basal_u_per_h"] = merged["prog_basal_u_per_h"].ffill().fillna(0.0).to_numpy()

    daily_auto = _daily_auto_insulin(sections.aggregated)
    grid["auto_basal_u"] = 0.0
    if auto_basal_mode != "none" and not daily_auto.empty:
        for day, group in grid.groupby("day"):
            total = float(daily_auto.get(day, np.nan))
            if not np.isfinite(total) or total <= 0.0:
                continue
            if auto_basal_mode == "uniform":
                weights = np.ones(len(group), dtype=float)
            else:
                weights = group["prog_basal_u_per_h"].to_numpy(dtype=float)
                if weights.sum() <= 0.0:
                    weights = np.ones(len(group), dtype=float)
            grid.loc[group.index, "auto_basal_u"] = total * weights / weights.sum()

    grid["insulin_u"] = grid["bolus_u"] + grid["auto_basal_u"]
    grid["insulin_rate_u_per_h"] = grid["insulin_u"] * 60.0 / _GRID_MINUTES
    grid["step_min"] = grid["t_min"].diff()
    grid["daily_auto_total_u"] = grid["day"].map(daily_auto).astype(float)
    return grid.reset_index(drop=True)


def reconcile_report(sections: CareLinkSections, timeline: pd.DataFrame) -> pd.DataFrame:
    """Per-day insulin balance: what the pump reported versus what was booked."""
    boluses = _deduplicated_boluses(sections.pump)
    boluses["day"] = boluses["timestamp_dt"].dt.normalize()
    reported_bolus = boluses.groupby("day")["bolus_u"].sum()
    reported_auto = _daily_auto_insulin(sections.aggregated)
    booked = timeline.groupby("day")[["bolus_u", "auto_basal_u", "insulin_u"]].sum()
    cgm = timeline.groupby("day")["glucose"].count()

    report = pd.DataFrame(
        {
            "reported_bolus_u": reported_bolus,
            "reported_auto_u": reported_auto,
            "booked_bolus_u": booked["bolus_u"],
            "booked_auto_u": booked["auto_basal_u"],
            "booked_total_u": booked["insulin_u"],
            "cgm_samples": cgm,
        }
    )
    report["reported_total_u"] = report["reported_bolus_u"].fillna(0) + report["reported_auto_u"].fillna(0)
    report["balance_error_u"] = report["booked_total_u"] - report["reported_total_u"]
    report["cgm_coverage_pct"] = 100.0 * report["cgm_samples"] / 288.0
    return report
