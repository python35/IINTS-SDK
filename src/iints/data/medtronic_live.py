from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlparse
import json
import time
import urllib.request

import pandas as pd

from iints.data.importer import ImportResult, import_cgm_dataframe, scenario_from_dataframe
from iints.data.tidepool import MMOL_L_TO_MG_DL
from iints.utils.url_safety import validate_service_base_url


TIMESTAMP_KEYS = (
    "timestamp",
    "date",
    "datetime",
    "dateTime",
    "time",
    "eventTime",
    "eventDateTime",
    "sensorDate",
    "medicalDeviceTime",
    "clientTime",
    "serverTime",
    "displayTime",
)
GLUCOSE_KEYS = (
    "sg",
    "sgv",
    "glucose",
    "glucoseValue",
    "sensorGlucose",
    "sensorGlucoseValue",
    "sensorGlucoseMgDl",
    "sensorGlucoseMGDL",
    "currentSensorGlucose",
)
CARB_KEYS = (
    "carbs",
    "carbohydrates",
    "carbInput",
    "carbGrams",
    "mealCarbs",
    "foodCarbs",
    "bwzCarbInput",
    "bwzCarbInputGrams",
)
INSULIN_KEYS = (
    "insulin",
    "insulinDelivered",
    "deliveredInsulin",
    "bolus",
    "bolusAmount",
    "bolusUnits",
    "bolusVolumeDelivered",
    "basalUnits",
    "basalAmount",
)
UNIT_KEYS = ("unit", "units", "glucoseUnit", "glucoseUnits")
TYPE_KEYS = ("type", "kind", "eventType", "measurementType")
LIST_KEYS = (
    "data",
    "entries",
    "events",
    "items",
    "readings",
    "measurements",
    "sgs",
    "sgvs",
    "timeline",
)
SINGLE_RECORD_KEYS = ("current", "latest", "last", "lastSensorGlucose", "currentSensorGlucose")
GLUCOSE_TYPES = {"cgm", "sg", "sgv", "sensor_glucose", "sensorglucose", "glucose"}


@dataclass
class MedtronicLiveConfig:
    """
    Configuration for an authorized read-only Medtronic/CareLink live relay.

    This client intentionally does not embed private CareLink or mobile-app endpoints.
    Point it at an official/internal gateway, consented care-partner relay, or test
    fixture that returns JSON snapshots.
    """

    base_url: str
    token: Optional[str] = None
    endpoint_path: str = "/carelink/live"
    device_id: Optional[str] = None
    patient_id: Optional[str] = None
    since: Optional[str] = None
    limit: Optional[int] = None
    event_tolerance_minutes: float = 7.5
    source: str = "medtronic_carelink_live"

    def __post_init__(self) -> None:
        self.base_url = validate_service_base_url(self.base_url, label="Medtronic live base URL")
        self.endpoint_path = _normalize_endpoint_path(self.endpoint_path)


@dataclass
class MedtronicLiveClient:
    """Small authenticated JSON client for a read-only Medtronic live data relay."""

    base_url: str
    token: Optional[str] = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.base_url = validate_service_base_url(self.base_url, label="Medtronic live base URL")

    def get_json(self, path: str, *, query: Optional[Dict[str, Any]] = None) -> Any:
        endpoint_path = _normalize_endpoint_path(path)
        url = self.base_url.rstrip("/") + endpoint_path
        if query:
            clean_query = {
                key: value
                for key, value in query.items()
                if value is not None and str(value).strip() != ""
            }
            if clean_query:
                url = f"{url}?{urlencode(clean_query)}"

        headers = {
            "Accept": "application/json",
            "User-Agent": "iints-medtronic-live/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec - URL is validated.
            payload = response.read().decode("utf-8")
        return json.loads(payload)


def _normalize_endpoint_path(path: str) -> str:
    parsed = urlparse(path.strip())
    if parsed.scheme or parsed.netloc:
        raise ValueError("Medtronic live endpoint path must be a path, not a full URL.")
    if parsed.query or parsed.fragment:
        raise ValueError("Medtronic live endpoint path must not include query or fragment components.")
    clean_path = "/" + path.strip().lstrip("/")
    if clean_path == "/":
        raise ValueError("Medtronic live endpoint path must not be empty.")
    return clean_path


def _normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _lookup(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    wanted = {_normalize_key(key) for key in keys}
    for key, value in record.items():
        if _normalize_key(key) in wanted:
            return value
    for value in record.values():
        if isinstance(value, dict):
            found = _lookup(value, keys)
            if found is not None:
                return found
    return None


def _parse_number(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = _lookup(value, ("value", "amount", "reading"))
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace(",", ".")
    numeric = pd.to_numeric(pd.Series([cleaned]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _parse_timestamp(value: Any) -> Any:
    if isinstance(value, dict):
        value = _lookup(value, TIMESTAMP_KEYS)
    if value is None or value == "":
        return pd.NaT
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if numeric_value > 1_000_000_000_000:
            return pd.to_datetime(numeric_value, unit="ms", errors="coerce", utc=True)
        return pd.to_datetime(numeric_value, unit="s", errors="coerce", utc=True)
    return pd.to_datetime(str(value), errors="coerce", utc=True)


def _record_type(record: Dict[str, Any]) -> str:
    raw_type = _lookup(record, TYPE_KEYS)
    return _normalize_key(raw_type or "")


def _glucose_mgdl(record: Dict[str, Any]) -> Optional[float]:
    raw_glucose = _lookup(record, GLUCOSE_KEYS)
    units = _lookup(record, UNIT_KEYS)
    if raw_glucose is None and _record_type(record) in GLUCOSE_TYPES:
        raw_glucose = _lookup(record, ("value", "amount", "reading"))
    if isinstance(raw_glucose, dict):
        units = units or _lookup(raw_glucose, UNIT_KEYS)
    glucose = _parse_number(raw_glucose)
    if glucose is None:
        return None
    unit_text = str(units or "").strip().lower()
    if "mmol" in unit_text:
        return glucose * MMOL_L_TO_MG_DL
    return glucose


def _event_amount(record: Dict[str, Any], keys: Iterable[str]) -> float:
    amount = _parse_number(_lookup(record, keys))
    if amount is None or amount <= 0:
        return 0.0
    return amount


def _looks_like_record(record: Dict[str, Any]) -> bool:
    return (
        _lookup(record, TIMESTAMP_KEYS) is not None
        and (
            _lookup(record, GLUCOSE_KEYS) is not None
            or _lookup(record, CARB_KEYS) is not None
            or _lookup(record, INSULIN_KEYS) is not None
        )
    )


def _extract_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    records: List[Dict[str, Any]] = []
    if _looks_like_record(payload):
        records.append(dict(payload))

    for key in SINGLE_RECORD_KEYS:
        value = _lookup(payload, (key,))
        if isinstance(value, dict) and _looks_like_record(value):
            records.append(dict(value))

    lower_to_key = {_normalize_key(key): key for key in payload.keys()}
    for key in LIST_KEYS:
        actual_key = lower_to_key.get(_normalize_key(key))
        if not actual_key:
            continue
        value = payload.get(actual_key)
        if isinstance(value, list):
            records.extend(dict(item) for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            records.extend(_extract_records(value))

    return records


def _nearest_glucose_index(
    glucose_times: List[pd.Timestamp],
    event_time: pd.Timestamp,
    *,
    tolerance_minutes: float,
) -> Optional[int]:
    if not glucose_times:
        return None
    ordered = pd.Series(glucose_times)
    insert_at = int(ordered.searchsorted(event_time))
    candidates = [idx for idx in (insert_at - 1, insert_at) if 0 <= idx < len(glucose_times)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda idx: abs(glucose_times[idx] - event_time))
    if abs(glucose_times[nearest] - event_time) > pd.Timedelta(minutes=tolerance_minutes):
        return None
    return nearest


def normalize_medtronic_live_payload(
    payload: Any,
    *,
    event_tolerance_minutes: float = 7.5,
    source: str = "medtronic_carelink_live",
) -> pd.DataFrame:
    """
    Normalize a CareLink/mobile-relay JSON snapshot into a live glucose timeline.

    Expected output columns are `timestamp_dt`, `glucose`, `carbs`, `insulin`, and `source`.
    The parser accepts common field aliases so engineering relays can evolve without
    changing downstream IINTS ingestion code.
    """

    glucose_rows: List[Dict[str, Any]] = []
    carb_events: List[Tuple[pd.Timestamp, float]] = []
    insulin_events: List[Tuple[pd.Timestamp, float]] = []

    for record in _extract_records(payload):
        timestamp = _parse_timestamp(_lookup(record, TIMESTAMP_KEYS))
        if pd.isna(timestamp):
            continue

        glucose = _glucose_mgdl(record)
        if glucose is not None:
            glucose_rows.append(
                {
                    "timestamp_dt": timestamp,
                    "glucose": glucose,
                    "carbs": 0.0,
                    "insulin": 0.0,
                    "source": source,
                }
            )

        carbs = _event_amount(record, CARB_KEYS)
        if carbs > 0:
            carb_events.append((timestamp, carbs))

        insulin = _event_amount(record, INSULIN_KEYS)
        if insulin > 0:
            insulin_events.append((timestamp, insulin))

    if not glucose_rows:
        return pd.DataFrame(columns=["timestamp_dt", "glucose", "carbs", "insulin", "source"])

    dataframe = (
        pd.DataFrame(glucose_rows)
        .drop_duplicates(subset=["timestamp_dt"], keep="last")
        .sort_values("timestamp_dt")
        .reset_index(drop=True)
    )
    glucose_times = list(dataframe["timestamp_dt"])
    for timestamp, value in carb_events:
        index = _nearest_glucose_index(
            glucose_times,
            timestamp,
            tolerance_minutes=event_tolerance_minutes,
        )
        if index is not None:
            existing_carbs = _parse_number(dataframe.at[index, "carbs"]) or 0.0
            dataframe.at[index, "carbs"] = existing_carbs + float(value)
    for timestamp, value in insulin_events:
        index = _nearest_glucose_index(
            glucose_times,
            timestamp,
            tolerance_minutes=event_tolerance_minutes,
        )
        if index is not None:
            existing_insulin = _parse_number(dataframe.at[index, "insulin"]) or 0.0
            dataframe.at[index, "insulin"] = existing_insulin + float(value)

    return dataframe[["timestamp_dt", "glucose", "carbs", "insulin", "source"]]


def fetch_medtronic_live_payload(config: MedtronicLiveConfig) -> Any:
    client = MedtronicLiveClient(base_url=config.base_url, token=config.token)
    return client.get_json(
        config.endpoint_path,
        query={
            "deviceId": config.device_id,
            "patientId": config.patient_id,
            "since": config.since,
            "limit": config.limit,
        },
    )


def fetch_medtronic_live_timeline(config: MedtronicLiveConfig) -> pd.DataFrame:
    payload = fetch_medtronic_live_payload(config)
    return normalize_medtronic_live_payload(
        payload,
        event_tolerance_minutes=config.event_tolerance_minutes,
        source=config.source,
    )


def medtronic_live_timeline_to_standard(
    timeline: pd.DataFrame,
    *,
    source: str = "medtronic_carelink_live",
) -> pd.DataFrame:
    if timeline.empty:
        return pd.DataFrame(columns=["timestamp", "glucose", "carbs", "insulin", "source"])
    standard_input = timeline.rename(columns={"timestamp_dt": "timestamp"})[
        ["timestamp", "glucose", "carbs", "insulin"]
    ].copy()
    return import_cgm_dataframe(standard_input, data_format="generic", source=source)


def fetch_medtronic_live_dataframe(config: MedtronicLiveConfig) -> pd.DataFrame:
    timeline = fetch_medtronic_live_timeline(config)
    return medtronic_live_timeline_to_standard(timeline, source=config.source)


def import_medtronic_live(
    config: MedtronicLiveConfig,
    scenario_name: str = "Medtronic CareLink Live Import",
    scenario_version: str = "1.0",
    carb_threshold: float = 0.1,
) -> ImportResult:
    dataframe = fetch_medtronic_live_dataframe(config)
    if dataframe.empty:
        raise ValueError("No Medtronic live CGM readings found for the given parameters.")
    scenario = scenario_from_dataframe(
        dataframe,
        scenario_name=scenario_name,
        scenario_version=scenario_version,
        carb_threshold=carb_threshold,
    )
    return ImportResult(dataframe=dataframe, scenario=scenario)


def poll_medtronic_live_timeline(
    config: MedtronicLiveConfig,
    *,
    samples: int = 1,
    poll_seconds: float = 30.0,
) -> Iterable[pd.DataFrame]:
    """
    Poll a Medtronic live relay.

    `samples=0` runs until interrupted by the caller. Each yielded dataframe is a
    point-in-time normalized timeline; callers can merge/dedupe as needed.
    """

    count = 0
    while samples == 0 or count < samples:
        yield fetch_medtronic_live_timeline(config)
        count += 1
        if samples == 0 or count < samples:
            time.sleep(max(0.0, float(poll_seconds)))


def write_latest_medtronic_live_snapshot(timeline: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / "live_timeline.csv"
    standard_path = output_dir / "cgm_standard.csv"
    latest_path = output_dir / "latest.json"

    if timeline.empty:
        timeline.to_csv(timeline_path, index=False)
        pd.DataFrame(columns=["timestamp", "glucose", "carbs", "insulin", "source"]).to_csv(
            standard_path,
            index=False,
        )
        latest_path.write_text(json.dumps({"rows": 0, "latest": None}, indent=2), encoding="utf-8")
        return {
            "timeline_csv": str(timeline_path),
            "standard_csv": str(standard_path),
            "latest_json": str(latest_path),
        }

    timeline = timeline.sort_values("timestamp_dt").drop_duplicates("timestamp_dt", keep="last")
    timeline.to_csv(timeline_path, index=False)

    source = str(timeline["source"].iloc[-1]) if "source" in timeline.columns else "medtronic_carelink_live"
    standard = medtronic_live_timeline_to_standard(timeline, source=source)
    standard.to_csv(standard_path, index=False)

    latest_row = timeline.iloc[-1]
    latest = {
        "rows": int(len(timeline)),
        "latest": {
            "timestamp": str(latest_row["timestamp_dt"]),
            "glucose": float(latest_row["glucose"]),
            "carbs": float(latest_row["carbs"]),
            "insulin": float(latest_row["insulin"]),
            "source": source,
        },
    }
    latest_path.write_text(json.dumps(latest, indent=2), encoding="utf-8")
    return {
        "timeline_csv": str(timeline_path),
        "standard_csv": str(standard_path),
        "latest_json": str(latest_path),
    }
