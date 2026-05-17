from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, cast
from urllib.parse import urlencode
import json
import urllib.request

import pandas as pd

from iints.data.importer import ImportResult, import_cgm_dataframe, scenario_from_dataframe
from iints.utils.url_safety import validate_service_base_url


MMOL_L_TO_MG_DL = 18.01559


@dataclass
class TidepoolConfig:
    base_url: str = "https://api.tidepool.org"
    token: Optional[str] = None
    user_id: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    types: tuple[str, ...] = ("cbg", "bolus", "wizard", "food")
    event_tolerance_minutes: float = 7.5

    def __post_init__(self) -> None:
        self.base_url = validate_service_base_url(self.base_url, label="Tidepool API base URL")


@dataclass
class TidepoolClient:
    """Small authenticated Tidepool Data Platform client for read-only imports."""

    base_url: str = "https://api.tidepool.org"
    token: Optional[str] = None

    def __post_init__(self) -> None:
        self.base_url = validate_service_base_url(self.base_url, label="Tidepool API base URL")

    def _headers(self) -> Dict[str, str]:
        if not self.token:
            raise RuntimeError("TidepoolClient requires a session token before making requests.")
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Tidepool-Session-Token": self.token,
        }

    def get_json(self, path: str, *, query: Optional[dict[str, Any]] = None) -> Any:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        if query:
            clean_query = {key: value for key, value in query.items() if value is not None}
            if clean_query:
                url = f"{url}?{urlencode(clean_query)}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req) as response:  # nosec - URL is validated and caller-controlled.
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def current_user(self) -> dict[str, Any]:
        payload = self.get_json("/auth/user")
        if not isinstance(payload, dict):
            raise RuntimeError("Tidepool /auth/user returned a non-object payload.")
        return payload

    def fetch_device_data(
        self,
        user_id: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        types: Iterable[str] = ("cbg", "bolus", "wizard", "food"),
    ) -> list[dict[str, Any]]:
        payload = self.get_json(
            f"/data/{user_id}",
            query={
                "startDate": start,
                "endDate": end,
                "type": ",".join(str(item) for item in types),
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError("Tidepool /data endpoint returned a non-list payload.")
        return [dict(item) for item in payload if isinstance(item, dict)]


def _timestamp_from_event(event: dict[str, Any]) -> Any:
    raw_timestamp = event.get("time") or event.get("deviceTime")
    if raw_timestamp is None:
        return pd.NaT
    return pd.to_datetime(str(raw_timestamp), errors="coerce", utc=True)


def _glucose_to_mgdl(value: Any, units: Any) -> Optional[float]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    unit_text = str(units or "").strip().lower()
    if unit_text in {"mmol/l", "mmol"}:
        return float(numeric) * MMOL_L_TO_MG_DL
    return float(numeric)


def _bolus_units(event: dict[str, Any]) -> float:
    for key in ("normal", "extended", "value", "expectedNormal"):
        value = pd.to_numeric(pd.Series([event.get(key)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return 0.0


def _carb_grams(event: dict[str, Any]) -> float:
    direct = pd.to_numeric(
        pd.Series([event.get("carbInput") or event.get("carbs")]),
        errors="coerce",
    ).iloc[0]
    if pd.notna(direct) and float(direct) > 0:
        return float(direct)

    nutrition = event.get("nutrition")
    if isinstance(nutrition, dict):
        carbohydrate = nutrition.get("carbohydrate")
        if isinstance(carbohydrate, dict):
            for key in ("net", "total"):
                value = pd.to_numeric(pd.Series([carbohydrate.get(key)]), errors="coerce").iloc[0]
                if pd.notna(value) and float(value) > 0:
                    return float(value)
    return 0.0


def _nearest_glucose_index(
    glucose_times: list[pd.Timestamp],
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
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    if abs(glucose_times[nearest] - event_time) > tolerance:
        return None
    return nearest


def _events_to_dataframe(
    events: Iterable[dict[str, Any]],
    *,
    event_tolerance_minutes: float,
) -> pd.DataFrame:
    glucose_rows: list[dict[str, Any]] = []
    carb_events: list[tuple[pd.Timestamp, float]] = []
    insulin_events: list[tuple[pd.Timestamp, float]] = []
    for event in events:
        event_type = str(event.get("type") or "").strip()
        timestamp = _timestamp_from_event(event)
        if pd.isna(timestamp):
            continue
        if event_type == "cbg":
            glucose = _glucose_to_mgdl(event.get("value"), event.get("units"))
            if glucose is None:
                continue
            glucose_rows.append(
                {
                    "timestamp_dt": timestamp,
                    "glucose": glucose,
                    "carbs": 0.0,
                    "insulin": 0.0,
                }
            )
        elif event_type == "bolus":
            insulin = _bolus_units(event)
            if insulin > 0:
                insulin_events.append((timestamp, insulin))
        elif event_type in {"wizard", "food"}:
            carbs = _carb_grams(event)
            if carbs > 0:
                carb_events.append((timestamp, carbs))

    if not glucose_rows:
        return pd.DataFrame(columns=["timestamp", "glucose", "carbs", "insulin"])

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
            existing_carbs = cast(Any, dataframe.at[index, "carbs"])
            dataframe.at[index, "carbs"] = float(existing_carbs) + float(value)
    for timestamp, value in insulin_events:
        index = _nearest_glucose_index(
            glucose_times,
            timestamp,
            tolerance_minutes=event_tolerance_minutes,
        )
        if index is not None:
            existing_insulin = cast(Any, dataframe.at[index, "insulin"])
            dataframe.at[index, "insulin"] = float(existing_insulin) + float(value)

    return dataframe.rename(columns={"timestamp_dt": "timestamp"})[
        ["timestamp", "glucose", "carbs", "insulin"]
    ]


def fetch_tidepool_dataframe(config: TidepoolConfig) -> pd.DataFrame:
    client = TidepoolClient(base_url=config.base_url, token=config.token)
    user_id = config.user_id
    if not user_id:
        user_payload = client.current_user()
        user_id = str(user_payload.get("userid") or user_payload.get("userId") or "").strip()
    if not user_id:
        raise ValueError("Could not resolve a Tidepool user id. Pass --user-id explicitly.")
    events = client.fetch_device_data(
        user_id,
        start=config.start,
        end=config.end,
        types=config.types,
    )
    return _events_to_dataframe(
        events,
        event_tolerance_minutes=config.event_tolerance_minutes,
    )


def import_tidepool(
    config: TidepoolConfig,
    scenario_name: str = "Tidepool Import",
    scenario_version: str = "1.0",
    carb_threshold: float = 0.1,
) -> ImportResult:
    dataframe = fetch_tidepool_dataframe(config)
    if dataframe.empty:
        raise ValueError("No Tidepool CGM entries found for the given parameters.")
    standard_df = import_cgm_dataframe(
        dataframe,
        data_format="generic",
        time_unit="minutes",
        source="tidepool",
    )
    scenario = scenario_from_dataframe(
        standard_df,
        scenario_name=scenario_name,
        scenario_version=scenario_version,
        carb_threshold=carb_threshold,
    )
    return ImportResult(dataframe=standard_df, scenario=scenario)


def load_openapi_spec(path: str) -> Dict[str, Any]:
    """Load a local Tidepool OpenAPI JSON spec for reference tooling."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
