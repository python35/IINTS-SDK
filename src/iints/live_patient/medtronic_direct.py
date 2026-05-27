from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol
import json
import math
import time

import pandas as pd

from iints.core.safety.config import SENSOR_GLUCOSE_MAX_MGDL, SENSOR_GLUCOSE_MIN_MGDL
from iints.data.importer import import_cgm_dataframe


DIRECT_PUMP_READ_ONLY_CONFIRMATION = "I confirm this is an authorized read-only Medtronic transport"
DIRECT_PUMP_SOURCE = "medtronic_direct_pump"
AUTHORIZED_IDENTITY_MODE = "authorized_read_only"
DISALLOWED_IDENTITY_MODES = {
    "spoof_mobile_app",
    "spoof_minimed_mobile",
    "spoof_carelink_connect",
    "spoof_sensor",
    "impersonate_mobile_app",
    "impersonate_sensor",
    "emulate_sensor",
    "emulate_mobile_app",
}
COMMAND_LIKE_KEYS = {
    "actuate",
    "authchallenge",
    "authresponse",
    "basalcommand",
    "blepairingkey",
    "boluscommand",
    "bolusrequest",
    "command",
    "dosecommand",
    "pairingkey",
    "primecommand",
    "sessionkey",
    "writecharacteristic",
}
TIMESTAMP_KEYS = ("timestamp", "dateTime", "eventTime", "medicalDeviceTime", "pumpTime")
GLUCOSE_KEYS = ("glucose", "glucoseMgDl", "sensorGlucose", "sensorGlucoseMgDl", "sg", "sgv")


class PumpTransport(Protocol):
    """Read-only direct pump transport protocol.

    Official Medtronic protocol code can implement this protocol and be loaded
    into the SDK without teaching IINTS how to spoof app or sensor identity.
    """

    def connect(self) -> None:
        ...

    def read_snapshot(self) -> "PumpSnapshot | Mapping[str, Any]":
        ...

    def disconnect(self) -> None:
        ...


@dataclass(frozen=True)
class PumpSnapshot:
    timestamp: pd.Timestamp
    glucose_mgdl: float
    carbs_g: float = 0.0
    insulin_u: float = 0.0
    iob_u: Optional[float] = None
    basal_rate_u_per_hr: Optional[float] = None
    reservoir_u: Optional[float] = None
    battery_percent: Optional[float] = None
    alert: str = ""
    pump_state: str = "unknown"
    source: str = DIRECT_PUMP_SOURCE

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PumpSnapshot":
        _reject_command_like_payload(payload)
        timestamp = _parse_timestamp(_lookup(payload, TIMESTAMP_KEYS))
        if pd.isna(timestamp):
            raise ValueError("Pump snapshot is missing a valid timestamp.")
        glucose = _parse_float(_lookup(payload, GLUCOSE_KEYS))
        if glucose is None:
            raise ValueError("Pump snapshot is missing a valid glucose value.")
        snapshot = cls(
            timestamp=timestamp,
            glucose_mgdl=glucose,
            carbs_g=_parse_float(_lookup(payload, ("carbs", "carbsG", "carbInput"))) or 0.0,
            insulin_u=_parse_float(_lookup(payload, ("insulin", "insulinU", "insulinDelivered", "bolusUnits"))) or 0.0,
            iob_u=_parse_float(_lookup(payload, ("iob", "iobU", "activeInsulin"))),
            basal_rate_u_per_hr=_parse_float(
                _lookup(payload, ("basalRate", "basalRateUPerHr", "basalRateUnitsPerHour"))
            ),
            reservoir_u=_parse_float(_lookup(payload, ("reservoir", "reservoirU", "reservoirUnits"))),
            battery_percent=_parse_float(_lookup(payload, ("battery", "batteryPercent"))),
            alert=str(_lookup(payload, ("alert", "alarm", "notification")) or ""),
            pump_state=str(_lookup(payload, ("pumpState", "state", "status")) or "unknown"),
            source=str(_lookup(payload, ("source",)) or DIRECT_PUMP_SOURCE),
        )
        snapshot.validate()
        return snapshot

    def validate(self) -> None:
        if not SENSOR_GLUCOSE_MIN_MGDL <= float(self.glucose_mgdl) <= SENSOR_GLUCOSE_MAX_MGDL:
            raise ValueError(
                "Glucose value outside broad CGM/sensor-valid range "
                f"({int(SENSOR_GLUCOSE_MIN_MGDL)}-{int(SENSOR_GLUCOSE_MAX_MGDL)} mg/dL)."
            )
        if self.carbs_g < 0:
            raise ValueError("Carbohydrate grams must be non-negative.")
        if self.insulin_u < 0:
            raise ValueError("Insulin units must be non-negative.")
        if self.battery_percent is not None and not 0 <= self.battery_percent <= 100:
            raise ValueError("Battery percent must be between 0 and 100.")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(frozen=True)
class DirectPumpConfig:
    transport: str = "simulated"
    official_factory: Optional[str] = None
    identity_mode: str = AUTHORIZED_IDENTITY_MODE
    source: str = DIRECT_PUMP_SOURCE
    simulated_seed: int = 42
    simulated_start_glucose_mgdl: float = 118.0
    simulated_step_minutes: float = 5.0

    def __post_init__(self) -> None:
        normalized_identity = _normalize_key(self.identity_mode)
        if normalized_identity in DISALLOWED_IDENTITY_MODES or self.identity_mode != AUTHORIZED_IDENTITY_MODE:
            raise ValueError(
                "Direct pump transport supports authorized read-only identity only. "
                "App/sensor impersonation modes are intentionally not implemented."
            )
        if self.transport not in {"simulated", "official-module"}:
            raise ValueError("Direct pump transport must be 'simulated' or 'official-module'.")
        if self.transport == "official-module" and not self.official_factory:
            raise ValueError("official-module transport requires an official factory reference.")


class SimulatedMedtronicPumpTransport:
    """Bench-only direct pump transport for SDK integration testing."""

    def __init__(
        self,
        *,
        source: str = DIRECT_PUMP_SOURCE,
        seed: int = 42,
        start_glucose_mgdl: float = 118.0,
        step_minutes: float = 5.0,
    ) -> None:
        self.source = source
        self.seed = int(seed)
        self.start_glucose_mgdl = float(start_glucose_mgdl)
        self.step_minutes = float(step_minutes)
        self.connected = False
        self.index = 0
        self.started_at = pd.Timestamp(datetime(2026, 5, 25, 8, 0, tzinfo=timezone.utc))

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_snapshot(self) -> PumpSnapshot:
        if not self.connected:
            raise RuntimeError("Simulated Medtronic pump transport is not connected.")

        phase = (self.index + self.seed % 17) / 6.0
        meal_bump = 26.0 * math.exp(-((self.index - 7.0) ** 2) / 18.0)
        glucose = self.start_glucose_mgdl + 12.0 * math.sin(phase) + meal_bump
        carbs = 38.0 if self.index == 5 else 0.0
        insulin = 3.1 if self.index == 6 else 0.0
        timestamp = self.started_at + pd.Timedelta(minutes=self.index * self.step_minutes)
        self.index += 1
        return PumpSnapshot(
            timestamp=timestamp,
            glucose_mgdl=round(float(glucose), 1),
            carbs_g=carbs,
            insulin_u=insulin,
            iob_u=max(0.0, 3.1 - max(0, self.index - 6) * 0.15),
            basal_rate_u_per_hr=0.85,
            reservoir_u=142.0 - self.index * 0.03,
            battery_percent=max(20.0, 96.0 - self.index * 0.02),
            pump_state="simulated_read_only",
            source=self.source,
        )


class OfficialModulePumpTransport:
    """Wrapper for an approved internal Medtronic read-only transport factory."""

    def __init__(self, factory_ref: str) -> None:
        self.factory_ref = factory_ref
        self._transport = _load_transport_factory(factory_ref)()

    def connect(self) -> None:
        if hasattr(self._transport, "connect"):
            self._transport.connect()

    def read_snapshot(self) -> PumpSnapshot | Mapping[str, Any]:
        if not hasattr(self._transport, "read_snapshot"):
            raise RuntimeError("Official transport object must expose read_snapshot().")
        return self._transport.read_snapshot()

    def disconnect(self) -> None:
        if hasattr(self._transport, "disconnect"):
            self._transport.disconnect()


def create_direct_pump_transport(config: DirectPumpConfig) -> PumpTransport:
    if config.transport == "simulated":
        return SimulatedMedtronicPumpTransport(
            source=config.source,
            seed=config.simulated_seed,
            start_glucose_mgdl=config.simulated_start_glucose_mgdl,
            step_minutes=config.simulated_step_minutes,
        )
    return OfficialModulePumpTransport(str(config.official_factory))


def stream_direct_pump_snapshots(
    config: DirectPumpConfig,
    *,
    samples: int = 1,
    poll_seconds: float = 30.0,
) -> Iterable[PumpSnapshot]:
    transport = create_direct_pump_transport(config)
    transport.connect()
    try:
        count = 0
        while samples == 0 or count < samples:
            raw_snapshot = transport.read_snapshot()
            snapshot = (
                raw_snapshot
                if isinstance(raw_snapshot, PumpSnapshot)
                else PumpSnapshot.from_mapping(raw_snapshot)
            )
            snapshot.validate()
            yield snapshot
            count += 1
            if samples == 0 or count < samples:
                time.sleep(max(0.0, float(poll_seconds)))
    finally:
        transport.disconnect()


def snapshots_to_dataframes(snapshots: Iterable[PumpSnapshot]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [snapshot.to_json_dict() for snapshot in snapshots]
    timeline = pd.DataFrame(rows)
    if timeline.empty:
        standard = pd.DataFrame(columns=["timestamp", "glucose", "carbs", "insulin", "source"])
        return timeline, standard

    timeline["timestamp_dt"] = pd.to_datetime(timeline["timestamp"], errors="coerce", utc=True)
    timeline = timeline.sort_values("timestamp_dt").drop_duplicates("timestamp_dt", keep="last")
    standard_input = pd.DataFrame(
        {
            "timestamp": timeline["timestamp_dt"],
            "glucose": timeline["glucose_mgdl"],
            "carbs": timeline["carbs_g"],
            "insulin": timeline["insulin_u"],
        }
    )
    source = str(timeline["source"].iloc[-1])
    standard = import_cgm_dataframe(standard_input, data_format="generic", source=source)
    return timeline, standard


def write_direct_pump_snapshot(snapshots: Iterable[PumpSnapshot], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_list = list(snapshots)
    timeline, standard = snapshots_to_dataframes(snapshot_list)

    timeline_path = output / "pump_timeline.csv"
    standard_path = output / "cgm_standard.csv"
    latest_path = output / "pump_latest.json"
    timeline.to_csv(timeline_path, index=False)
    standard.to_csv(standard_path, index=False)

    latest = snapshot_list[-1].to_json_dict() if snapshot_list else None
    latest_path.write_text(
        json.dumps({"rows": len(snapshot_list), "latest": latest}, indent=2),
        encoding="utf-8",
    )
    return {
        "timeline_csv": str(timeline_path),
        "standard_csv": str(standard_path),
        "latest_json": str(latest_path),
    }


def _load_transport_factory(factory_ref: str) -> Any:
    if ":" not in factory_ref:
        raise ValueError("Official factory reference must use 'module.path:factory_name'.")
    module_name, factory_name = factory_ref.split(":", 1)
    module = import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise ValueError(f"Official factory '{factory_name}' not found in module '{module_name}'.")
    if not callable(factory):
        raise ValueError("Official factory reference must point to a callable.")
    return factory


def _normalize_key(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum() or ch == "_")


def _lookup(payload: Mapping[str, Any], keys: Iterable[str]) -> Any:
    wanted = {_normalize_key(key) for key in keys}
    for key, value in payload.items():
        if _normalize_key(key) in wanted:
            return value
    for value in payload.values():
        if isinstance(value, Mapping):
            found = _lookup(value, keys)
            if found is not None:
                return found
    return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Mapping):
        value = _lookup(value, ("value", "amount", "reading"))
    numeric = pd.to_numeric(pd.Series([str(value).strip().replace(",", ".")]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _parse_timestamp(value: Any) -> Any:
    if value is None or value == "":
        return pd.NaT
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if numeric_value > 1_000_000_000_000:
            return pd.to_datetime(numeric_value, unit="ms", errors="coerce", utc=True)
        return pd.to_datetime(numeric_value, unit="s", errors="coerce", utc=True)
    return pd.to_datetime(str(value), errors="coerce", utc=True)


def _reject_command_like_payload(payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        if _normalize_key(key) in COMMAND_LIKE_KEYS:
            raise ValueError("Direct pump snapshots must be read-only and must not contain command/auth fields.")
        if isinstance(value, Mapping):
            _reject_command_like_payload(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    _reject_command_like_payload(item)
