from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from iints.api.base_algorithm import InsulinAlgorithm
from iints.core.devices.models import SensorModel
from iints.core.patient.patient_factory import PatientFactory
from iints.core.patient.profile import PatientProfile
from iints.core.safety import SafetyConfig
from iints.core.simulator import SimulationLimitError, Simulator, StressEvent
from iints.utils.run_io import build_run_manifest, build_run_metadata, generate_run_id, resolve_seed, write_json
from iints.validation import load_patient_config, load_patient_config_by_name, validate_patient_config_dict


logger = logging.getLogger("iints.live_patient")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_algorithm_instance_silent(algo: Path) -> InsulinAlgorithm:
    import importlib.util
    import iints as iints_sdk

    if not algo.is_file():
        raise FileNotFoundError(f"Algorithm file '{algo}' not found.")
    resolved = algo.expanduser().resolve()
    module_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    module_name = f"_iints_live_patient_{resolved.stem}_{module_hash}_{time.monotonic_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.setdefault("iints", iints_sdk)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    for _, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, InsulinAlgorithm) and obj is not InsulinAlgorithm:
            return obj()
    raise ImportError(f"No subclass of InsulinAlgorithm found in {resolved}")


@dataclass(frozen=True)
class DailyEventTemplate:
    minute_of_day: int
    event_type: str
    value: float
    reported_value: float | None = None
    duration: int = 0
    absorption_delay_minutes: int = 0
    label: str = ""
    isf: float | None = None
    icr: float | None = None
    basal_rate: float | None = None
    dia_minutes: float | None = None


@dataclass(frozen=True)
class RuntimeScenarioProfile:
    name: str
    description: str
    templates: tuple[DailyEventTemplate, ...]
    default_seed: int
    warm_start_minutes: int = 0
    warmup_history_limit: int = 96
    require_recent_override: bool = False
    reset_note: str = ""


def _scenario_profiles() -> dict[str, RuntimeScenarioProfile]:
    return {
        "school_day": RuntimeScenarioProfile(
            name="school_day",
            description="Structured weekday rhythm with breakfast, lunch, and dinner around a typical school schedule.",
            default_seed=1001,
            templates=(
                DailyEventTemplate(minute_of_day=435, event_type="meal", value=46.0, label="Early breakfast"),
                DailyEventTemplate(minute_of_day=720, event_type="meal", value=68.0, label="School lunch"),
                DailyEventTemplate(minute_of_day=1095, event_type="meal", value=78.0, label="Family dinner"),
                DailyEventTemplate(minute_of_day=1285, event_type="meal", value=14.0, label="Homework snack"),
            ),
            reset_note="School-day reset complete.",
        ),
        "normal_day": RuntimeScenarioProfile(
            name="normal_day",
            description="Balanced school-day profile with breakfast, lunch walk, dinner, and a stable daytime rhythm.",
            default_seed=1101,
            templates=(
                DailyEventTemplate(minute_of_day=450, event_type="meal", value=48.0, label="Breakfast"),
                DailyEventTemplate(minute_of_day=720, event_type="meal", value=62.0, label="Lunch"),
                DailyEventTemplate(minute_of_day=750, event_type="exercise", value=0.35, duration=30, label="Lunch walk"),
                DailyEventTemplate(minute_of_day=1080, event_type="meal", value=74.0, label="Dinner"),
                DailyEventTemplate(minute_of_day=1290, event_type="meal", value=18.0, label="Evening snack"),
            ),
            reset_note="Normal day reset complete.",
        ),
        "sport_day": RuntimeScenarioProfile(
            name="sport_day",
            description="A more active day with afternoon exercise and a faster-moving glucose curve.",
            default_seed=2202,
            templates=(
                DailyEventTemplate(minute_of_day=435, event_type="meal", value=42.0, label="Light breakfast"),
                DailyEventTemplate(minute_of_day=735, event_type="meal", value=68.0, label="Lunch"),
                DailyEventTemplate(minute_of_day=1010, event_type="exercise", value=0.55, duration=55, label="Training session"),
                DailyEventTemplate(minute_of_day=1110, event_type="meal", value=26.0, label="Post-workout snack"),
                DailyEventTemplate(minute_of_day=1140, event_type="meal", value=78.0, label="Dinner"),
            ),
            reset_note="Sport-day reset complete.",
        ),
        "bad_carb_count": RuntimeScenarioProfile(
            name="bad_carb_count",
            description="Meals are under-counted on purpose, creating a realistic challenge for the controller.",
            default_seed=3303,
            templates=(
                DailyEventTemplate(minute_of_day=450, event_type="meal", value=52.0, reported_value=40.0, label="Breakfast undercounted"),
                DailyEventTemplate(minute_of_day=720, event_type="meal", value=96.0, reported_value=48.0, label="Lunch undercounted"),
                DailyEventTemplate(minute_of_day=1080, event_type="meal", value=108.0, reported_value=62.0, label="Dinner undercounted"),
                DailyEventTemplate(minute_of_day=1290, event_type="meal", value=12.0, reported_value=12.0, label="Late correction snack"),
            ),
            reset_note="Bad-carb-estimate reset complete.",
        ),
        "night_hypo_risk": RuntimeScenarioProfile(
            name="night_hypo_risk",
            description="A late-evening exertion and more aggressive overnight ratios create overnight low-risk pressure.",
            default_seed=4404,
            templates=(
                DailyEventTemplate(minute_of_day=455, event_type="meal", value=46.0, label="Breakfast"),
                DailyEventTemplate(minute_of_day=720, event_type="meal", value=58.0, label="Lunch"),
                DailyEventTemplate(minute_of_day=1090, event_type="meal", value=54.0, reported_value=44.0, label="Small dinner undercounted"),
                DailyEventTemplate(minute_of_day=1230, event_type="exercise", value=0.45, duration=45, label="Late walk"),
                DailyEventTemplate(
                    minute_of_day=1320,
                    event_type="ratio_change",
                    value=0.0,
                    duration=240,
                    label="Overnight aggressive settings",
                    isf=26.0,
                    icr=7.5,
                    basal_rate=1.2,
                    dia_minutes=300.0,
                ),
            ),
            reset_note="Night-risk reset complete.",
        ),
        "expo_hot_start": RuntimeScenarioProfile(
            name="expo_hot_start",
            description="Starts mid-challenge after an under-counted lunch so the curve is already moving when visitors arrive.",
            default_seed=5505,
            templates=(
                DailyEventTemplate(minute_of_day=450, event_type="meal", value=50.0, label="Breakfast"),
                DailyEventTemplate(minute_of_day=720, event_type="meal", value=104.0, reported_value=52.0, label="Expo lunch undercounted"),
                DailyEventTemplate(minute_of_day=750, event_type="exercise", value=0.35, duration=35, label="Busy expo walk"),
                DailyEventTemplate(minute_of_day=1080, event_type="meal", value=86.0, reported_value=64.0, label="Dinner undercounted"),
            ),
            warm_start_minutes=785,
            warmup_history_limit=96,
            require_recent_override=True,
            reset_note="Expo reset complete. The patient was warm-started mid-challenge.",
        ),
        "relaxed_day": RuntimeScenarioProfile(
            name="relaxed_day",
            description="Softer weekend schedule with a later breakfast, lighter daytime pace, and a calmer evening profile.",
            default_seed=6606,
            templates=(
                DailyEventTemplate(minute_of_day=540, event_type="meal", value=44.0, label="Late breakfast"),
                DailyEventTemplate(minute_of_day=825, event_type="meal", value=58.0, label="Weekend lunch"),
                DailyEventTemplate(minute_of_day=1110, event_type="meal", value=70.0, label="Relaxed dinner"),
                DailyEventTemplate(minute_of_day=1230, event_type="exercise", value=0.20, duration=25, label="Easy evening walk"),
            ),
            reset_note="Relaxed-day reset complete.",
        ),
    }


def get_runtime_scenario_profile(name: str) -> RuntimeScenarioProfile:
    profiles = _scenario_profiles()
    if name not in profiles:
        raise ValueError(f"Unknown digital patient scenario profile: {name}")
    return profiles[name]


def list_runtime_scenario_profiles() -> list[RuntimeScenarioProfile]:
    return list(_scenario_profiles().values())


@dataclass(frozen=True)
class PatientRuntimeConfig:
    workspace: str
    algo_path: str
    patient_config: str = "default_patient"
    patient_model_type: str = "auto"
    mode: str = "demo-time"
    speed: float = 60.0
    time_step_minutes: int = 5
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    allow_remote_api: bool = False
    api_token_env: str | None = None
    api_token_file: str | None = None
    scenario_profile: str = "normal_day"
    seed: int | None = None

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    @property
    def algo_file(self) -> Path:
        return Path(self.algo_path).expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.workspace_path / "patient_state.db"

    @property
    def snapshot_path(self) -> Path:
        return self.workspace_path / "simulator_snapshot.json"

    @property
    def pid_path(self) -> Path:
        return self.workspace_path / "patient.pid"

    @property
    def log_path(self) -> Path:
        return self.workspace_path / "patient.log"

    @property
    def config_path(self) -> Path:
        return self.workspace_path / "patient_runtime_config.json"

    @property
    def bundle_dir(self) -> Path:
        return self.workspace_path / "live_bundle"

    @property
    def dashboard_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}/dashboard"

    @property
    def api_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: Path) -> "PatientRuntimeConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


class PatientRuntimeStore:
    _lock_registry: dict[str, threading.RLock] = {}
    _lock_registry_guard = threading.Lock()

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self._shared_lock_for(self.db_path)
        self._initialize()

    @classmethod
    def _shared_lock_for(cls, db_path: Path) -> threading.RLock:
        key = str(db_path.expanduser().resolve())
        with cls._lock_registry_guard:
            return cls._lock_registry.setdefault(key, threading.RLock())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_status (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    daemon_status TEXT NOT NULL,
                    paused INTEGER NOT NULL,
                    pid INTEGER,
                    algorithm_name TEXT,
                    algorithm_path TEXT,
                    workspace TEXT,
                    mode TEXT,
                    speed REAL,
                    scenario_profile TEXT,
                    active_seed INTEGER,
                    time_step_minutes INTEGER,
                    api_host TEXT,
                    api_port INTEGER,
                    started_at_utc TEXT,
                    updated_at_utc TEXT,
                    simulated_minutes INTEGER,
                    simulated_clock TEXT,
                    simulated_day INTEGER,
                    last_glucose_mgdl REAL,
                    last_delivered_insulin_units REAL,
                    last_safety_reason TEXT,
                    last_event_summary TEXT,
                    message TEXT
                )
                """
            )
            existing_columns = {
                str(row["name"]) for row in conn.execute("PRAGMA table_info(runtime_status)").fetchall()
            }
            if "scenario_profile" not in existing_columns:
                conn.execute("ALTER TABLE runtime_status ADD COLUMN scenario_profile TEXT")
            if "active_seed" not in existing_columns:
                conn.execute("ALTER TABLE runtime_status ADD COLUMN active_seed INTEGER")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at_utc TEXT NOT NULL,
                    simulated_minutes INTEGER NOT NULL,
                    simulated_clock TEXT NOT NULL,
                    glucose_mgdl REAL,
                    delivered_insulin_units REAL,
                    recommended_insulin_units REAL,
                    carbs_grams REAL,
                    safety_triggered INTEGER NOT NULL,
                    safety_reason TEXT,
                    event_summary TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    processed_at_utc TEXT,
                    result_json TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO runtime_status (
                    id, daemon_status, paused, updated_at_utc, simulated_minutes, simulated_clock, simulated_day
                ) VALUES (1, 'stopped', 0, ?, 0, 'Day 1 00:00', 1)
                """,
                (_now_utc(),),
            )
            conn.commit()

    def update_status(self, **fields: Any) -> None:
        allowed = {
            "daemon_status", "paused", "pid", "algorithm_name", "algorithm_path", "workspace", "mode", "speed",
            "scenario_profile", "active_seed", "time_step_minutes", "api_host", "api_port", "started_at_utc", "updated_at_utc", "simulated_minutes",
            "simulated_clock", "simulated_day", "last_glucose_mgdl", "last_delivered_insulin_units",
            "last_safety_reason", "last_event_summary", "message",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if "updated_at_utc" not in updates:
            updates["updated_at_utc"] = _now_utc()
        query_map = {
            "daemon_status": "UPDATE runtime_status SET daemon_status = ? WHERE id = 1",
            "paused": "UPDATE runtime_status SET paused = ? WHERE id = 1",
            "pid": "UPDATE runtime_status SET pid = ? WHERE id = 1",
            "algorithm_name": "UPDATE runtime_status SET algorithm_name = ? WHERE id = 1",
            "algorithm_path": "UPDATE runtime_status SET algorithm_path = ? WHERE id = 1",
            "workspace": "UPDATE runtime_status SET workspace = ? WHERE id = 1",
            "mode": "UPDATE runtime_status SET mode = ? WHERE id = 1",
            "speed": "UPDATE runtime_status SET speed = ? WHERE id = 1",
            "scenario_profile": "UPDATE runtime_status SET scenario_profile = ? WHERE id = 1",
            "active_seed": "UPDATE runtime_status SET active_seed = ? WHERE id = 1",
            "time_step_minutes": "UPDATE runtime_status SET time_step_minutes = ? WHERE id = 1",
            "api_host": "UPDATE runtime_status SET api_host = ? WHERE id = 1",
            "api_port": "UPDATE runtime_status SET api_port = ? WHERE id = 1",
            "started_at_utc": "UPDATE runtime_status SET started_at_utc = ? WHERE id = 1",
            "updated_at_utc": "UPDATE runtime_status SET updated_at_utc = ? WHERE id = 1",
            "simulated_minutes": "UPDATE runtime_status SET simulated_minutes = ? WHERE id = 1",
            "simulated_clock": "UPDATE runtime_status SET simulated_clock = ? WHERE id = 1",
            "simulated_day": "UPDATE runtime_status SET simulated_day = ? WHERE id = 1",
            "last_glucose_mgdl": "UPDATE runtime_status SET last_glucose_mgdl = ? WHERE id = 1",
            "last_delivered_insulin_units": "UPDATE runtime_status SET last_delivered_insulin_units = ? WHERE id = 1",
            "last_safety_reason": "UPDATE runtime_status SET last_safety_reason = ? WHERE id = 1",
            "last_event_summary": "UPDATE runtime_status SET last_event_summary = ? WHERE id = 1",
            "message": "UPDATE runtime_status SET message = ? WHERE id = 1",
        }
        with self._lock, closing(self._connect()) as conn:
            for key, value in updates.items():
                conn.execute(query_map[key], (value,))
            conn.commit()

    def read_status(self) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM runtime_status WHERE id = 1").fetchone()
        return dict(row) if row is not None else {}

    def append_reading(self, payload: dict[str, Any], *, event_summary: str = "") -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO readings (
                    recorded_at_utc, simulated_minutes, simulated_clock, glucose_mgdl,
                    delivered_insulin_units, recommended_insulin_units, carbs_grams,
                    safety_triggered, safety_reason, event_summary, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_utc(),
                    int(payload.get("time_minutes", 0)),
                    str(payload.get("simulated_clock", "Day 1 00:00")),
                    float(payload.get("glucose_actual_mgdl", 0.0) or 0.0),
                    float(payload.get("delivered_insulin_units", 0.0) or 0.0),
                    float(payload.get("algo_recommended_insulin_units", 0.0) or 0.0),
                    float(payload.get("carb_intake_grams", 0.0) or 0.0),
                    1 if bool(payload.get("safety_triggered", False)) else 0,
                    str(payload.get("safety_reason", "") or ""),
                    event_summary,
                    json.dumps(payload, sort_keys=True),
                ),
            )
            conn.commit()

    def get_recent_readings(self, limit: int = 288) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM readings ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_latest_reading(self) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row is not None else None

    def enqueue_command(self, command: str, payload: dict[str, Any] | None = None) -> int:
        with self._lock, closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO commands (command, payload_json, status, created_at_utc) VALUES (?, ?, 'pending', ?)",
                (command, json.dumps(payload or {}, sort_keys=True), _now_utc()),
            )
            conn.commit()
            lastrowid = cursor.lastrowid
            return int(lastrowid) if lastrowid is not None else 0

    def fetch_pending_commands(self) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM commands WHERE status = 'pending' ORDER BY id ASC"
            ).fetchall()
            command_ids = [int(row["id"]) for row in rows]
            if command_ids:
                conn.executemany(
                    "UPDATE commands SET status = 'processing' WHERE id = ? AND status = 'pending'",
                    [(command_id,) for command_id in command_ids],
                )
            conn.commit()
        commands: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"] or "{}")
            commands.append({**dict(row), "status": "processing", "payload": payload})
        return commands

    def complete_command(self, command_id: int, *, status: str, result: dict[str, Any] | None = None) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                "UPDATE commands SET status = ?, processed_at_utc = ?, result_json = ? WHERE id = ?",
                (status, _now_utc(), json.dumps(result or {}, sort_keys=True), int(command_id)),
            )
            conn.commit()

    def await_command(self, command_id: int, timeout_seconds: float = 5.0) -> dict[str, Any] | None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute("SELECT * FROM commands WHERE id = ?", (int(command_id),)).fetchone()
            if row is None:
                return None
            if row["status"] in {"done", "failed"}:
                payload = dict(row)
                payload["result"] = json.loads(row["result_json"] or "{}")
                return payload
            time.sleep(0.1)
        return None

    def clear_runtime_data(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute("DELETE FROM readings")
            conn.execute("DELETE FROM commands WHERE status != 'pending'")
            conn.commit()

    def build_audit_summary(self) -> dict[str, Any]:
        with self._lock, closing(self._connect()) as conn:
            total_steps = int(conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0])
            total_overrides = int(conn.execute("SELECT COUNT(*) FROM readings WHERE safety_triggered = 1").fetchone()[0])
            reasons_rows = conn.execute(
                "SELECT safety_reason, COUNT(*) AS count FROM readings WHERE safety_reason != '' GROUP BY safety_reason ORDER BY count DESC"
            ).fetchall()
        return {
            "total_steps": total_steps,
            "total_overrides": total_overrides,
            "top_reasons": {str(row["safety_reason"]): int(row["count"]) for row in reasons_rows},
        }


class DailySchedulePlanner:
    def __init__(self, templates: Iterable[DailyEventTemplate]) -> None:
        self.templates = list(templates)
        self._scheduled_days: set[int] = set()

    def set_templates(self, templates: Iterable[DailyEventTemplate]) -> None:
        self.templates = list(templates)
        self.reset()

    def reset(self) -> None:
        self._scheduled_days.clear()

    def schedule_for_time(self, simulator: Simulator, current_time_minutes: int) -> str:
        day_index = int(current_time_minutes // 1440)
        if day_index in self._scheduled_days:
            return self.describe_clock(current_time_minutes)
        base = day_index * 1440
        for template in self.templates:
            simulator.add_stress_event(
                StressEvent(
                    start_time=base + template.minute_of_day,
                    event_type=template.event_type,
                    value=template.value,
                    reported_value=template.reported_value,
                    duration=template.duration,
                    absorption_delay_minutes=template.absorption_delay_minutes,
                    isf=template.isf,
                    icr=template.icr,
                    basal_rate=template.basal_rate,
                    dia_minutes=template.dia_minutes,
                )
            )
        self._scheduled_days.add(day_index)
        return self.describe_clock(current_time_minutes)

    def event_labels_for_time(self, current_time_minutes: int) -> list[str]:
        minute_of_day = int(current_time_minutes % 1440)
        return [template.label for template in self.templates if template.label and template.minute_of_day == minute_of_day]

    @staticmethod
    def describe_clock(current_time_minutes: int) -> str:
        day_index = int(current_time_minutes // 1440) + 1
        minute_of_day = int(current_time_minutes % 1440)
        hour = minute_of_day // 60
        minute = minute_of_day % 60
        return f"Day {day_index} {hour:02d}:{minute:02d}"


class LivePatientDaemon:
    def __init__(self, config: PatientRuntimeConfig) -> None:
        self.config = config
        self.store = PatientRuntimeStore(config.db_path)
        self.stop_requested = False
        self.paused = False
        self.algorithm_instance: InsulinAlgorithm | None = None
        self.simulator: Simulator | None = None
        self.generator: Any = None
        self.profile = get_runtime_scenario_profile(config.scenario_profile)
        self.active_seed = self._resolve_effective_seed(self.profile, config.seed)
        self.planner = DailySchedulePlanner(self.profile.templates)
        self._next_step_due = time.monotonic()
        self._run_id = generate_run_id(resolve_seed(self.active_seed))
        self._server: Any = None
        self._server_thread: Any = None
        self._latest_event_summary = ""
        self._ensure_workspace()

    def _resolve_effective_seed(self, profile: RuntimeScenarioProfile, seed: int | None) -> int:
        return int(seed if seed is not None else profile.default_seed)

    def _apply_runtime_profile(self, scenario_profile: str, seed: int | None = None) -> None:
        self.profile = get_runtime_scenario_profile(scenario_profile)
        self.active_seed = self._resolve_effective_seed(self.profile, seed)
        self.config = replace(self.config, scenario_profile=self.profile.name, seed=self.active_seed)
        self.planner.set_templates(self.profile.templates)
        self._run_id = generate_run_id(resolve_seed(self.active_seed))

    def _ensure_workspace(self) -> None:
        self.config.workspace_path.mkdir(parents=True, exist_ok=True)
        self.config.bundle_dir.mkdir(parents=True, exist_ok=True)

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        logger.info("Live patient daemon received signal %s", signum)
        self.stop_requested = True

    def _resolve_patient_config(self) -> dict[str, Any]:
        raw = self.config.patient_config
        path = Path(raw).expanduser()
        if path.is_file():
            return load_patient_config(path).model_dump()
        if raw.endswith(".yaml"):
            candidate = self.config.workspace_path / raw
            if candidate.is_file():
                return load_patient_config(candidate).model_dump()
        if isinstance(raw, str):
            return load_patient_config_by_name(raw).model_dump()
        if isinstance(raw, PatientProfile):
            return validate_patient_config_dict(raw.to_patient_config()).model_dump()
        raise ValueError(f"Unsupported patient configuration: {raw}")

    def _build_simulator(self) -> Simulator:
        patient_params = self._resolve_patient_config()
        self.algorithm_instance = _load_algorithm_instance_silent(self.config.algo_file)
        patient_model = PatientFactory.create_patient(patient_type=self.config.patient_model_type, **patient_params)
        sensor_model = SensorModel(noise_std=7.0, lag_minutes=10, dropout_prob=0.0, bias=0.0, seed=self.active_seed)
        simulator = Simulator(
            patient_model=patient_model,
            algorithm=self.algorithm_instance,
            time_step=self.config.time_step_minutes,
            seed=self.active_seed,
            safety_config=SafetyConfig(),
            sensor_model=sensor_model,
        )
        self._write_static_bundle_files(patient_params)
        return simulator

    def _write_static_bundle_files(self, patient_params: dict[str, Any]) -> None:
        if self.algorithm_instance is None:
            raise RuntimeError("Algorithm instance unavailable.")
        bundle_dir = self.config.bundle_dir
        config_payload = {
            "algorithm": {
                "class": f"{self.algorithm_instance.__class__.__module__}.{self.algorithm_instance.__class__.__name__}",
                "metadata": self.algorithm_instance.get_algorithm_metadata().to_dict(),
            },
            "patient_config": patient_params,
            "patient_model_type": self.config.patient_model_type,
            "scenario": {
                "scenario_name": "Digital Patient Live Runtime",
                "profile": self.profile.name,
                "description": self.profile.description,
                "warm_start_minutes": self.profile.warm_start_minutes,
            },
            "mode": self.config.mode,
            "speed": self.config.speed,
            "seed": self.active_seed,
            "time_step_minutes": self.config.time_step_minutes,
            "api": {"host": self.config.api_host, "port": self.config.api_port},
        }
        write_json(bundle_dir / "config.json", config_payload)
        run_metadata = build_run_metadata(self._run_id, resolve_seed(self.active_seed), config_payload, bundle_dir)
        write_json(bundle_dir / "run_metadata.json", run_metadata)
        if not (bundle_dir / "results.csv").exists():
            with (bundle_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "time_minutes",
                    "glucose_actual_mgdl",
                    "glucose_to_algo_mgdl",
                    "glucose_trend_mgdl_min",
                    "predicted_glucose_30min",
                    "algo_recommended_insulin_units",
                    "delivered_insulin_units",
                    "carb_intake_grams",
                    "patient_iob_units",
                    "patient_cob_grams",
                    "safety_triggered",
                    "safety_reason",
                    "simulated_clock",
                ])
                writer.writeheader()
        (bundle_dir / "audit").mkdir(exist_ok=True)

    def bootstrap(self, *, reset: bool = False) -> None:
        self._apply_runtime_profile(self.config.scenario_profile, self.config.seed)
        self.store.update_status(
            daemon_status="starting",
            paused=0,
            pid=os.getpid(),
            workspace=str(self.config.workspace_path),
            mode=self.config.mode,
            speed=self.config.speed,
            scenario_profile=self.profile.name,
            active_seed=self.active_seed,
            time_step_minutes=self.config.time_step_minutes,
            api_host=self.config.api_host,
            api_port=self.config.api_port,
            started_at_utc=_now_utc(),
            message="Bootstrapping digital patient runtime.",
        )
        self.simulator = self._build_simulator()
        self.planner.reset()
        if not reset and self.config.snapshot_path.is_file():
            try:
                snapshot = json.loads(self.config.snapshot_path.read_text(encoding="utf-8"))
                self.simulator.load_state(snapshot)
            except Exception as exc:
                logger.warning("Could not resume from snapshot: %s", exc)
        else:
            self._reset_runtime_files()
        self.generator = self.simulator.run_live(10**9)
        if reset or not self.config.snapshot_path.is_file():
            self._prime_profile_start()
        self._next_step_due = time.monotonic()
        algo_name = self.algorithm_instance.get_algorithm_metadata().name if self.algorithm_instance else "unknown"
        self.store.update_status(
            daemon_status="running",
            paused=0,
            algorithm_name=algo_name,
            algorithm_path=str(self.config.algo_file),
            scenario_profile=self.profile.name,
            active_seed=self.active_seed,
            message=f"Digital patient online at {self.config.dashboard_url}",
        )
        self._write_pid_file()
        self._refresh_bundle_manifest()

    def _write_pid_file(self) -> None:
        self.config.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self.config.config_path.write_text(json.dumps(self.config.to_json(), indent=2, sort_keys=True), encoding="utf-8")

    def _reset_runtime_files(self) -> None:
        self.store.clear_runtime_data()
        if self.config.snapshot_path.exists():
            self.config.snapshot_path.unlink()
        bundle_dir = self.config.bundle_dir
        for relative in ["results.csv", "run_manifest.json", "audit/audit_summary.json"]:
            path = bundle_dir / relative
            if path.exists():
                path.unlink()
        self._write_static_bundle_files(self._resolve_patient_config())

    def _event_summary_for_time(self, current_time_minutes: int) -> str:
        labels: list[str] = []
        scheduled_labels = self.planner.event_labels_for_time(current_time_minutes)
        if scheduled_labels:
            labels.extend(scheduled_labels)
        if self._latest_event_summary:
            labels.append(self._latest_event_summary)
        return " | ".join(labels)

    def _update_live_status_from_record(self, record: dict[str, Any], *, event_summary: str, message: str) -> None:
        simulated_minutes = int(record.get("time_minutes", 0))
        simulated_day = int(simulated_minutes // 1440) + 1
        self.store.update_status(
            daemon_status="running",
            paused=1 if self.paused else 0,
            scenario_profile=self.profile.name,
            active_seed=self.active_seed,
            simulated_minutes=simulated_minutes,
            simulated_clock=str(record.get("simulated_clock", "Day 1 00:00")),
            simulated_day=simulated_day,
            last_glucose_mgdl=float(record.get("glucose_actual_mgdl", 0.0) or 0.0),
            last_delivered_insulin_units=float(record.get("delivered_insulin_units", 0.0) or 0.0),
            last_safety_reason=str(record.get("safety_reason", "") or ""),
            last_event_summary=event_summary,
            message=message,
        )

    def _persist_record(self, record: dict[str, Any], *, event_summary: str, message: str, save_snapshot: bool = True) -> None:
        self.store.append_reading(record, event_summary=event_summary)
        self._record_to_csv(record)
        self._update_live_status_from_record(record, event_summary=event_summary, message=message)
        if save_snapshot:
            self._save_snapshot(record)

    def _prime_profile_start(self) -> None:
        if self.generator is None or self.profile.warm_start_minutes <= 0:
            self.store.update_status(
                scenario_profile=self.profile.name,
                active_seed=self.active_seed,
                last_event_summary=self.profile.reset_note or "",
            )
            return

        target_minutes = int(self.profile.warm_start_minutes)
        max_minutes = target_minutes + (180 if self.profile.require_recent_override else 0)
        recent_records: list[tuple[dict[str, Any], str]] = []
        recent_override_seen = False
        last_record: dict[str, Any] | None = None
        last_progress_minutes: int | None = None
        stagnant_iterations = 0
        max_iterations = max(1000, (max_minutes // max(self.config.time_step_minutes, 1)) * 20)
        iterations = 0

        while True:
            iterations += 1
            if iterations > max_iterations:
                raise SimulationLimitError(
                    f"Warm-start exceeded {max_iterations} iterations without reaching the requested profile state.",
                    float(last_progress_minutes or 0),
                    float(last_record.get("glucose_actual_mgdl", 0.0) if last_record is not None else 0.0),
                    float(max_minutes),
                )
            simulated_clock = self._schedule_day()
            record = next(self.generator)
            record["simulated_clock"] = simulated_clock
            event_summary = self._event_summary_for_time(int(record.get("time_minutes", 0)))
            recent_records.append((record, event_summary))
            if len(recent_records) > max(1, self.profile.warmup_history_limit):
                recent_records.pop(0)
            recent_override_seen = recent_override_seen or bool(record.get("safety_triggered", False))
            last_record = record
            self._latest_event_summary = ""

            current_minutes = int(record.get("time_minutes", 0))
            if last_progress_minutes is not None and current_minutes <= last_progress_minutes:
                stagnant_iterations += 1
                if stagnant_iterations >= 25:
                    raise SimulationLimitError(
                        "Warm-start generator stopped advancing simulated time; aborting to avoid a CPU spin.",
                        float(current_minutes),
                        float(record.get("glucose_actual_mgdl", 0.0) or 0.0),
                        float(max_minutes),
                    )
            else:
                stagnant_iterations = 0
            last_progress_minutes = current_minutes
            if current_minutes < target_minutes:
                continue
            if self.profile.require_recent_override and not recent_override_seen and current_minutes < max_minutes:
                continue
            break

        for stored_record, event_summary in recent_records:
            self.store.append_reading(stored_record, event_summary=event_summary)
            self._record_to_csv(stored_record)

        if last_record is not None:
            self._save_snapshot(last_record)
            last_summary = recent_records[-1][1] if recent_records else self.profile.reset_note
            self._update_live_status_from_record(
                last_record,
                event_summary=last_summary or self.profile.reset_note,
                message=self.profile.reset_note or f"{self.profile.name} warm-start complete.",
            )

    def _wall_step_seconds(self) -> float:
        if self.config.mode == "real-time":
            return float(self.config.time_step_minutes * 60)
        speed = max(float(self.config.speed), 1.0)
        return float(self.config.time_step_minutes * 60.0 / speed)

    def _record_to_csv(self, record: dict[str, Any]) -> None:
        row = {
            key: record.get(key)
            for key in [
                "time_minutes",
                "glucose_actual_mgdl",
                "glucose_to_algo_mgdl",
                "glucose_trend_mgdl_min",
                "predicted_glucose_30min",
                "algo_recommended_insulin_units",
                "delivered_insulin_units",
                "carb_intake_grams",
                "patient_iob_units",
                "patient_cob_grams",
                "safety_triggered",
                "safety_reason",
                "simulated_clock",
            ]
        }
        with (self.config.bundle_dir / "results.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _refresh_bundle_manifest(self) -> None:
        bundle_dir = self.config.bundle_dir
        audit_summary_path = bundle_dir / "audit" / "audit_summary.json"
        write_json(audit_summary_path, self.store.build_audit_summary())
        manifest = build_run_manifest(
            bundle_dir,
            {
                "config": bundle_dir / "config.json",
                "run_metadata": bundle_dir / "run_metadata.json",
                "results_csv": bundle_dir / "results.csv",
                "audit_summary": audit_summary_path,
            },
        )
        write_json(bundle_dir / "run_manifest.json", manifest)

    def _save_snapshot(self, record: dict[str, Any]) -> None:
        if self.simulator is None:
            return
        snapshot = self.simulator.save_state()
        snapshot["current_time"] = int(record.get("time_minutes", 0)) + self.config.time_step_minutes
        self.config.snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")

    def _schedule_day(self) -> str:
        if self.simulator is None:
            return "Day 1 00:00"
        current_time = int(getattr(self.simulator, "_current_time", 0))
        return self.planner.schedule_for_time(self.simulator, current_time)

    def _process_commands(self) -> None:
        if self.simulator is None:
            return
        for command in self.store.fetch_pending_commands():
            name = str(command["command"])
            payload = command["payload"]
            result: dict[str, Any] = {}
            try:
                if name == "pause":
                    self.paused = True
                    self.store.update_status(daemon_status="paused", paused=1, message="Simulation paused.")
                    result = {"paused": True}
                elif name == "resume":
                    self.paused = False
                    self._next_step_due = time.monotonic()
                    self.store.update_status(daemon_status="running", paused=0, message="Simulation resumed.")
                    result = {"paused": False}
                elif name == "inject_meal":
                    carbs = float(payload.get("carbs", 0.0))
                    if carbs <= 0.0:
                        raise ValueError("Meal carbs must be positive.")
                    current_time = int(getattr(self.simulator, "_current_time", 0))
                    self.simulator.add_stress_event(
                        StressEvent(
                            start_time=current_time,
                            event_type="meal",
                            value=carbs,
                            reported_value=carbs,
                        )
                    )
                    self._latest_event_summary = f"Manual meal injected: {carbs:.0f} g carbs"
                    self.store.update_status(last_event_summary=self._latest_event_summary)
                    result = {"meal_carbs": carbs, "scheduled_at_minute": current_time}
                elif name == "expo_reset":
                    target_profile = str(payload.get("scenario_profile") or "expo_hot_start")
                    target_seed = payload.get("seed")
                    resolved_seed = int(target_seed) if target_seed is not None else None
                    self._apply_runtime_profile(target_profile, resolved_seed)
                    self.simulator = self._build_simulator()
                    self.paused = False
                    self._latest_event_summary = self.profile.reset_note or "Expo mode reset complete."
                    self.store.clear_runtime_data()
                    self._reset_runtime_files()
                    self.generator = self.simulator.run_live(10**9)
                    self._prime_profile_start()
                    self._next_step_due = time.monotonic()
                    self.store.update_status(
                        daemon_status="running",
                        paused=0,
                        scenario_profile=self.profile.name,
                        active_seed=self.active_seed,
                        message=self.profile.reset_note or "Expo mode reset complete.",
                    )
                    self._write_pid_file()
                    self._refresh_bundle_manifest()
                    result = {"reset": True, "scenario_profile": self.profile.name, "active_seed": self.active_seed}
                elif name == "stop":
                    self.stop_requested = True
                    self.store.update_status(daemon_status="stopping", message="Stop requested.")
                    result = {"stopping": True}
                else:
                    raise ValueError(f"Unknown command: {name}")
                self.store.complete_command(int(command["id"]), status="done", result=result)
            except Exception as exc:
                self.store.complete_command(int(command["id"]), status="failed", result={"error": str(exc)})

    def advance_once(self) -> dict[str, Any]:
        if self.generator is None:
            raise RuntimeError("Live patient generator is not initialized.")
        simulated_clock = self._schedule_day()
        record = next(self.generator)
        record["simulated_clock"] = simulated_clock
        event_summary = self._event_summary_for_time(int(record.get("time_minutes", 0)))
        self._persist_record(record, event_summary=event_summary, message="Digital patient running.")
        self._refresh_bundle_manifest()
        self._latest_event_summary = ""
        return record

    def run(self, *, max_steps: int | None = None) -> None:
        step_count = 0
        while not self.stop_requested:
            self._process_commands()
            if self.paused:
                time.sleep(0.1)
                continue
            if time.monotonic() < self._next_step_due:
                time.sleep(min(0.1, self._next_step_due - time.monotonic()))
                continue
            try:
                self.advance_once()
            except StopIteration:
                self.stop_requested = True
                self.store.update_status(daemon_status="stopped", message="Simulation completed.")
                break
            except SimulationLimitError as exc:
                self.stop_requested = True
                self.store.update_status(daemon_status="error", message=str(exc))
                break
            step_count += 1
            self._next_step_due = time.monotonic() + self._wall_step_seconds()
            if max_steps is not None and step_count >= max_steps:
                self.stop_requested = True
        self.shutdown()

    def shutdown(self) -> None:
        self.store.update_status(daemon_status="stopped", paused=1 if self.paused else 0, message="Digital patient stopped.")
        if self.config.pid_path.exists():
            self.config.pid_path.unlink()


def is_process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def load_runtime_status(workspace: Path) -> dict[str, Any]:
    db_path = workspace / "patient_state.db"
    if not db_path.is_file():
        return {}
    store = PatientRuntimeStore(db_path)
    return store.read_status()
