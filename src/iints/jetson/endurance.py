from __future__ import annotations

import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

import numpy as np
import pandas as pd
import yaml

from iints.api.base_algorithm import InsulinAlgorithm
from iints.core.devices.models import create_sensor_model
from iints.core.patient.patient_factory import PatientFactory
from iints.core.safety import SafetyConfig
from iints.core.simulator import Simulator, StressEvent
from iints.utils.run_io import compute_sha256


ENDURANCE_PROFILES = {
    "normal",
    "stress",
    "adversarial",
    "mixed_adversarial",
    "sensor_failure",
    "nighttime_risk",
    "custom",
}

ENDURANCE_EXECUTION_MODES = {"accelerated", "wall_clock"}


class JetsonEnduranceError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnduranceConfig:
    algo_path: str
    predictor_path: Optional[str]
    duration: str
    duration_minutes: int
    time_step_minutes: int
    output_dir: str
    profile: str
    seed: Optional[int]
    patient_model: str = "bergman"
    sensor_profile: str = "free_living_cgm"
    custom_profile_path: Optional[str] = None
    resume: bool = False
    checkpoint_interval_minutes: int = 360
    hardware_sample_interval_minutes: int = 60
    status_interval_steps: int = 25
    execution_mode: str = "accelerated"
    research_export: bool = True

    @property
    def expected_steps(self) -> int:
        return max(1, self.duration_minutes // self.time_step_minutes)

    @property
    def simulator_end_minutes(self) -> int:
        return max(0, self.duration_minutes - self.time_step_minutes)

    @property
    def wall_clock_target_seconds(self) -> int:
        return self.duration_minutes * 60 if self.execution_mode == "wall_clock" else 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_duration_to_minutes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([mhdw])\s*", value.lower())
    if not match:
        raise JetsonEnduranceError("Duration must look like 30m, 1h, 24h, 7d, or 2w.")
    amount = int(match.group(1))
    unit = match.group(2)
    multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    minutes = amount * multipliers[unit]
    if minutes <= 0:
        raise JetsonEnduranceError("Duration must be greater than zero.")
    return minutes


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    if isinstance(value, tuple):
        return [_safe_json(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(payload), indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_command(command: List[str], *, timeout_seconds: float = 2.0) -> Dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "stdout": "", "stderr": "not found"}
    try:
        completed = subprocess.run(  # noqa: S603 - executable is resolved via PATH and no shell is used.
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "available": True,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "command": command,
            "timeout": True,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
        }


def _read_thermal_zones() -> List[Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        temp_path = zone / "temp"
        type_path = zone / "type"
        if not temp_path.is_file():
            continue
        try:
            raw = float(temp_path.read_text(encoding="utf-8").strip())
        except Exception:
            continue
        label = type_path.read_text(encoding="utf-8").strip() if type_path.is_file() else zone.name
        celsius = raw / 1000.0 if raw > 1000 else raw
        zones.append({"zone": zone.name, "type": label, "temp_c": round(celsius, 2)})
    return zones


def collect_jetson_hardware_info() -> Dict[str, Any]:
    nvidia_smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    tegrastats = _run_command(["tegrastats", "--interval", "1000"], timeout_seconds=1.5)
    thermal_zones = _read_thermal_zones()
    is_jetson_like = bool(tegrastats.get("available")) or any(
        "gpu" in str(zone.get("type", "")).lower() for zone in thermal_zones
    )
    return {
        "captured_at_utc": utc_now_iso(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "is_jetson_like": is_jetson_like,
        "cuda_available": bool(nvidia_smi.get("available")) or bool(tegrastats.get("available")),
        "nvidia_smi": nvidia_smi,
        "tegrastats": tegrastats,
        "thermal_zones": thermal_zones,
    }


def _hardware_metric_row(info: Dict[str, Any], *, elapsed_minutes: int) -> Dict[str, Any]:
    return {
        "captured_at_utc": info["captured_at_utc"],
        "elapsed_minutes": elapsed_minutes,
        "is_jetson_like": bool(info.get("is_jetson_like")),
        "cuda_available": bool(info.get("cuda_available")),
        "thermal_zones_json": json.dumps(info.get("thermal_zones", [])),
    }


def _event(start_time: int, event_type: str, **kwargs: Any) -> Dict[str, Any]:
    payload = {"start_time": int(start_time), "event_type": event_type}
    payload.update({key: value for key, value in kwargs.items() if value is not None})
    return payload


def _normal_day_events(day_offset: int) -> List[Dict[str, Any]]:
    return [
        _event(day_offset + 450, "meal", value=45, reported_value=45, duration=45, absorption_delay_minutes=10),
        _event(day_offset + 735, "meal", value=65, reported_value=62, duration=75, absorption_delay_minutes=15),
        _event(day_offset + 1095, "meal", value=78, reported_value=74, duration=90, absorption_delay_minutes=20),
        _event(day_offset + 1290, "meal", value=18, reported_value=18, duration=30, absorption_delay_minutes=5),
    ]


def _adversarial_day_events(day_offset: int, day_index: int) -> List[Dict[str, Any]]:
    pattern = day_index % 8
    if pattern == 0:
        return [
            _event(day_offset + 420, "meal", value=70, reported_value=70, absorption_delay_minutes=15),
            _event(day_offset + 445, "sensor_error", value=82),
            _event(day_offset + 450, "sensor_error", value=82),
            _event(day_offset + 455, "sensor_error", value=82),
        ]
    if pattern == 1:
        return [
            _event(day_offset + 705, "meal", value=82, reported_value=62, absorption_delay_minutes=20),
            _event(day_offset + 785, "meal", value=58, reported_value=35, absorption_delay_minutes=10),
        ]
    if pattern == 2:
        return [
            _event(day_offset + 540, "ratio_change", isf=75, duration=240),
            _event(day_offset + 555, "meal", value=64, reported_value=64, absorption_delay_minutes=15),
        ]
    if pattern == 3:
        return [
            _event(day_offset + 840, "meal", value=54, reported_value=32, absorption_delay_minutes=15),
            _event(day_offset + 900, "exercise", value=0.65, duration=75),
        ]
    if pattern == 4:
        return [
            _event(day_offset + 90, "ratio_change", basal_rate=0.25, duration=240),
            _event(day_offset + 240, "sensor_error", value=128),
        ]
    if pattern == 5:
        return [
            _event(day_offset + 720, "sensor_error", value=250),
            _event(day_offset + 725, "sensor_error", value=48),
            _event(day_offset + 730, "sensor_error", value=265),
            _event(day_offset + 735, "sensor_error", value=52),
            _event(day_offset + 740, "sensor_error", value=240),
        ]
    if pattern == 6:
        return [
            _event(day_offset + 0, "meal", value=42, reported_value=0, absorption_delay_minutes=25),
            _event(day_offset + 45, "sensor_error", value=210),
        ]
    return [
        _event(day_offset + 680, "meal", value=70, reported_value=70, absorption_delay_minutes=15),
        _event(day_offset + 705, "sensor_error", value=260),
        _event(day_offset + 730, "sensor_error", value=260),
        _event(day_offset + 755, "meal", value=60, reported_value=45, absorption_delay_minutes=15),
    ]


def _profile_events(profile: str, duration_minutes: int, custom_profile_path: Optional[str]) -> List[Dict[str, Any]]:
    if profile == "custom":
        if not custom_profile_path:
            raise JetsonEnduranceError("--custom-profile is required when --profile custom is used.")
        payload = yaml.safe_load(Path(custom_profile_path).read_text(encoding="utf-8")) or {}
        return [dict(event) for event in payload.get("stress_events", [])]

    days = max(1, math.ceil(duration_minutes / 1440))
    events: List[Dict[str, Any]] = []
    for day_index in range(days):
        offset = day_index * 1440
        if profile in {"normal", "stress", "mixed_adversarial", "nighttime_risk"}:
            events.extend(_normal_day_events(offset))
        if profile in {"stress", "mixed_adversarial"}:
            events.extend(
                [
                    _event(offset + 870, "exercise", value=0.45, duration=50),
                    _event(offset + 1040, "meal", value=28, reported_value=18, absorption_delay_minutes=10),
                ]
            )
        if profile in {"adversarial", "mixed_adversarial"}:
            events.extend(_adversarial_day_events(offset, day_index))
        if profile == "sensor_failure":
            for minute in range(offset + 700, offset + 760, 10):
                events.append(_event(minute, "sensor_error", value=55 if minute % 20 == 0 else 245))
        if profile == "nighttime_risk":
            events.extend(
                [
                    _event(offset + 60, "ratio_change", isf=85, basal_rate=0.3, duration=330),
                    _event(offset + 180, "sensor_error", value=105),
                ]
            )
    return [event for event in events if int(event["start_time"]) < duration_minutes]


def _stress_events_from_payloads(payloads: Iterable[Dict[str, Any]]) -> List[StressEvent]:
    return [StressEvent(**payload) for payload in payloads]


def _build_simulator(
    *,
    algorithm: InsulinAlgorithm,
    predictor: Optional[object],
    config: EnduranceConfig,
    stress_events: List[Dict[str, Any]],
) -> Simulator:
    patient_model = PatientFactory.create_patient(patient_type=config.patient_model)
    sensor_model = create_sensor_model(profile=config.sensor_profile, seed=config.seed)
    simulator = Simulator(
        patient_model=patient_model,
        algorithm=algorithm,
        time_step=config.time_step_minutes,
        seed=config.seed,
        safety_config=SafetyConfig(),
        predictor=predictor,
        sensor_model=sensor_model,
        enable_profiling=True,
    )
    for event in _stress_events_from_payloads(stress_events):
        simulator.add_stress_event(event)
    return simulator


def _status_payload(
    *,
    config: EnduranceConfig,
    status: str,
    started_at_utc: str,
    completed_steps: int,
    current_record: Optional[Dict[str, Any]] = None,
    message: str = "",
    wall_elapsed_seconds: Optional[float] = None,
    hardware_latest: Optional[Dict[str, Any]] = None,
    last_checkpoint_minute: Optional[int] = None,
    resume_count: int = 0,
) -> Dict[str, Any]:
    progress = completed_steps / max(1, config.expected_steps)
    elapsed_minutes = completed_steps * config.time_step_minutes
    steps_per_second = None
    estimated_wall_remaining_seconds = None
    wall_clock_progress_pct = None
    wall_clock_target_seconds = config.wall_clock_target_seconds or None
    if config.execution_mode == "wall_clock" and wall_clock_target_seconds is not None:
        observed_wall_elapsed = float(wall_elapsed_seconds or 0.0)
        estimated_wall_remaining_seconds = max(0.0, wall_clock_target_seconds - observed_wall_elapsed)
        wall_clock_progress_pct = min(100.0, observed_wall_elapsed / wall_clock_target_seconds * 100.0)
    if wall_elapsed_seconds is not None and wall_elapsed_seconds > 0 and completed_steps > 0:
        steps_per_second = completed_steps / wall_elapsed_seconds
        if config.execution_mode != "wall_clock":
            estimated_wall_remaining_seconds = max(0, config.expected_steps - completed_steps) / steps_per_second
    return {
        "status": status,
        "message": message,
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now_iso(),
        "duration": config.duration,
        "duration_minutes": config.duration_minutes,
        "execution_mode": config.execution_mode,
        "profile": config.profile,
        "expected_steps": config.expected_steps,
        "completed_steps": completed_steps,
        "elapsed_minutes": elapsed_minutes,
        "remaining_minutes": max(0, config.duration_minutes - elapsed_minutes),
        "progress_pct": round(min(100.0, progress * 100.0), 3),
        "current_glucose_mgdl": _safe_float(current_record, "glucose_actual_mgdl"),
        "tir_so_far_pct": None,
        "interventions": None,
        "critical_events": None,
        "worst_glucose_mgdl": None,
        "wall_elapsed_seconds": round(wall_elapsed_seconds, 3) if wall_elapsed_seconds is not None else None,
        "wall_clock_target_seconds": wall_clock_target_seconds,
        "wall_clock_progress_pct": round(wall_clock_progress_pct, 3) if wall_clock_progress_pct is not None else None,
        "steps_per_second": round(steps_per_second, 6) if steps_per_second is not None else None,
        "estimated_wall_remaining_seconds": (
            round(estimated_wall_remaining_seconds, 3)
            if estimated_wall_remaining_seconds is not None
            else None
        ),
        "hardware_latest": hardware_latest,
        "last_checkpoint_minute": last_checkpoint_minute,
        "resume_count": resume_count,
    }


def _safe_float(record: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if record is None or key not in record:
        return None
    value = record.get(key)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), 3)


def _coerce_scalar_int(value: object) -> int:
    if isinstance(value, np.generic):
        return int(value.item())
    return int(cast(Any, value))


def _glucose_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df.get("glucose_actual_mgdl", pd.Series(dtype=float)), errors="coerce").dropna()


def _tir_pct(glucose: pd.Series) -> float:
    if glucose.empty:
        return 0.0
    return float(((glucose >= 70.0) & (glucose <= 180.0)).mean() * 100.0)


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _glucose_metrics(glucose: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(glucose, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean_mgdl": None,
            "min_mgdl": None,
            "max_mgdl": None,
            "tir_70_180_pct": 0.0,
            "time_below_70_pct": 0.0,
            "time_below_54_pct": 0.0,
            "time_above_180_pct": 0.0,
            "time_above_250_pct": 0.0,
            "cv_pct": None,
        }
    mean = float(clean.mean())
    return {
        "mean_mgdl": round(mean, 3),
        "min_mgdl": round(float(clean.min()), 3),
        "max_mgdl": round(float(clean.max()), 3),
        "tir_70_180_pct": round(_tir_pct(clean), 3),
        "time_below_70_pct": round(float((clean < 70.0).mean() * 100.0), 3),
        "time_below_54_pct": round(float((clean < 54.0).mean() * 100.0), 3),
        "time_above_180_pct": round(float((clean > 180.0).mean() * 100.0), 3),
        "time_above_250_pct": round(float((clean > 250.0).mean() * 100.0), 3),
        "cv_pct": round(float(clean.std(ddof=0) / mean * 100.0), 3) if mean else None,
    }


def _physiology_quality(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"warning_count": 0, "warnings": [], "truth_metrics": _glucose_metrics(pd.Series(dtype=float))}

    truth = _series(df, "glucose_actual_mgdl")
    mechanistic = _series(df, "glucose_mechanistic_mgdl") if "glucose_mechanistic_mgdl" in df else truth
    algo_glucose = _series(df, "glucose_to_algo_mgdl")
    predicted = _series(df, "predicted_glucose_30min")
    fail_soft = df.get("input_validator_fail_soft", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    safety = df.get("safety_triggered", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    sensor_status = df.get("sensor_status", pd.Series("ok", index=df.index)).astype(str)

    sensor_gap = (truth - algo_glucose).abs()
    hidden_gap = (truth - mechanistic).abs()
    blind_hyper = (truth > 250.0) & (algo_glucose < 180.0)
    false_hypo_like = safety & (truth >= 100.0) & (predicted < 60.0)
    flat_truth = int((truth.diff().fillna(0.0).abs() < 1e-9).sum())
    warnings: List[str] = []
    if int(fail_soft.sum()) > max(3, len(df) * 0.05):
        warnings.append("Many algorithm-facing CGM values required fail-soft validation.")
    if int(blind_hyper.sum()) > 0:
        warnings.append("Algorithm-facing CGM missed one or more severe hyperglycemia truth states.")
    if int(false_hypo_like.sum()) > max(2, len(df) * 0.02):
        warnings.append("Predicted-hypoglycemia alarms look dominated by noisy sensor trend extrapolation.")
    if int((hidden_gap > 1.0).sum()) > 0:
        warnings.append("Reported actual glucose diverged from the mechanistic patient trace.")
    if flat_truth > len(df) * 0.35 and truth.nunique(dropna=True) > 1:
        warnings.append("Truth glucose contains long flat segments; review patient model and validator settings.")

    return {
        "warning_count": len(warnings),
        "warnings": warnings,
        "truth_metrics": _glucose_metrics(truth),
        "mechanistic_metrics": _glucose_metrics(mechanistic),
        "algorithm_input_metrics": _glucose_metrics(algo_glucose),
        "input_validator_fail_soft_rows": int(fail_soft.sum()),
        "sensor_artifact_steps": int((sensor_status != "ok").sum()),
        "mean_abs_truth_sensor_gap_mgdl": round(float(sensor_gap.mean()), 3),
        "max_abs_truth_sensor_gap_mgdl": round(float(sensor_gap.max()), 3),
        "truth_mechanistic_gap_rows": int((hidden_gap > 1.0).sum()),
        "algorithm_blind_hyperglycemia_rows": int(blind_hyper.sum()),
        "false_predicted_hypo_alert_rows": int(false_hypo_like.sum()),
        "flat_truth_rows": flat_truth,
    }


def _hourly_summary(df: pd.DataFrame, time_step_minutes: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["hour_index"] = (pd.to_numeric(work["time_minutes"], errors="coerce") // 60).astype(int)
    rows: List[Dict[str, Any]] = []
    for hour, group in work.groupby("hour_index"):
        hour_index = _coerce_scalar_int(hour)
        glucose = _glucose_series(group)
        algo_glucose = _series(group, "glucose_to_algo_mgdl").dropna()
        interventions = group.get("safety_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool)
        uncertainty = pd.to_numeric(group.get("predictor_uncertainty_std_mgdl", pd.Series(dtype=float)), errors="coerce")
        latency = pd.to_numeric(group.get("algorithm_latency_ms", pd.Series(dtype=float)), errors="coerce")
        fail_soft = group.get("input_validator_fail_soft", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        rows.append(
            {
                "hour_index": hour_index,
                "start_minute": hour_index * 60,
                "mean_glucose_mgdl": round(float(glucose.mean()), 3) if not glucose.empty else None,
                "mean_algorithm_input_mgdl": round(float(algo_glucose.mean()), 3) if not algo_glucose.empty else None,
                "tir_70_180_pct": round(_tir_pct(glucose), 3),
                "time_below_70_pct": round(float((glucose < 70.0).mean() * 100.0), 3) if not glucose.empty else 0.0,
                "time_above_180_pct": round(float((glucose > 180.0).mean() * 100.0), 3) if not glucose.empty else 0.0,
                "time_above_250_pct": round(float((glucose > 250.0).mean() * 100.0), 3) if not glucose.empty else 0.0,
                "supervisor_interventions": int(interventions.sum()),
                "input_validator_fail_soft_rows": int(fail_soft.sum()),
                "predictor_uncertainty_mean_mgdl": round(float(uncertainty.dropna().mean()), 3) if uncertainty.dropna().size else None,
                "gpu_inference_latency_ms": round(float(latency.dropna().mean()), 3) if latency.dropna().size else None,
                "step_count": int(len(group)),
                "time_step_minutes": time_step_minutes,
            }
        )
    return pd.DataFrame(rows)


def _daily_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work["day_index"] = (pd.to_numeric(work["time_minutes"], errors="coerce") // 1440).astype(int)
    summaries: List[Dict[str, Any]] = []
    for day, group in work.groupby("day_index"):
        day_index = _coerce_scalar_int(day)
        glucose = _glucose_series(group)
        interventions = group.get("safety_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool)
        fail_soft = group.get("input_validator_fail_soft", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        critical = glucose[glucose < 54.0]
        summaries.append(
            {
                "day": day_index + 1,
                "start_minute": day_index * 1440,
                "step_count": int(len(group)),
                "tir_70_180_pct": round(_tir_pct(glucose), 3),
                "worst_glucose_mgdl": round(float(glucose.min()), 3) if not glucose.empty else None,
                "mean_glucose_mgdl": round(float(glucose.mean()), 3) if not glucose.empty else None,
                "supervisor_interventions": int(interventions.sum()),
                "supervisor_intervention_rate_pct": round(float(interventions.mean() * 100.0), 3) if len(interventions) else 0.0,
                "input_validator_fail_soft_rows": int(fail_soft.sum()),
                "time_above_250_pct": round(float((glucose > 250.0).mean() * 100.0), 3) if not glucose.empty else 0.0,
                "critical_events_below_54": int(len(critical)),
            }
        )
    return summaries


def _critical_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "glucose_actual_mgdl" not in df.columns:
        return pd.DataFrame()
    work = df[pd.to_numeric(df["glucose_actual_mgdl"], errors="coerce") < 54.0].copy()
    if work.empty:
        return work
    work["day"] = (pd.to_numeric(work["time_minutes"], errors="coerce") // 1440).astype(int) + 1
    return work


def _interventions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mask = df.get("safety_triggered", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    return df[mask].copy()


def _worst_case_events(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    glucose = pd.to_numeric(work.get("glucose_actual_mgdl", pd.Series(dtype=float)), errors="coerce")
    work["danger_score"] = np.where(glucose < 70.0, 70.0 - glucose, np.maximum(glucose - 250.0, 0.0))
    work = work.sort_values(["danger_score", "time_minutes"], ascending=[False, True]).head(limit)
    fields = [
        "time_minutes",
        "glucose_actual_mgdl",
        "glucose_mechanistic_mgdl",
        "glucose_to_algo_mgdl",
        "predicted_glucose_30min",
        "algo_recommended_insulin_units",
        "delivered_insulin_units",
        "safety_triggered",
        "safety_reason",
        "sensor_status",
        "input_validator_fail_soft",
        "predictor_uncertainty_std_mgdl",
        "danger_score",
    ]
    return [
        {field: _safe_json(row.get(field)) for field in fields if field in row}
        for row in work.to_dict(orient="records")
    ]


def _total_summary(df: pd.DataFrame, safety_report: Dict[str, Any], config: EnduranceConfig) -> Dict[str, Any]:
    glucose = _glucose_series(df)
    interventions = _interventions(df)
    critical = _critical_events(df)
    physiology_quality = _physiology_quality(df)
    p = _tir_pct(glucose) / 100.0 if not glucose.empty else 0.0
    n = max(1, len(glucose))
    ci_half = 1.96 * math.sqrt((p * (1.0 - p)) / n) * 100.0
    critical_after_supervisor = 0
    if not critical.empty and "safety_triggered" in critical.columns:
        critical_after_supervisor = int(critical["safety_triggered"].fillna(False).astype(bool).sum())
    return {
        "status": "completed",
        "duration": config.duration,
        "duration_minutes": config.duration_minutes,
        "execution_mode": config.execution_mode,
        "wall_clock_target_seconds": config.wall_clock_target_seconds or None,
        "expected_steps": config.expected_steps,
        "actual_steps": int(len(df)),
        "total_tir_70_180_pct": round(p * 100.0, 3),
        "tir_95_ci_pct": [round(max(0.0, p * 100.0 - ci_half), 3), round(min(100.0, p * 100.0 + ci_half), 3)],
        "mean_glucose_mgdl": round(float(glucose.mean()), 3) if not glucose.empty else None,
        "worst_glucose_mgdl": round(float(glucose.min()), 3) if not glucose.empty else None,
        "max_glucose_mgdl": round(float(glucose.max()), 3) if not glucose.empty else None,
        "supervisor_interventions": int(len(interventions)),
        "supervisor_intervention_rate_pct": round(float(len(interventions) / max(1, len(df)) * 100.0), 3),
        "critical_events_below_54": int(len(critical)),
        "supervisor_failure_rate_pct": round(float(critical_after_supervisor / max(1, len(interventions)) * 100.0), 3),
        "sensor_artifact_steps": int((df.get("sensor_status", pd.Series(dtype=str)).astype(str) != "ok").sum()) if not df.empty else 0,
        "input_validator_fail_soft_rows": physiology_quality["input_validator_fail_soft_rows"],
        "algorithm_blind_hyperglycemia_rows": physiology_quality["algorithm_blind_hyperglycemia_rows"],
        "false_predicted_hypo_alert_rows": physiology_quality["false_predicted_hypo_alert_rows"],
        "mean_abs_truth_sensor_gap_mgdl": physiology_quality["mean_abs_truth_sensor_gap_mgdl"],
        "physiology_warning_count": physiology_quality["warning_count"],
        "physiology_warnings": physiology_quality["warnings"],
        "physiology_quality": physiology_quality,
        "checkpoint_interval_minutes": config.checkpoint_interval_minutes,
        "hardware_sample_interval_minutes": config.hardware_sample_interval_minutes,
        "performance_report": safety_report.get("performance_report", {}),
        "safety_report": safety_report,
        "completed_at_utc": utc_now_iso(),
    }


def _write_main_figure(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_hours = pd.to_numeric(df["time_minutes"], errors="coerce") / 60.0
    glucose = pd.to_numeric(df["glucose_actual_mgdl"], errors="coerce")
    algo_glucose = pd.to_numeric(df.get("glucose_to_algo_mgdl", pd.Series(dtype=float)), errors="coerce")
    mechanistic = pd.to_numeric(df.get("glucose_mechanistic_mgdl", pd.Series(dtype=float)), errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    ax.axhspan(70, 180, color="#DCFCE7", alpha=0.8, label="70-180 mg/dL")
    if not mechanistic.empty and not np.allclose(mechanistic.fillna(-1), glucose.fillna(-1)):
        ax.plot(x_hours, mechanistic, color="#DC2626", linewidth=1.0, alpha=0.8, label="Mechanistic truth")
    ax.plot(x_hours, glucose, color="#0F766E", linewidth=1.4, label="Truth glucose")
    if not algo_glucose.empty:
        ax.plot(x_hours, algo_glucose, color="#2563EB", linewidth=0.9, alpha=0.72, label="Algorithm input")
    interventions = _interventions(df)
    if not interventions.empty:
        ax.scatter(
            pd.to_numeric(interventions["time_minutes"], errors="coerce") / 60.0,
            pd.to_numeric(interventions["glucose_actual_mgdl"], errors="coerce"),
            color="#DC2626",
            s=12,
            label="Supervisor intervention",
        )
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("IINTS Jetson endurance glucose trace")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_pdf(markdown_path: Path, pdf_path: Path) -> None:
    try:
        from fpdf import FPDF
    except Exception:
        return
    text = markdown_path.read_text(encoding="utf-8")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.splitlines():
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        if not safe_line:
            pdf.ln(3)
            continue
        # Fixed-width cells avoid fpdf2 edge cases when the cursor is already
        # near the right margin after a previous wrapped line.
        usable_width = pdf.w - pdf.l_margin - pdf.r_margin
        for start in range(0, len(safe_line), 100):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_width, 6, safe_line[start : start + 100])
    pdf.output(str(pdf_path))


def _write_report(output_dir: Path, summary: Dict[str, Any], daily: List[Dict[str, Any]]) -> Path:
    final_dir = output_dir / "final"
    report_path = final_dir / "ENDURANCE_REPORT.md"
    physiology = summary.get("physiology_quality", {}) or {}
    truth_metrics = physiology.get("truth_metrics", {}) or {}
    algorithm_metrics = physiology.get("algorithm_input_metrics", {}) or {}
    warnings = list(physiology.get("warnings", []) or [])
    lines = [
        "# IINTS Jetson Endurance Report",
        "",
        f"- Duration: `{summary['duration']}` ({summary['duration_minutes']} minutes)",
        f"- Execution mode: `{summary['execution_mode']}`",
        f"- Steps: `{summary['actual_steps']}` / `{summary['expected_steps']}`",
        f"- Total TIR 70-180: `{summary['total_tir_70_180_pct']}%`",
        f"- TIR 95% CI: `{summary['tir_95_ci_pct'][0]}%` to `{summary['tir_95_ci_pct'][1]}%`",
        f"- Worst glucose: `{summary['worst_glucose_mgdl']} mg/dL`",
        f"- Max glucose: `{summary['max_glucose_mgdl']} mg/dL`",
        f"- Critical events <54 mg/dL: `{summary['critical_events_below_54']}`",
        f"- Supervisor interventions: `{summary['supervisor_interventions']}`",
        f"- Supervisor failure rate: `{summary['supervisor_failure_rate_pct']}%`",
        f"- Input-validator fail-soft rows: `{summary.get('input_validator_fail_soft_rows', 0)}`",
        f"- Algorithm-blind hyperglycemia rows: `{summary.get('algorithm_blind_hyperglycemia_rows', 0)}`",
        f"- Physiology warnings: `{summary.get('physiology_warning_count', 0)}`",
        f"- Checkpoint interval: `{summary['checkpoint_interval_minutes']} minutes`",
        f"- Hardware sample interval: `{summary['hardware_sample_interval_minutes']} minutes`",
        "",
        "## Physiology Quality",
        "",
        f"- Truth TIR 70-180: `{truth_metrics.get('tir_70_180_pct', 0.0)}%`",
        f"- Truth time >250: `{truth_metrics.get('time_above_250_pct', 0.0)}%`",
        f"- Truth CV: `{truth_metrics.get('cv_pct', 'n/a')}%`",
        f"- Algorithm-input TIR 70-180: `{algorithm_metrics.get('tir_70_180_pct', 0.0)}%`",
        f"- Mean absolute truth/sensor gap: `{summary.get('mean_abs_truth_sensor_gap_mgdl', 0.0)} mg/dL`",
        "",
    ]
    if warnings:
        lines.extend(["### Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
        lines.append("")
    else:
        lines.extend(["No physiology-quality warnings were raised for this run.", ""])
    lines.extend(["## Daily Summaries", ""])
    for day in daily:
        lines.append(
            f"- Day {day['day']}: TIR `{day['tir_70_180_pct']}%`, "
            f">250 `{day.get('time_above_250_pct', 0.0)}%`, "
            f"worst `{day['worst_glucose_mgdl']} mg/dL`, "
            f"interventions `{day['supervisor_interventions']}`, "
            f"fail-soft rows `{day.get('input_validator_fail_soft_rows', 0)}`, "
            f"critical events `{day['critical_events_below_54']}`"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_pdf(report_path, final_dir / "ENDURANCE_REPORT.pdf")
    return report_path


def _row_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _reference_teacher_insulin(row: pd.Series, time_step_minutes: int) -> float:
    """Conservative, auditable research label for local controller training.

    This is not a clinical dosing rule. It provides a safer synthetic teacher
    than copying weak demo-algorithm output, especially for long Jetson bundles.
    """
    glucose = _row_float(row, "glucose_actual_mgdl", 120.0)
    predicted = _row_float(row, "predicted_glucose_30min", glucose)
    trend = _row_float(row, "glucose_trend_mgdl_min", 0.0)
    iob = max(0.0, _row_float(row, "patient_iob_units", 0.0))
    carbs = max(0.0, _row_float(row, "carb_intake_grams", 0.0))
    isf = max(10.0, _row_float(row, "effective_isf", 50.0))
    icr = max(3.0, _row_float(row, "effective_icr", 12.0))
    basal_rate = min(3.0, max(0.0, _row_float(row, "effective_basal_rate_u_per_hr", 0.0)))

    basal_units = basal_rate * max(1, time_step_minutes) / 60.0
    if glucose <= 90.0 or predicted <= 75.0 or trend <= -1.2:
        basal_units *= 0.25
    if glucose < 70.0 or predicted < 70.0:
        return 0.0

    meal_units = 0.0
    if carbs > 0.0 and glucose > 90.0 and predicted > 75.0:
        meal_units = min(carbs / icr, 4.0)
        if trend < -0.8:
            meal_units *= 0.5
        if iob > 2.0:
            meal_units *= 0.5

    correction_units = 0.0
    reference_glucose = max(glucose, predicted)
    if reference_glucose >= 145.0 and iob < 2.5 and trend > -1.2:
        raw_correction = (reference_glucose - 110.0) / isf
        trend_factor = 0.5 if trend >= 0.0 else 0.25
        correction_units = max(0.0, min(raw_correction * trend_factor, 2.5))

    total = basal_units + meal_units + correction_units
    if iob > 3.0:
        total = min(total, 0.2)
    return round(float(min(max(total, 0.0), 3.5)), 4)


def _write_research_outputs(output_dir: Path, config: EnduranceConfig, df: pd.DataFrame) -> Dict[str, str]:
    research_dir = output_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    predictor_columns = [
        "subject_id",
        "segment",
        "time_minutes",
        "glucose_actual_mgdl",
        "glucose_to_algo_mgdl",
        "predicted_glucose_30min",
        "glucose_trend_mgdl_min",
        "carb_intake_grams",
        "carb_grams",
        "patient_iob_units",
        "patient_cob_grams",
        "effective_isf",
        "effective_icr",
        "effective_basal_rate_u_per_hr",
        "steps",
        "calories",
        "heart_rate",
        "sleep_minutes",
        "time_of_day_sin",
        "time_of_day_cos",
        "delivered_insulin_units",
        "input_validator_fail_soft",
        "sensor_status",
    ]
    training_df = df.copy()
    training_df["subject_id"] = f"jetson_{Path(config.output_dir).name}"
    training_df["segment"] = 0
    carb_source = (
        training_df["carb_intake_grams"]
        if "carb_intake_grams" in training_df.columns
        else pd.Series(0.0, index=training_df.index)
    )
    training_df["carb_grams"] = pd.to_numeric(carb_source, errors="coerce").fillna(0.0)
    for column in ("steps", "calories", "heart_rate", "sleep_minutes"):
        training_df[column] = 0.0
    day_fraction = pd.to_numeric(training_df["time_minutes"], errors="coerce").fillna(0.0) % 1440 / 1440.0
    training_df["time_of_day_sin"] = np.sin(day_fraction * 2.0 * math.pi)
    training_df["time_of_day_cos"] = np.cos(day_fraction * 2.0 * math.pi)
    for column in predictor_columns:
        if column not in training_df.columns:
            training_df[column] = 0.0
    training_path = research_dir / "predictor_training.csv"
    training_df[predictor_columns].to_csv(training_path, index=False)
    controller_columns = [
        "glucose_actual_mgdl",
        "glucose_to_algo_mgdl",
        "predicted_glucose_30min",
        "glucose_trend_mgdl_min",
        "patient_iob_units",
        "patient_cob_grams",
        "effective_isf",
        "effective_icr",
        "effective_basal_rate_u_per_hr",
        "carb_intake_grams",
        "observed_delivered_insulin_units",
        "reference_teacher_insulin_units",
        "teacher_insulin_units",
        "algo_recommended_insulin_units",
        "safety_triggered",
        "input_validator_fail_soft",
        "time_minutes",
    ]
    controller_path = research_dir / "controller_teacher_dataset.csv"
    controller_df = df.copy()
    for column in [
        "glucose_actual_mgdl",
        "glucose_to_algo_mgdl",
        "predicted_glucose_30min",
        "glucose_trend_mgdl_min",
        "patient_iob_units",
        "patient_cob_grams",
        "effective_isf",
        "effective_icr",
        "effective_basal_rate_u_per_hr",
        "carb_intake_grams",
        "delivered_insulin_units",
        "algo_recommended_insulin_units",
        "safety_triggered",
        "input_validator_fail_soft",
        "time_minutes",
    ]:
        if column not in controller_df.columns:
            controller_df[column] = 0.0
    controller_df["observed_delivered_insulin_units"] = pd.to_numeric(
        controller_df["delivered_insulin_units"], errors="coerce"
    ).fillna(0.0)
    controller_df["reference_teacher_insulin_units"] = controller_df.apply(
        lambda row: _reference_teacher_insulin(row, config.time_step_minutes), axis=1
    )
    controller_df["teacher_insulin_units"] = controller_df["reference_teacher_insulin_units"]
    controller_df[controller_columns].to_csv(controller_path, index=False)

    manifest_path = research_dir / "training_manifest.json"
    training_manifest = {
        "created_at_utc": utc_now_iso(),
        "execution_mode": config.execution_mode,
        "duration": config.duration,
        "duration_minutes": config.duration_minutes,
        "row_count": int(len(training_df)),
        "subject_id": f"jetson_{Path(config.output_dir).name}",
        "dataset_path": str(training_path),
        "controller_dataset_path": str(controller_path),
        "columns": predictor_columns,
        "recommended_predictor_configs": [
            "research/configs/predictor.yaml",
            "research/configs/predictor_multimodal_dual_guard.yaml",
            "research/configs/predictor_multimodal_dual_guard_preannounce.yaml",
        ],
        "example_training_command": (
            "PYTHONPATH=src python3 research/train_predictor.py "
            f"--data {training_path} "
            "--config research/configs/predictor.yaml "
            f"--out models/{Path(config.output_dir).name}_predictor"
        ),
        "example_controller_training_command": (
            "iints research train-controller "
            f"--data {controller_path} "
            f"--output models/{Path(config.output_dir).name}_controller.json"
        ),
        "controller_teacher_policy": "conservative_reference_v1",
        "controller_teacher_note": (
            "teacher_insulin_units is generated by a conservative, auditable research-only reference policy; "
            "observed_delivered_insulin_units preserves the controller output that actually ran during acquisition."
        ),
        "ministral_training_supported": False,
        "ministral_note": (
            "Ministral/Ollama is currently the local explanation backend, not the trainable glucose predictor. "
            "Use this bundle to train or fine-tune the predictor; treat Ministral outputs as review artifacts."
        ),
    }
    _write_json(manifest_path, training_manifest)

    readme_path = research_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Jetson Research Bundle",
                "",
                "This folder turns the endurance run into a reusable research artifact.",
                "",
                "## Files",
                "",
                "- `predictor_training.csv`: rows compatible with the SDK predictor-training pipeline.",
                "- `controller_teacher_dataset.csv`: supervised research labels with both observed delivery and a conservative reference teacher.",
                "- `training_manifest.json`: lineage, columns, and a reproducible training command.",
                "",
                "## Important distinction",
                "",
                "- The trainable model in the SDK research pipeline is the **glucose predictor**.",
                "- `Ministral` / Ollama is the **local explanation assistant**, not an online-trained controller.",
                "- `teacher_insulin_units` uses `conservative_reference_v1`; it is a research label, not a medical dosing rule.",
                "- Keep acquisition and training separate so the same run is not both training data and unbiased evaluation evidence.",
                "- A single Jetson run is one synthetic subject; combine multiple runs or external data before claiming generalization.",
                "",
                "## Example next step",
                "",
                "```bash",
                str(training_manifest["example_training_command"]),
                "```",
                "",
                "Controller imitation baseline:",
                "",
                "```bash",
                str(training_manifest["example_controller_training_command"]),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "research_training_csv": str(training_path),
        "research_controller_teacher_csv": str(controller_path),
        "research_training_manifest_json": str(manifest_path),
        "research_readme_md": str(readme_path),
    }


def _write_outputs(
    *,
    output_dir: Path,
    config: EnduranceConfig,
    df: pd.DataFrame,
    safety_report: Dict[str, Any],
) -> Dict[str, str]:
    raw_dir = output_dir / "raw"
    daily_dir = output_dir / "daily"
    final_dir = output_dir / "final"
    raw_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    steps_path = raw_dir / "steps.csv"
    interventions_path = raw_dir / "interventions.csv"
    critical_path = raw_dir / "critical_events.csv"
    hourly_path = final_dir / "tir_timeseries.csv"
    summary_path = final_dir / "test_summary.json"
    supervisor_path = final_dir / "supervisor_analysis.json"
    worst_path = final_dir / "worst_case_events.json"
    figure_path = final_dir / "main_figure.png"
    hardware_path = raw_dir / "hardware_metrics.csv"

    df.to_csv(steps_path, index=False)
    _interventions(df).to_csv(interventions_path, index=False)
    _critical_events(df).to_csv(critical_path, index=False)
    _hourly_summary(df, config.time_step_minutes).to_csv(hourly_path, index=False)

    daily = _daily_summary(df)
    for day in daily:
        _write_json(daily_dir / f"day_{day['day']:02d}_summary.json", day)

    summary = _total_summary(df, safety_report, config)
    _write_json(summary_path, summary)
    _write_json(
        supervisor_path,
        {
            "intervention_count": summary["supervisor_interventions"],
            "intervention_rate_pct": summary["supervisor_intervention_rate_pct"],
            "failure_rate_pct": summary["supervisor_failure_rate_pct"],
            "safety_report": safety_report,
            "top_reasons": _interventions(df).get("safety_reason", pd.Series(dtype=str)).value_counts().to_dict(),
        },
    )
    _write_json(worst_path, {"events": _worst_case_events(df)})
    _write_main_figure(df, figure_path)
    report_path = _write_report(output_dir, summary, daily)
    outputs = {
        "steps_csv": str(steps_path),
        "interventions_csv": str(interventions_path),
        "critical_events_csv": str(critical_path),
        "test_summary_json": str(summary_path),
        "supervisor_analysis_json": str(supervisor_path),
        "worst_case_events_json": str(worst_path),
        "tir_timeseries_csv": str(hourly_path),
        "endurance_report_md": str(report_path),
        "endurance_report_pdf": str(final_dir / "ENDURANCE_REPORT.pdf"),
        "main_figure_png": str(figure_path),
        "hardware_metrics_csv": str(hardware_path),
    }
    if config.research_export:
        outputs.update(_write_research_outputs(output_dir, config, df))
    return outputs


def _latest_snapshot(snapshot_dir: Path) -> Optional[Path]:
    snapshots = sorted([*snapshot_dir.glob("snapshot_*m.json"), *snapshot_dir.glob("snapshot_*h.json")])
    return snapshots[-1] if snapshots else None


def run_endurance_study(
    *,
    algorithm: InsulinAlgorithm,
    predictor: Optional[object],
    config: EnduranceConfig,
    progress_callback: Optional[Any] = None,
    monotonic_fn: Any = time.monotonic,
    sleep_fn: Any = time.sleep,
) -> Dict[str, Any]:
    if config.profile not in ENDURANCE_PROFILES:
        raise JetsonEnduranceError(f"Unknown endurance profile '{config.profile}'.")
    if config.execution_mode not in ENDURANCE_EXECUTION_MODES:
        valid = ", ".join(sorted(ENDURANCE_EXECUTION_MODES))
        raise JetsonEnduranceError(f"Unknown execution mode '{config.execution_mode}'. Choose one of: {valid}.")

    output_dir = Path(config.output_dir)
    protocol_dir = output_dir / "protocol"
    raw_dir = output_dir / "raw"
    snapshot_dir = output_dir / "snapshots"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    existing_status_path = output_dir / "status.json"
    existing_status = _read_json(existing_status_path) if config.resume and existing_status_path.is_file() else {}
    started_at = str(existing_status.get("started_at_utc") or utc_now_iso())
    previous_wall_elapsed_seconds = float(existing_status.get("wall_elapsed_seconds") or 0.0)
    stress_events = _profile_events(config.profile, config.duration_minutes, config.custom_profile_path)
    test_config = asdict(config)
    test_config["stress_event_count"] = len(stress_events)
    test_config["stress_events"] = stress_events
    (protocol_dir / "test_config.yaml").write_text(yaml.safe_dump(_safe_json(test_config), sort_keys=False), encoding="utf-8")
    hardware = collect_jetson_hardware_info()
    _write_json(protocol_dir / "hardware_info.json", hardware)

    simulator = _build_simulator(algorithm=algorithm, predictor=predictor, config=config, stress_events=stress_events)
    records: List[Dict[str, Any]] = []
    hardware_records: List[Dict[str, Any]] = []
    resume_count = int(existing_status.get("resume_count") or 0)
    latest = _latest_snapshot(snapshot_dir) if config.resume else None
    if latest is not None:
        snapshot = _read_json(latest)
        simulator.load_state(snapshot["simulator_state"])
        steps_path = raw_dir / "steps.csv"
        if not steps_path.is_file():
            raise JetsonEnduranceError(
                f"Cannot resume from {latest.name}: raw/steps.csv is missing."
            )
        csv_records = pd.read_csv(steps_path).to_dict(orient="records")
        completed_steps = int(snapshot.get("completed_steps") or len(csv_records))
        records = [{str(key): value for key, value in row.items()} for row in csv_records[:completed_steps]]
        resume_count += 1

    hardware_path = raw_dir / "hardware_metrics.csv"
    if config.resume and hardware_path.is_file():
        hardware_records = [
            {str(key): value for key, value in row.items()}
            for row in pd.read_csv(hardware_path).to_dict(orient="records")
        ]
    latest_hardware = collect_jetson_hardware_info()
    hardware_records.append(
        _hardware_metric_row(
            latest_hardware,
            elapsed_minutes=len(records) * config.time_step_minutes,
        )
    )
    pd.DataFrame(hardware_records).to_csv(hardware_path, index=False)
    wall_start = monotonic_fn()
    last_checkpoint_minute = int(existing_status.get("last_checkpoint_minute") or 0)

    _write_json(
        output_dir / "status.json",
        _status_payload(
            config=config,
            status="running",
            started_at_utc=started_at,
            completed_steps=len(records),
            message="Endurance run started.",
            wall_elapsed_seconds=0.0,
            hardware_latest=latest_hardware,
            last_checkpoint_minute=last_checkpoint_minute,
            resume_count=resume_count,
        ),
    )

    stopped_early = False
    stop_path = output_dir / "STOP_REQUESTED"
    if not config.resume and stop_path.exists():
        stop_path.unlink()
    for record in simulator.run_live(config.simulator_end_minutes):
        if stop_path.is_file():
            stopped_early = True
            break
        records.append(record)
        completed_steps = len(records)
        if completed_steps % max(1, config.status_interval_steps) == 0 or completed_steps == config.expected_steps:
            partial_df = pd.DataFrame(records)
            partial_df.to_csv(raw_dir / "steps.csv", index=False)
            elapsed_minutes = completed_steps * config.time_step_minutes
            if (
                config.hardware_sample_interval_minutes > 0
                and elapsed_minutes % config.hardware_sample_interval_minutes == 0
            ):
                latest_hardware = collect_jetson_hardware_info()
                hardware_records.append(
                    _hardware_metric_row(
                        latest_hardware,
                        elapsed_minutes=elapsed_minutes,
                    )
                )
                pd.DataFrame(hardware_records).to_csv(hardware_path, index=False)
            status = _status_payload(
                config=config,
                status="running",
                started_at_utc=started_at,
                completed_steps=completed_steps,
                current_record=record,
                wall_elapsed_seconds=previous_wall_elapsed_seconds + (monotonic_fn() - wall_start),
                hardware_latest=latest_hardware,
                last_checkpoint_minute=last_checkpoint_minute,
                resume_count=resume_count,
            )
            glucose = _glucose_series(partial_df)
            physiology = _physiology_quality(partial_df)
            status["tir_so_far_pct"] = round(_tir_pct(glucose), 3)
            status["interventions"] = int(len(_interventions(partial_df)))
            status["critical_events"] = int(len(_critical_events(partial_df)))
            status["worst_glucose_mgdl"] = round(float(glucose.min()), 3) if not glucose.empty else None
            status["physiology_warning_count"] = physiology["warning_count"]
            status["input_validator_fail_soft_rows"] = physiology["input_validator_fail_soft_rows"]
            status["algorithm_blind_hyperglycemia_rows"] = physiology["algorithm_blind_hyperglycemia_rows"]
            status["mean_abs_truth_sensor_gap_mgdl"] = physiology["mean_abs_truth_sensor_gap_mgdl"]
            _write_json(output_dir / "status.json", status)
            if progress_callback is not None:
                progress_callback(status)

        elapsed_minutes = completed_steps * config.time_step_minutes
        if (
            config.checkpoint_interval_minutes > 0
            and elapsed_minutes > 0
            and elapsed_minutes % config.checkpoint_interval_minutes == 0
            and elapsed_minutes != last_checkpoint_minute
        ):
            pd.DataFrame(records).to_csv(raw_dir / "steps.csv", index=False)
            state = simulator.save_state()
            state["current_time"] = int(float(record["time_minutes"]) + config.time_step_minutes)
            _write_json(
                snapshot_dir / f"snapshot_{elapsed_minutes:06d}m.json",
                {
                    "captured_at_utc": utc_now_iso(),
                    "completed_steps": completed_steps,
                    "simulator_state": state,
                    "steps_sha256": compute_sha256(raw_dir / "steps.csv") if (raw_dir / "steps.csv").is_file() else None,
                },
            )
            last_checkpoint_minute = elapsed_minutes

        if config.execution_mode == "wall_clock":
            target_elapsed_seconds = completed_steps * config.time_step_minutes * 60
            current_elapsed_seconds = previous_wall_elapsed_seconds + (monotonic_fn() - wall_start)
            remaining_sleep_seconds = max(0.0, target_elapsed_seconds - current_elapsed_seconds)
            if remaining_sleep_seconds > 0:
                sleep_fn(remaining_sleep_seconds)

    df = pd.DataFrame(records)
    safety_report = simulator.supervisor.get_safety_report()
    if stopped_early:
        safety_report["stopped_early"] = True
        safety_report["stop_requested_at_utc"] = stop_path.read_text(encoding="utf-8").strip()
    if simulator.enable_profiling:
        safety_report["performance_report"] = simulator._build_performance_report()
    outputs = _write_outputs(output_dir=output_dir, config=config, df=df, safety_report=safety_report)
    final_status = _status_payload(
        config=config,
        status="stopped" if stopped_early else "completed",
        started_at_utc=started_at,
        completed_steps=len(records),
        current_record=records[-1] if records else None,
        message="Endurance run stopped early." if stopped_early else "Endurance run completed.",
        wall_elapsed_seconds=previous_wall_elapsed_seconds + (monotonic_fn() - wall_start),
        hardware_latest=latest_hardware,
        last_checkpoint_minute=last_checkpoint_minute,
        resume_count=resume_count,
    )
    summary = _read_json(Path(outputs["test_summary_json"]))
    final_status.update(
        {
            "tir_so_far_pct": summary["total_tir_70_180_pct"],
            "interventions": summary["supervisor_interventions"],
            "critical_events": summary["critical_events_below_54"],
            "worst_glucose_mgdl": summary["worst_glucose_mgdl"],
            "physiology_warning_count": summary.get("physiology_warning_count"),
            "input_validator_fail_soft_rows": summary.get("input_validator_fail_soft_rows"),
            "algorithm_blind_hyperglycemia_rows": summary.get("algorithm_blind_hyperglycemia_rows"),
            "false_predicted_hypo_alert_rows": summary.get("false_predicted_hypo_alert_rows"),
            "mean_abs_truth_sensor_gap_mgdl": summary.get("mean_abs_truth_sensor_gap_mgdl"),
            "outputs": outputs,
        }
    )
    _write_json(output_dir / "status.json", final_status)
    return {"status": final_status, "outputs": outputs, "summary": summary}


def load_endurance_status(output_dir: str | Path) -> Dict[str, Any]:
    status_path = Path(output_dir) / "status.json"
    if not status_path.is_file():
        raise JetsonEnduranceError(f"No status.json found in {output_dir}.")
    return _read_json(status_path)


def stop_endurance_study(output_dir: str | Path, *, generate_report: bool = False) -> Dict[str, Any]:
    output_path = Path(output_dir)
    stop_path = output_path / "STOP_REQUESTED"
    stop_path.write_text(utc_now_iso(), encoding="utf-8")
    status = load_endurance_status(output_path) if (output_path / "status.json").is_file() else {}
    status["stop_requested_at_utc"] = utc_now_iso()
    status["generate_report_requested"] = bool(generate_report)
    _write_json(output_path / "status.json", status)
    return status


def export_endurance_archive(output_dir: str | Path, output: str | Path) -> Path:
    source = Path(output_dir)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))
    return target


def build_endurance_service_file(
    *,
    algo: str,
    duration: str,
    output_dir: str,
    predictor: Optional[str] = None,
    profile: str = "mixed_adversarial",
    seed: Optional[int] = None,
    wall_clock: bool = False,
    working_directory: Optional[str] = None,
) -> str:
    command = [
        "iints",
        "jetson",
        "endurance",
        "start",
        "--algo",
        algo,
        "--duration",
        duration,
        "--output-dir",
        output_dir,
        "--profile",
        profile,
        "--resume",
    ]
    if predictor:
        command.extend(["--predictor", predictor])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if wall_clock:
        command.append("--wall-clock")
    workdir = working_directory or str(Path.cwd())
    return "\n".join(
        [
            "[Unit]",
            "Description=IINTS Jetson Endurance Study",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={workdir}",
            f"ExecStart={' '.join(command)}",
            "Restart=on-failure",
            "RestartSec=20",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )
