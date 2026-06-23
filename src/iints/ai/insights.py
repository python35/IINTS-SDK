from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import math


AI_INSIGHT_CONTEXT_VERSION = "iints-ai-insight-context-v1"

GLUCOSE_KEYS = {
    "glucose",
    "glucose_mgdl",
    "glucose_mg_dl",
    "glucose_actual_mgdl",
    "glucose_to_algo_mgdl",
    "sensor_glucose_mgdl",
    "current_glucose",
    "predicted_glucose",
    "predicted_glucose_30min",
    "cgm",
}
TIME_KEYS = {"time", "time_min", "time_minutes", "timestamp", "minute", "minutes"}
INSULIN_KEYS = {
    "insulin",
    "insulin_units",
    "delivered_insulin",
    "delivered_insulin_units",
    "insulin_delivered",
    "active_insulin",
    "insulin_on_board",
    "iob",
}
CARB_KEYS = {"carbs", "carb_intake_grams", "carbs_on_board", "cob", "meal_carbs"}
SAFETY_KEYS = {
    "safety_triggered",
    "safety_override",
    "supervisor_intervention",
    "supervisor_blocked",
    "blocked_by_supervisor",
}
OOD_KEYS = {"predictor_ood_status", "out_of_distribution", "in_distribution"}

MAX_RECORDS = 5000


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _lookup(record: dict[str, Any], candidates: set[str]) -> Any:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _iter_records(payload: Any, *, depth: int = 0) -> Iterable[dict[str, Any]]:
    if depth > 6:
        return
    if isinstance(payload, dict):
        normalized_keys = {str(key).lower() for key in payload}
        known = GLUCOSE_KEYS | TIME_KEYS | INSULIN_KEYS | CARB_KEYS | SAFETY_KEYS | OOD_KEYS
        if normalized_keys.intersection(known):
            yield payload
        for value in payload.values():
            yield from _iter_records(value, depth=depth + 1)
    elif isinstance(payload, list):
        for item in payload[:MAX_RECORDS]:
            yield from _iter_records(item, depth=depth + 1)


def _extract_series(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {"glucose": [], "time": [], "insulin": [], "carbs": []}
    for record in records:
        glucose = _to_float(_lookup(record, GLUCOSE_KEYS))
        time = _to_float(_lookup(record, TIME_KEYS))
        insulin = _to_float(_lookup(record, INSULIN_KEYS))
        carbs = _to_float(_lookup(record, CARB_KEYS))
        if glucose is not None:
            series["glucose"].append(glucose)
            if time is not None:
                series["time"].append(time)
        if insulin is not None:
            series["insulin"].append(insulin)
        if carbs is not None:
            series["carbs"].append(carbs)
    return series


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "first": _round(values[0]),
        "last": _round(values[-1]),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "mean": _round(mean),
        "std": _round(math.sqrt(variance)),
    }


def _max_rate(glucose: list[float], time_values: list[float]) -> float | None:
    if len(glucose) < 2 or len(time_values) != len(glucose):
        return None
    rates: list[float] = []
    for previous_g, current_g, previous_t, current_t in zip(
        glucose,
        glucose[1:],
        time_values,
        time_values[1:],
        strict=False,
    ):
        dt = current_t - previous_t
        if dt > 0:
            rates.append((current_g - previous_g) / dt)
    if not rates:
        return None
    return max(rates, key=abs)


def _time_in_range(glucose: list[float], low: float, high: float) -> float | None:
    if not glucose:
        return None
    return sum(1 for value in glucose if low <= value <= high) / len(glucose) * 100.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "blocked", "triggered"}
    return False


def _count_safety(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        for key in SAFETY_KEYS:
            if _truthy(_lookup(record, {key})):
                count += 1
                break
    return count


def _count_ood(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        raw = _lookup(record, OOD_KEYS)
        if raw is None:
            continue
        if isinstance(raw, str) and raw.strip().lower() in {"ood", "out_of_distribution", "false"}:
            count += 1
        elif isinstance(raw, bool) and raw is False:
            count += 1
    return count


def _extract_provided_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {}
    allowed = {
        "steps",
        "duration_minutes",
        "mean_glucose_mgdl",
        "min_glucose_mgdl",
        "max_glucose_mgdl",
        "max_glucose_delta_per_step_mgdl",
        "time_in_range_70_180_pct",
        "time_below_70_pct",
        "time_below_54_pct",
        "time_above_180_pct",
        "time_above_250_pct",
        "delivered_insulin_total_units",
        "recommended_insulin_total_units",
        "safety_trigger_count",
        "audit_override_count",
    }
    return {key: summary[key] for key in sorted(allowed) if key in summary}


def build_insight_context(task: str, payload: Any) -> dict[str, Any]:
    """Build deterministic evidence for the local LLM to explain.

    The returned context is intentionally conservative: it summarizes supplied
    traces and precomputed metrics, but it does not diagnose, dose, or replace
    the simulator/safety layer.
    """

    records = list(_iter_records(payload))[:MAX_RECORDS]
    series = _extract_series(records)
    glucose = series["glucose"]
    rate = _max_rate(glucose, series["time"])
    flags: list[str] = []
    if not glucose:
        flags.append("missing_glucose_trace")
    else:
        if min(glucose) < 54.0:
            flags.append("severe_low_glucose_below_54")
        elif min(glucose) < 70.0:
            flags.append("low_glucose_below_70")
        if max(glucose) > 250.0:
            flags.append("very_high_glucose_above_250")
        elif max(glucose) > 180.0:
            flags.append("high_glucose_above_180")
        stats = _summary_stats(glucose)
        if stats.get("std") is not None and float(stats["std"]) < 1.0 and len(glucose) >= 12:
            flags.append("near_flat_glucose_trace")
        if rate is not None and abs(rate) > 15.0:
            flags.append("red_team_impossible_glucose_rate")
        elif rate is not None and abs(rate) > 5.0:
            flags.append("suspicious_fast_glucose_rate")

    safety_count = _count_safety(records)
    ood_count = _count_ood(records)
    if safety_count:
        flags.append("safety_supervisor_activity_present")
    if ood_count:
        flags.append("predictor_out_of_distribution_present")

    if not flags:
        flags.append("no_major_deterministic_flags")

    return {
        "context_version": AI_INSIGHT_CONTEXT_VERSION,
        "task": task,
        "authority": {
            "ai_numeric_authority": False,
            "ai_role": "explain deterministic SDK evidence only",
            "medical_use": "research_only_not_for_treatment",
        },
        "input_coverage": {
            "records_scanned": len(records),
            "glucose_points": len(glucose),
            "insulin_points": len(series["insulin"]),
            "carb_points": len(series["carbs"]),
            "safety_event_like_records": safety_count,
            "predictor_ood_like_records": ood_count,
        },
        "provided_summary": _extract_provided_summary(payload),
        "computed_trace_summary": {
            "glucose_mgdl": _summary_stats(glucose),
            "insulin_units": _summary_stats(series["insulin"]),
            "carbs_grams": _summary_stats(series["carbs"]),
            "time_in_range_70_180_pct": _round(_time_in_range(glucose, 70.0, 180.0)),
            "time_below_70_pct": _round(
                None if not glucose else sum(1 for value in glucose if value < 70.0) / len(glucose) * 100.0
            ),
            "time_above_180_pct": _round(
                None if not glucose else sum(1 for value in glucose if value > 180.0) / len(glucose) * 100.0
            ),
            "max_observed_rate_mgdl_per_min": _round(rate),
        },
        "deterministic_flags": flags,
        "response_guidance": [
            "Use provided/computed metrics exactly; do not invent new values.",
            "Separate evidence from hypotheses.",
            "Mention missing context instead of filling gaps.",
            "Frame all conclusions as research-simulation insights, not care advice.",
        ],
    }
