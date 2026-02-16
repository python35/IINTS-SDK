from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random


@dataclass
class ScenarioGeneratorConfig:
    name: str
    version: str = "1.0"
    description: str = "Generated scenario"
    duration_minutes: int = 1440
    seed: Optional[int] = None

    meal_count: int = 3
    meal_min_grams: float = 30.0
    meal_max_grams: float = 80.0
    meal_delay_min: int = 10
    meal_delay_max: int = 60
    meal_duration_min: int = 30
    meal_duration_max: int = 120

    exercise_count: int = 0
    exercise_intensity_min: float = 0.2
    exercise_intensity_max: float = 0.8
    exercise_duration_min: int = 30
    exercise_duration_max: int = 90

    sensor_error_count: int = 0
    sensor_error_min: float = 40.0
    sensor_error_max: float = 300.0


def _unique_times(rng: random.Random, max_time: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if max_time <= count:
        return sorted(rng.sample(range(max_time + 1), k=max_time + 1))[:count]
    return sorted(rng.sample(range(max_time), k=count))


def generate_random_scenario(config: ScenarioGeneratorConfig) -> Dict[str, Any]:
    rng = random.Random(config.seed)
    events: List[Dict[str, Any]] = []
    max_time = max(config.duration_minutes - 1, 1)

    meal_times = _unique_times(rng, max_time, config.meal_count)
    for t in meal_times:
        events.append(
            {
                "start_time": int(t),
                "event_type": "meal",
                "value": float(rng.uniform(config.meal_min_grams, config.meal_max_grams)),
                "absorption_delay_minutes": int(rng.randint(config.meal_delay_min, config.meal_delay_max)),
                "duration": int(rng.randint(config.meal_duration_min, config.meal_duration_max)),
            }
        )

    exercise_times = _unique_times(rng, max_time, config.exercise_count)
    for t in exercise_times:
        events.append(
            {
                "start_time": int(t),
                "event_type": "exercise",
                "value": float(rng.uniform(config.exercise_intensity_min, config.exercise_intensity_max)),
                "duration": int(rng.randint(config.exercise_duration_min, config.exercise_duration_max)),
            }
        )

    sensor_times = _unique_times(rng, max_time, config.sensor_error_count)
    for t in sensor_times:
        events.append(
            {
                "start_time": int(t),
                "event_type": "sensor_error",
                "value": float(rng.uniform(config.sensor_error_min, config.sensor_error_max)),
            }
        )

    events = sorted(events, key=lambda e: e["start_time"])

    return {
        "scenario_name": config.name,
        "scenario_version": config.version,
        "description": config.description,
        "stress_events": events,
    }
