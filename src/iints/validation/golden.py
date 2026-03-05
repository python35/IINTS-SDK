from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class GoldenScenarioSpec:
    preset: str
    description: str
    expected: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class GoldenBenchmarkPack:
    name: str
    version: int
    scenarios: List[GoldenScenarioSpec]


def load_golden_benchmark_pack(path: Optional[Path] = None) -> GoldenBenchmarkPack:
    if path is None:
        import sys

        if sys.version_info >= (3, 9):
            from importlib.resources import files

            content = files("iints.presets").joinpath("golden_benchmark.yaml").read_text()
        else:
            from importlib import resources

            content = resources.read_text("iints.presets", "golden_benchmark.yaml")
    else:
        content = path.read_text()

    payload = yaml.safe_load(content) or {}
    if not isinstance(payload, dict):
        raise ValueError("Golden benchmark pack must be a YAML mapping.")

    name = str(payload.get("name", "golden-benchmark"))
    version = int(payload.get("version", 1))
    raw_scenarios = payload.get("scenarios", [])
    if not isinstance(raw_scenarios, list):
        raise ValueError("Golden benchmark pack 'scenarios' must be a list.")

    scenarios: List[GoldenScenarioSpec] = []
    for idx, item in enumerate(raw_scenarios):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario #{idx} must be a mapping.")
        preset = str(item.get("preset", "")).strip()
        if not preset:
            raise ValueError(f"Scenario #{idx} missing required 'preset'.")
        expected = item.get("expected", {})
        if not isinstance(expected, dict):
            raise ValueError(f"Scenario #{idx} expected must be a mapping.")
        scenarios.append(
            GoldenScenarioSpec(
                preset=preset,
                description=str(item.get("description", "")).strip(),
                expected={str(key): value for key, value in expected.items() if isinstance(value, dict)},
            )
        )

    return GoldenBenchmarkPack(name=name, version=version, scenarios=scenarios)


def evaluate_expected_ranges(
    metrics: Dict[str, float],
    expected: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for metric, limits in expected.items():
        value = metrics.get(metric)
        if value is None:
            checks[metric] = {"passed": False, "reason": "missing_metric", "value": None}
            continue
        min_value = limits.get("min")
        max_value = limits.get("max")
        passed = True
        reason = "ok"
        if min_value is not None and float(value) < float(min_value):
            passed = False
            reason = f"value {value:.3f} < min {float(min_value):.3f}"
        if passed and max_value is not None and float(value) > float(max_value):
            passed = False
            reason = f"value {value:.3f} > max {float(max_value):.3f}"
        checks[metric] = {
            "passed": passed,
            "value": float(value),
            "min": float(min_value) if min_value is not None else None,
            "max": float(max_value) if max_value is not None else None,
            "reason": reason,
        }
    return checks

