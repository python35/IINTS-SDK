from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Python 3.9+
    from importlib.resources import files
except Exception:  # pragma: no cover
    files = None  # type: ignore[assignment]
    from importlib import resources
else:
    from importlib import resources


@dataclass(frozen=True)
class EmpiricalResidualProfile:
    id: str
    label: str
    source_dataset_ids: list[str]
    sample_interval_minutes: int
    templates: list[list[float]]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmpiricalResidualProfile":
        return cls(
            id=str(payload["id"]),
            label=str(payload["label"]),
            source_dataset_ids=[str(value) for value in payload.get("source_dataset_ids", [])],
            sample_interval_minutes=int(payload.get("sample_interval_minutes", 5)),
            templates=[
                [float(value) for value in template]
                for template in payload.get("templates", [])
            ],
        )


def _read_profile_registry_text() -> str:
    if files is not None:
        return files("iints.data").joinpath("physiology_residual_profiles.json").read_text()  # type: ignore[call-arg]
    return resources.read_text("iints.data", "physiology_residual_profiles.json")


def load_empirical_residual_profiles() -> list[EmpiricalResidualProfile]:
    payload = json.loads(_read_profile_registry_text())
    return [EmpiricalResidualProfile.from_dict(entry) for entry in payload]


def get_empirical_residual_profile(profile_id: str) -> EmpiricalResidualProfile:
    normalized = profile_id.strip()
    for profile in load_empirical_residual_profiles():
        if profile.id == normalized:
            return profile
    available = ", ".join(profile.id for profile in load_empirical_residual_profiles())
    raise KeyError(f"Unknown empirical residual profile '{profile_id}'. Available profiles: {available}")


class EmpiricalResidualModel:
    """Small additive model-discrepancy layer sampled from real CGM day residuals."""

    def __init__(
        self,
        profile: EmpiricalResidualProfile,
        *,
        seed: int | None = None,
        scale: float = 1.0,
        max_residual_rate_mgdl_per_min: float | None = 0.75,
    ) -> None:
        if not profile.templates:
            raise ValueError("Empirical residual profile must contain at least one template.")
        self.profile = profile
        self.seed = int(seed or 0)
        self.scale = float(scale)
        self.max_residual_rate_mgdl_per_min = max_residual_rate_mgdl_per_min
        self._templates = self._prepare_templates(profile.templates)
        self.reset()

    @classmethod
    def from_profile_id(
        cls,
        profile_id: str,
        *,
        seed: int | None = None,
        scale: float = 1.0,
        max_residual_rate_mgdl_per_min: float | None = 0.75,
    ) -> "EmpiricalResidualModel":
        return cls(
            get_empirical_residual_profile(profile_id),
            seed=seed,
            scale=scale,
            max_residual_rate_mgdl_per_min=max_residual_rate_mgdl_per_min,
        )

    def _prepare_templates(self, templates: list[list[float]]) -> list[list[float]]:
        if self.max_residual_rate_mgdl_per_min is None:
            return [[float(value) for value in template] for template in templates]
        max_rate = max(float(self.max_residual_rate_mgdl_per_min), 0.0)
        sample_interval = max(float(self.profile.sample_interval_minutes), 1.0)
        scale = max(abs(float(self.scale)), 1e-9)
        max_step = max_rate * sample_interval / scale

        limited_templates: list[list[float]] = []
        for template in templates:
            if not template:
                limited_templates.append([])
                continue
            limited = [float(template[0])]
            for raw_value in template[1:]:
                previous = limited[-1]
                requested_delta = float(raw_value) - previous
                bounded_delta = max(-max_step, min(requested_delta, max_step))
                limited.append(previous + bounded_delta)
            limited_templates.append(limited)
        return limited_templates

    def reset(self) -> None:
        self._template_offset = self.seed % len(self.profile.templates)

    def _template_for_day(self, day_index: int) -> list[float]:
        return self._templates[(self._template_offset + day_index) % len(self._templates)]

    def offset_at(self, current_time_minutes: float) -> float:
        sample_interval = max(float(self.profile.sample_interval_minutes), 1.0)
        day_index = max(int(current_time_minutes // 1440.0), 0)
        minute_of_day = float(current_time_minutes % 1440.0)
        template = self._template_for_day(day_index)
        if len(template) == 1:
            return float(template[0]) * self.scale

        position = minute_of_day / sample_interval
        lower = int(np.floor(position)) % len(template)
        upper = (lower + 1) % len(template)
        fraction = position - np.floor(position)
        interpolated = (1.0 - fraction) * template[lower] + fraction * template[upper]
        return float(interpolated) * self.scale

    def get_state(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile.id,
            "seed": self.seed,
            "scale": self.scale,
            "max_residual_rate_mgdl_per_min": self.max_residual_rate_mgdl_per_min,
            "template_offset": self._template_offset,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.seed = int(state.get("seed", self.seed))
        self.scale = float(state.get("scale", self.scale))
        self.max_residual_rate_mgdl_per_min = state.get(
            "max_residual_rate_mgdl_per_min",
            self.max_residual_rate_mgdl_per_min,
        )
        self._templates = self._prepare_templates(self.profile.templates)
        self._template_offset = int(state.get("template_offset", self.seed % len(self.profile.templates)))
