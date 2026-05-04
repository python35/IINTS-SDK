from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

try:  # Python 3.9+
    from importlib.resources import files
except Exception:  # pragma: no cover
    files = None  # type: ignore
    from importlib import resources
else:
    from importlib import resources

from .registry import DatasetRegistryError, get_dataset


ReferenceStatus = Literal["passed", "warning", "failed", "skipped"]


@dataclass(frozen=True)
class ReferenceBand:
    warning_low: float
    target_low: float
    median: float
    target_high: float
    warning_high: float

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReferenceBand":
        return cls(
            warning_low=float(payload["warning_low"]),
            target_low=float(payload["target_low"]),
            median=float(payload["median"]),
            target_high=float(payload["target_high"]),
            warning_high=float(payload["warning_high"]),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "warning_low": round(self.warning_low, 4),
            "target_low": round(self.target_low, 4),
            "median": round(self.median, 4),
            "target_high": round(self.target_high, 4),
            "warning_high": round(self.warning_high, 4),
        }


@dataclass(frozen=True)
class ReferenceComparison:
    metric_key: str
    label: str
    observed_value: float | None
    status: ReferenceStatus
    detail: str
    band: ReferenceBand

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "label": self.label,
            "observed_value": None if self.observed_value is None else round(self.observed_value, 4),
            "status": self.status,
            "detail": self.detail,
            "band": self.band.to_dict(),
        }


@dataclass(frozen=True)
class RealismReferenceProfile:
    id: str
    label: str
    source: str
    description: str
    dataset_ids: List[str]
    metric_bands: Dict[str, ReferenceBand]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RealismReferenceProfile":
        metric_bands = {
            str(metric_key): ReferenceBand.from_dict(metric_band)
            for metric_key, metric_band in dict(payload.get("metric_bands", {})).items()
        }
        return cls(
            id=str(payload["id"]),
            label=str(payload["label"]),
            source=str(payload["source"]),
            description=str(payload.get("description", "")),
            dataset_ids=[str(value) for value in payload.get("dataset_ids", [])],
            metric_bands=metric_bands,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "source": self.source,
            "description": self.description,
            "dataset_ids": self.dataset_ids,
            "metric_bands": {
                metric_key: metric_band.to_dict()
                for metric_key, metric_band in self.metric_bands.items()
            },
        }


def _read_reference_registry_text() -> str:
    try:
        if files is not None:
            return files("iints.data").joinpath("realism_references.json").read_text()  # type: ignore[call-arg]
        return resources.read_text("iints.data", "realism_references.json")
    except Exception as exc:
        raise DatasetRegistryError(f"Unable to locate realism_references.json: {exc}") from exc


def load_realism_reference_registry() -> List[RealismReferenceProfile]:
    payload = json.loads(_read_reference_registry_text())
    return [RealismReferenceProfile.from_dict(entry) for entry in payload]


def list_realism_reference_ids() -> List[str]:
    return [profile.id for profile in load_realism_reference_registry()]


def get_realism_reference(reference_id: str) -> RealismReferenceProfile:
    normalized = reference_id.strip()
    for profile in load_realism_reference_registry():
        if profile.id == normalized:
            return profile
    try:
        dataset = get_dataset(normalized)
    except DatasetRegistryError as exc:
        available = ", ".join(list_realism_reference_ids())
        raise DatasetRegistryError(
            f"Unknown realism reference '{reference_id}'. Available references: {available}"
        ) from exc

    mapped = dataset.get("realism_reference_profile")
    if not isinstance(mapped, str) or not mapped.strip():
        raise DatasetRegistryError(
            f"Dataset '{normalized}' does not define a realism reference profile yet."
        )
    return get_realism_reference(mapped)


def compare_to_reference_band(
    metric_key: str,
    label: str,
    observed_value: float | None,
    band: ReferenceBand,
) -> ReferenceComparison:
    if observed_value is None:
        return ReferenceComparison(
            metric_key=metric_key,
            label=label,
            observed_value=None,
            status="skipped",
            detail="This metric was not available for the current trace.",
            band=band,
        )

    if band.target_low <= observed_value <= band.target_high:
        status: ReferenceStatus = "passed"
        detail = (
            f"Observed {observed_value:.1f}, which sits inside the reference target band "
            f"[{band.target_low:.1f}, {band.target_high:.1f}]."
        )
    elif band.warning_low <= observed_value <= band.warning_high:
        status = "warning"
        detail = (
            f"Observed {observed_value:.1f}, which is still within the wider reference envelope "
            f"[{band.warning_low:.1f}, {band.warning_high:.1f}] but outside the target band."
        )
    else:
        status = "failed"
        detail = (
            f"Observed {observed_value:.1f}, which falls outside the reference envelope "
            f"[{band.warning_low:.1f}, {band.warning_high:.1f}]."
        )
    return ReferenceComparison(
        metric_key=metric_key,
        label=label,
        observed_value=observed_value,
        status=status,
        detail=detail,
        band=band,
    )


def metric_label(metric_key: str) -> str:
    labels = {
        "mean_glucose_mgdl": "Mean glucose (mg/dL)",
        "sd_mgdl": "SD (mg/dL)",
        "cv_pct": "CV (%)",
        "tir_70_180_pct": "TIR 70-180 (%)",
        "tir_above_180_pct": "Time >180 (%)",
        "tir_below_70_pct": "Time <70 (%)",
        "glucose_range_mgdl": "Daily range (mg/dL)",
        "meal_response_ratio": "Meal response ratio",
        "median_peak_lag_minutes": "Median peak lag (min)",
        "median_meal_rise_mgdl": "Median meal rise (mg/dL)",
    }
    return labels.get(metric_key, metric_key.replace("_", " "))


def build_reference_comparisons(
    observed_metrics: Dict[str, Any],
    profile: RealismReferenceProfile,
) -> List[ReferenceComparison]:
    comparisons: List[ReferenceComparison] = []
    for metric_key, band in profile.metric_bands.items():
        raw_value = observed_metrics.get(metric_key)
        observed_value: float | None
        if raw_value is None:
            observed_value = None
        else:
            try:
                observed_value = float(raw_value)
            except (TypeError, ValueError):
                observed_value = None
        comparisons.append(
            compare_to_reference_band(
                metric_key=metric_key,
                label=metric_label(metric_key),
                observed_value=observed_value,
                band=band,
            )
        )
    return comparisons
