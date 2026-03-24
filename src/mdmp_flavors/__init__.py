from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Dict

FlavorFactory = Callable[[], Dict[str, Any]]


def _health_template() -> Dict[str, Any]:
    return {
        "schema": {
            "name": "cgm_dataset",
            "version": "1.0",
            "industry": "health",
            "columns": [
                {"name": "timestamp", "type": "datetime", "required": True},
                {"name": "glucose", "type": "float", "unit": "mg/dL", "bounds": [40, 400], "required": True},
            ],
        },
        "consent": {
            "ai_training_allowed": True,
            "jurisdiction": "GDPR",
            "anonymized": True,
            "consent_date": "2025-01-01T00:00:00Z",
            "expiry": "2027-01-01T00:00:00Z",
            "legal_basis": "explicit_consent",
        },
    }


def _finance_template() -> Dict[str, Any]:
    return {
        "schema": {
            "name": "timeseries_dataset",
            "version": "1.0",
            "industry": "finance",
            "columns": [
                {"name": "timestamp", "type": "datetime", "required": True},
                {"name": "value", "type": "float", "required": True},
            ],
        },
        "consent": {
            "ai_training_allowed": True,
            "jurisdiction": "EU",
            "anonymized": True,
            "consent_date": "2025-01-01T00:00:00Z",
            "expiry": "2027-01-01T00:00:00Z",
            "legal_basis": "contractual",
        },
    }


def _industrial_template() -> Dict[str, Any]:
    return {
        "schema": {
            "name": "sensor_dataset",
            "version": "1.0",
            "industry": "industrial",
            "columns": [
                {"name": "timestamp", "type": "datetime", "required": True},
                {"name": "sensor_value", "type": "float", "required": True},
            ],
        },
        "consent": {
            "ai_training_allowed": True,
            "jurisdiction": "internal",
            "anonymized": True,
            "consent_date": "2025-01-01T00:00:00Z",
            "expiry": "2027-01-01T00:00:00Z",
            "legal_basis": "legitimate_interest",
        },
    }


def _llm_template() -> Dict[str, Any]:
    return {
        "schema": {
            "name": "llm_corpus",
            "version": "1.0",
            "industry": "ai_text",
            "columns": [
                {"name": "doc_id", "type": "string", "required": True},
                {"name": "source", "type": "string", "required": True},
                {"name": "license", "type": "string", "required": True},
                {"name": "tokens", "type": "int", "required": True},
            ],
        },
        "consent": {
            "ai_training_allowed": True,
            "jurisdiction": "multi_region",
            "anonymized": True,
            "consent_date": "2025-01-01T00:00:00Z",
            "expiry": "2027-01-01T00:00:00Z",
            "legal_basis": "license_compliance",
        },
    }


BUILTIN_TEMPLATES: Dict[str, FlavorFactory] = {
    "health": _health_template,
    "finance": _finance_template,
    "industrial": _industrial_template,
    "llm": _llm_template,
}


def load_external_templates() -> Dict[str, FlavorFactory]:
    discovered: Dict[str, FlavorFactory] = {}
    for ep in entry_points(group="mdmp.flavors"):
        loaded: Any | None = None
        try:
            loaded = ep.load()
        except Exception:
            loaded = None
        if callable(loaded):
            discovered[str(ep.name).strip().lower()] = loaded
    return discovered


def resolve_templates() -> Dict[str, FlavorFactory]:
    merged: Dict[str, FlavorFactory] = dict(BUILTIN_TEMPLATES)
    merged.update(load_external_templates())
    return merged


def available_flavors() -> list[str]:
    return sorted(resolve_templates().keys())


def get_template(flavor: str) -> Dict[str, Any]:
    normalized = flavor.strip().lower()
    templates = resolve_templates()
    if normalized not in templates:
        allowed = ", ".join(sorted(templates))
        raise KeyError(f"Unknown flavor '{flavor}'. Allowed: {allowed}")
    return templates[normalized]()


__all__ = [
    "FlavorFactory",
    "BUILTIN_TEMPLATES",
    "load_external_templates",
    "resolve_templates",
    "available_flavors",
    "get_template",
]
