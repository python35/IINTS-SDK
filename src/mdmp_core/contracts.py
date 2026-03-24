from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List
import json

import yaml

from mdmp_core.exceptions import MDMPContractError


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str
    unit: str | None = None
    bounds: tuple[float, float] | None = None
    required: bool = True


@dataclass(frozen=True)
class ConsentSpec:
    ai_training_allowed: bool = False
    jurisdiction: str = "unspecified"
    anonymized: bool = False
    consent_date: str | None = None
    expiry: str | None = None
    legal_basis: str | None = None


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    version: str
    industry: str
    columns: list[ColumnSpec] = field(default_factory=list)


@dataclass(frozen=True)
class DataContract:
    schema: SchemaSpec
    consent: ConsentSpec

    def to_dict(self) -> dict[str, Any]:
        cols: list[dict[str, Any]] = []
        for col in self.schema.columns:
            row: dict[str, Any] = {
                "name": col.name,
                "type": col.type,
                "required": col.required,
            }
            if col.unit is not None:
                row["unit"] = col.unit
            if col.bounds is not None:
                row["bounds"] = [float(col.bounds[0]), float(col.bounds[1])]
            cols.append(row)

        return {
            "schema": {
                "name": self.schema.name,
                "version": self.schema.version,
                "industry": self.schema.industry,
                "columns": cols,
            },
            "consent": {
                "ai_training_allowed": self.consent.ai_training_allowed,
                "jurisdiction": self.consent.jurisdiction,
                "anonymized": self.consent.anonymized,
                "consent_date": self.consent.consent_date,
                "expiry": self.consent.expiry,
                "legal_basis": self.consent.legal_basis,
            },
        }

    def fingerprint_sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()


def _parse_column(raw: dict[str, Any]) -> ColumnSpec:
    if "name" not in raw or "type" not in raw:
        raise MDMPContractError("Every column requires both 'name' and 'type'")
    bounds_raw = raw.get("bounds")
    bounds: tuple[float, float] | None = None
    if bounds_raw is not None:
        if not isinstance(bounds_raw, list) or len(bounds_raw) != 2:
            raise MDMPContractError("Column 'bounds' must be [min, max]")
        try:
            bounds = (float(bounds_raw[0]), float(bounds_raw[1]))
        except (TypeError, ValueError) as exc:
            raise MDMPContractError(
                f"Invalid bounds for column '{raw.get('name', '<unknown>')}': {bounds_raw!r}"
            ) from exc

    return ColumnSpec(
        name=str(raw["name"]).strip(),
        type=str(raw["type"]).strip().lower(),
        unit=str(raw.get("unit")).strip() if raw.get("unit") is not None else None,
        bounds=bounds,
        required=bool(raw.get("required", True)),
    )


def parse_contract(payload: dict[str, Any]) -> DataContract:
    if not isinstance(payload, dict):
        raise MDMPContractError("Contract root must be a mapping")
    schema_raw = payload.get("schema", {})
    consent_raw = payload.get("consent", {})

    if not isinstance(schema_raw, dict):
        raise MDMPContractError("'schema' must be a mapping")
    if not isinstance(consent_raw, dict):
        raise MDMPContractError("'consent' must be a mapping")

    columns_raw = schema_raw.get("columns", [])
    if not isinstance(columns_raw, list):
        raise MDMPContractError("'schema.columns' must be a list")

    schema = SchemaSpec(
        name=str(schema_raw.get("name", "dataset")).strip(),
        version=str(schema_raw.get("version", "1.0")).strip(),
        industry=str(schema_raw.get("industry", "general")).strip(),
        columns=[_parse_column(dict(col)) for col in columns_raw],
    )

    consent = ConsentSpec(
        ai_training_allowed=bool(consent_raw.get("ai_training_allowed", False)),
        jurisdiction=str(consent_raw.get("jurisdiction", "unspecified")).strip(),
        anonymized=bool(consent_raw.get("anonymized", False)),
        consent_date=(
            str(consent_raw["consent_date"]).strip()
            if consent_raw.get("consent_date") is not None and str(consent_raw.get("consent_date")).strip() != ""
            else None
        ),
        expiry=(
            str(consent_raw["expiry"]).strip()
            if consent_raw.get("expiry") is not None and str(consent_raw.get("expiry")).strip() != ""
            else None
        ),
        legal_basis=(
            str(consent_raw["legal_basis"]).strip()
            if consent_raw.get("legal_basis") is not None and str(consent_raw.get("legal_basis")).strip() != ""
            else None
        ),
    )

    return DataContract(schema=schema, consent=consent)


def load_contract(path: Path) -> DataContract:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise MDMPContractError(f"Failed to parse YAML contract: {path}") from exc
    if not isinstance(payload, dict):
        raise MDMPContractError("Contract must be a mapping")
    return parse_contract(payload)


def save_contract(path: Path, contract: DataContract) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(contract.to_dict(), sort_keys=False), encoding="utf-8")
