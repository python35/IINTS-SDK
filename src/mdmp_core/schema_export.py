from __future__ import annotations

from typing import Any

from mdmp_core.contracts import DataContract, parse_contract
from mdmp_core.exceptions import MDMPContractError


TYPE_MAP = {
    "float": "number",
    "int": "integer",
    "string": "string",
    "datetime": "string",
    "boolean": "boolean",
}

FRICTIONLESS_TYPE_MAP = {
    "float": "number",
    "int": "integer",
    "string": "string",
    "datetime": "datetime",
    "boolean": "boolean",
}


def _as_contract(payload: DataContract | dict[str, Any]) -> DataContract:
    if isinstance(payload, DataContract):
        return payload
    if not isinstance(payload, dict):
        raise MDMPContractError("contract payload must be a DataContract or mapping")
    return parse_contract(payload)


def contract_to_json_schema(payload: DataContract | dict[str, Any]) -> dict[str, Any]:
    """Convert an MDMP contract into JSON Schema (draft-07)."""
    contract = _as_contract(payload)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for column in contract.schema.columns:
        prop: dict[str, Any] = {"type": TYPE_MAP.get(column.type, "string")}
        if column.type == "datetime":
            prop["format"] = "date-time"
        if column.bounds is not None:
            prop["minimum"] = float(column.bounds[0])
            prop["maximum"] = float(column.bounds[1])
        if column.unit is not None:
            prop["x-unit"] = column.unit
        if column.required:
            required.append(column.name)
        properties[column.name] = prop

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": contract.schema.name or "mdmp_dataset",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True,
    }


def contract_to_frictionless_schema(payload: DataContract | dict[str, Any]) -> dict[str, Any]:
    """Convert an MDMP contract into a Frictionless Table Schema document."""
    contract = _as_contract(payload)
    fields: list[dict[str, Any]] = []
    required: list[str] = []

    for column in contract.schema.columns:
        field_payload: dict[str, Any] = {
            "name": column.name,
            "type": FRICTIONLESS_TYPE_MAP.get(column.type, "string"),
        }
        if column.unit is not None:
            field_payload["unit"] = column.unit
        if column.bounds is not None:
            field_payload["constraints"] = {
                "minimum": float(column.bounds[0]),
                "maximum": float(column.bounds[1]),
            }
        if column.required:
            required.append(column.name)
        fields.append(field_payload)

    return {
        "name": contract.schema.name,
        "title": contract.schema.name,
        "version": contract.schema.version,
        "missingValues": [""],
        "fields": fields,
        "primaryKey": required[0] if required else None,
    }
