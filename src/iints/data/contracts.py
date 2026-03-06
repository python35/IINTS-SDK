from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import yaml


@dataclass(frozen=True)
class StreamSpec:
    name: str
    source: str
    security: str = "PII_MINIMIZED"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "security": self.security,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    operation: str
    source: str
    window: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "operation": self.operation,
            "source": self.source,
        }
        if self.window is not None:
            payload["window"] = self.window
        if self.params:
            payload["params"] = self.params
        return payload


@dataclass(frozen=True)
class LabelSpec:
    name: str
    expression: str
    classes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "expression": self.expression,
        }
        if self.classes:
            payload["classes"] = self.classes
        return payload


@dataclass(frozen=True)
class ValidationSpec:
    expression: str
    on_fail: str = "DISCARD_AND_LOG"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression": self.expression,
            "on_fail": self.on_fail,
        }


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    input_stream: str
    features: List[FeatureSpec] = field(default_factory=list)
    labels: List[LabelSpec] = field(default_factory=list)
    validations: List[ValidationSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input_stream": self.input_stream,
            "features": [feature.to_dict() for feature in self.features],
            "labels": [label.to_dict() for label in self.labels],
            "validations": [rule.to_dict() for rule in self.validations],
        }


@dataclass(frozen=True)
class ModelReadyContract:
    version: int
    streams: List[StreamSpec]
    processes: List[ProcessSpec]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "streams": [stream.to_dict() for stream in self.streams],
            "processes": [process.to_dict() for process in self.processes],
        }

    def fingerprint(self) -> str:
        canonical = canonicalize_contract(self.to_dict())
        return sha256(canonical.encode("utf-8")).hexdigest()


def canonicalize_contract(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compile_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    contract = parse_contract(payload)
    compiled = contract.to_dict()
    compiled["fingerprint_sha256"] = contract.fingerprint()
    return compiled


def parse_contract(payload: Dict[str, Any]) -> ModelReadyContract:
    if not isinstance(payload, dict):
        raise ValueError("Contract payload must be a mapping")
    streams_raw = payload.get("streams", [])
    processes_raw = payload.get("processes", [])
    if not isinstance(streams_raw, list) or not isinstance(processes_raw, list):
        raise ValueError("Contract must contain list fields 'streams' and 'processes'")

    streams: List[StreamSpec] = []
    for raw_stream in streams_raw:
        if not isinstance(raw_stream, dict):
            raise ValueError("Each stream definition must be a mapping")
        streams.append(
            StreamSpec(
                name=str(raw_stream.get("name", "")).strip(),
                source=str(raw_stream.get("source", "")).strip(),
                security=str(raw_stream.get("security", "PII_MINIMIZED")).strip(),
                metadata=dict(raw_stream.get("metadata", {}) or {}),
            )
        )

    processes: List[ProcessSpec] = []
    for raw_process in processes_raw:
        if not isinstance(raw_process, dict):
            raise ValueError("Each process definition must be a mapping")

        features: List[FeatureSpec] = []
        for raw_feature in raw_process.get("features", []) or []:
            if not isinstance(raw_feature, dict):
                raise ValueError("Each feature definition must be a mapping")
            features.append(
                FeatureSpec(
                    name=str(raw_feature.get("name", "")).strip(),
                    operation=str(raw_feature.get("operation", "")).strip(),
                    source=str(raw_feature.get("source", "")).strip(),
                    window=str(raw_feature.get("window")).strip() if raw_feature.get("window") is not None else None,
                    params=dict(raw_feature.get("params", {}) or {}),
                )
            )

        labels: List[LabelSpec] = []
        for raw_label in raw_process.get("labels", []) or []:
            if not isinstance(raw_label, dict):
                raise ValueError("Each label definition must be a mapping")
            classes = raw_label.get("classes", []) or []
            if not isinstance(classes, list):
                raise ValueError("Label classes must be a list")
            labels.append(
                LabelSpec(
                    name=str(raw_label.get("name", "")).strip(),
                    expression=str(raw_label.get("expression", "")).strip(),
                    classes=[str(value) for value in classes],
                )
            )

        validations: List[ValidationSpec] = []
        for raw_validation in raw_process.get("validations", []) or []:
            if not isinstance(raw_validation, dict):
                raise ValueError("Each validation definition must be a mapping")
            validations.append(
                ValidationSpec(
                    expression=str(raw_validation.get("expression", "")).strip(),
                    on_fail=str(raw_validation.get("on_fail", "DISCARD_AND_LOG")).strip(),
                )
            )

        processes.append(
            ProcessSpec(
                name=str(raw_process.get("name", "")).strip(),
                input_stream=str(raw_process.get("input_stream", "")).strip(),
                features=features,
                labels=labels,
                validations=validations,
            )
        )

    return ModelReadyContract(
        version=int(payload.get("version", 1)),
        streams=streams,
        processes=processes,
    )


def load_contract_yaml(path: Path) -> ModelReadyContract:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML contract root must be a mapping")
    return parse_contract(payload)
