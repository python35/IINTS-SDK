from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .contracts import ModelReadyContract, ValidationSpec, parse_contract


TransformHook = Callable[[pd.DataFrame], pd.DataFrame]

_RESERVED_TOKENS = {
    "and",
    "or",
    "not",
    "is",
    "null",
    "true",
    "false",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    failed_rows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "failed_rows": self.failed_rows,
        }


@dataclass(frozen=True)
class ValidationResult:
    is_compliant: bool
    compliance_score: float
    contract_fingerprint_sha256: str
    dataset_fingerprint_sha256: str
    row_count: int
    checks: List[CheckResult]
    output_columns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "contract_fingerprint_sha256": self.contract_fingerprint_sha256,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "row_count": self.row_count,
            "output_columns": self.output_columns,
            "checks": [check.to_dict() for check in self.checks],
        }


def _normalize_column_ref(ref: str) -> str:
    token = ref.strip()
    if "." in token:
        token = token.split(".")[-1]
    return token.strip()


def _expression_tokens(expression: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
    return [token for token in tokens if token.lower() not in _RESERVED_TOKENS]


def _collect_required_columns(contract: ModelReadyContract) -> List[str]:
    columns: set[str] = set()
    for stream in contract.streams:
        required = stream.metadata.get("required_columns", [])
        if isinstance(required, list):
            for column in required:
                columns.add(str(column).strip())
    for process in contract.processes:
        columns.add(_normalize_column_ref(process.input_stream))
        for feature in process.features:
            columns.add(_normalize_column_ref(feature.source))
        for label in process.labels:
            for token in _expression_tokens(label.expression):
                columns.add(token)
        for rule in process.validations:
            for token in _expression_tokens(rule.expression):
                columns.add(token)
    return sorted(column for column in columns if column)


def _collect_expected_types(contract: ModelReadyContract) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for stream in contract.streams:
        raw_types = stream.metadata.get("column_types", {})
        if isinstance(raw_types, dict):
            for name, expected in raw_types.items():
                types[str(name).strip()] = str(expected).strip().lower()
    return types


def _collect_ranges(contract: ModelReadyContract) -> Dict[str, Dict[str, float]]:
    ranges: Dict[str, Dict[str, float]] = {}
    for stream in contract.streams:
        raw_ranges = stream.metadata.get("ranges", {})
        if not isinstance(raw_ranges, dict):
            continue
        for column, raw_bounds in raw_ranges.items():
            if not isinstance(raw_bounds, dict):
                continue
            parsed: Dict[str, float] = {}
            if "min" in raw_bounds:
                parsed["min"] = float(raw_bounds["min"])
            if "max" in raw_bounds:
                parsed["max"] = float(raw_bounds["max"])
            if parsed:
                ranges[str(column).strip()] = parsed
    return ranges


def _matches_expected_type(series: pd.Series, expected: str) -> bool:
    if expected in {"float", "number", "numeric"}:
        return bool(pd.api.types.is_numeric_dtype(series))
    if expected in {"int", "integer"}:
        return bool(pd.api.types.is_integer_dtype(series))
    if expected in {"str", "string"}:
        return bool(pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series))
    if expected in {"bool", "boolean"}:
        return bool(pd.api.types.is_bool_dtype(series))
    if expected in {"datetime", "timestamp"}:
        return bool(pd.api.types.is_datetime64_any_dtype(series))
    return False


def _evaluate_clause(df: pd.DataFrame, clause: str) -> pd.Series:
    normalized = clause.strip().strip("()").strip()
    if not normalized:
        return pd.Series([True] * len(df), index=df.index, dtype=bool)

    match_not_null = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s+is\s+not\s+null", normalized, flags=re.IGNORECASE)
    if match_not_null:
        column = match_not_null.group(1)
        if column not in df.columns:
            raise KeyError(column)
        return df[column].notna()

    match_null = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s+is\s+null", normalized, flags=re.IGNORECASE)
    if match_null:
        column = match_null.group(1)
        if column not in df.columns:
            raise KeyError(column)
        return df[column].isna()

    match_cmp = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)",
        normalized,
    )
    if match_cmp:
        column = match_cmp.group(1)
        if column not in df.columns:
            raise KeyError(column)
        op = match_cmp.group(2)
        value = float(match_cmp.group(3))
        numeric = pd.to_numeric(df[column], errors="coerce")
        if op == "<=":
            return numeric <= value
        if op == ">=":
            return numeric >= value
        if op == "==":
            return numeric == value
        if op == "!=":
            return numeric != value
        if op == "<":
            return numeric < value
        return numeric > value

    raise ValueError(f"Unsupported validation clause: '{clause}'")


def _evaluate_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    # Supported grammar:
    #   clause AND clause ...
    #   (clause AND ...) OR (clause AND ...)
    # where each clause is one of:
    #   <col> is not null
    #   <col> is null
    #   <col> <op> <number>
    or_groups = re.split(r"\s+or\s+", expression, flags=re.IGNORECASE)
    result = pd.Series([False] * len(df), index=df.index, dtype=bool)
    for group in or_groups:
        and_parts = re.split(r"\s+and\s+", group, flags=re.IGNORECASE)
        group_mask = pd.Series([True] * len(df), index=df.index, dtype=bool)
        for part in and_parts:
            group_mask = group_mask & _evaluate_clause(df, part)
        result = result | group_mask
    return result


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    normalized = df.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            normalized[column] = series.astype("datetime64[ns]").astype("int64").astype(str)
        else:
            normalized[column] = series.astype(str).fillna("__NA__")
    hashed_values = pd.util.hash_pandas_object(normalized, index=True).to_numpy(dtype=np.uint64, copy=False)
    row_hash = hashed_values.tobytes()
    digest = sha256()
    digest.update(",".join(normalized.columns).encode("utf-8"))
    digest.update(row_hash)
    return digest.hexdigest()


class ContractRunner:
    """
    Lightweight executor for model-ready data contracts.
    """

    def __init__(self, contract: Union[ModelReadyContract, Dict[str, Any]]) -> None:
        self.contract = parse_contract(contract) if isinstance(contract, dict) else contract

    def _apply_builtin_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        for stream in self.contract.streams:
            raw_conversions = stream.metadata.get("unit_conversions", {})
            if not isinstance(raw_conversions, dict):
                continue
            for column, spec in raw_conversions.items():
                if column not in transformed.columns or not isinstance(spec, dict):
                    continue
                from_unit = str(spec.get("from", "")).strip().lower()
                to_unit = str(spec.get("to", "")).strip().lower()
                if from_unit == "mmol/l" and to_unit == "mg/dl":
                    transformed[column] = pd.to_numeric(transformed[column], errors="coerce") * 18.0182
                elif from_unit == "mg/dl" and to_unit == "mmol/l":
                    transformed[column] = pd.to_numeric(transformed[column], errors="coerce") / 18.0182
        return transformed

    def run(
        self,
        df: pd.DataFrame,
        *,
        transform_hooks: Optional[Iterable[TransformHook]] = None,
        apply_builtin_transforms: bool = True,
    ) -> ValidationResult:
        working = df.copy()
        if apply_builtin_transforms:
            working = self._apply_builtin_transforms(working)
        if transform_hooks is not None:
            for hook in transform_hooks:
                working = hook(working)

        checks: List[CheckResult] = []

        required_columns = _collect_required_columns(self.contract)
        missing_columns = [column for column in required_columns if column not in working.columns]
        checks.append(
            CheckResult(
                name="schema_columns",
                passed=len(missing_columns) == 0,
                detail="all required columns present" if not missing_columns else f"missing columns: {missing_columns}",
            )
        )

        expected_types = _collect_expected_types(self.contract)
        type_failures: List[str] = []
        for column, expected in expected_types.items():
            if column not in working.columns:
                continue
            if not _matches_expected_type(working[column], expected):
                type_failures.append(f"{column} expected {expected}")
        checks.append(
            CheckResult(
                name="schema_types",
                passed=len(type_failures) == 0,
                detail="column types match" if not type_failures else "; ".join(type_failures),
            )
        )

        ranges = _collect_ranges(self.contract)
        range_failures = 0
        range_details: List[str] = []
        for column, bounds in ranges.items():
            if column not in working.columns:
                continue
            values = pd.to_numeric(working[column], errors="coerce")
            if "min" in bounds:
                violations = int((values < bounds["min"]).fillna(False).sum())
                if violations > 0:
                    range_failures += violations
                    range_details.append(f"{column} below {bounds['min']}: {violations}")
            if "max" in bounds:
                violations = int((values > bounds["max"]).fillna(False).sum())
                if violations > 0:
                    range_failures += violations
                    range_details.append(f"{column} above {bounds['max']}: {violations}")
        checks.append(
            CheckResult(
                name="value_ranges",
                passed=range_failures == 0,
                detail="all ranges satisfied" if range_failures == 0 else "; ".join(range_details),
                failed_rows=range_failures,
            )
        )

        validation_rules: List[ValidationSpec] = []
        for process in self.contract.processes:
            validation_rules.extend(process.validations)

        validation_failures = 0
        validation_details: List[str] = []
        for rule in validation_rules:
            try:
                mask = _evaluate_expression(working, rule.expression)
                failed_rows = int((~mask).sum())
                if failed_rows > 0:
                    validation_failures += failed_rows
                    validation_details.append(f"{rule.expression} failed rows={failed_rows}")
            except Exception as exc:
                validation_failures += len(working)
                validation_details.append(f"{rule.expression} error: {exc}")
        checks.append(
            CheckResult(
                name="rule_validations",
                passed=validation_failures == 0,
                detail="all validation rules satisfied" if validation_failures == 0 else "; ".join(validation_details),
                failed_rows=validation_failures,
            )
        )

        passed_count = sum(1 for check in checks if check.passed)
        compliance_score = round((passed_count / max(len(checks), 1)) * 100.0, 2)

        return ValidationResult(
            is_compliant=all(check.passed for check in checks),
            compliance_score=compliance_score,
            contract_fingerprint_sha256=self.contract.fingerprint(),
            dataset_fingerprint_sha256=dataframe_fingerprint(working),
            row_count=len(working),
            checks=checks,
            output_columns=list(working.columns),
        )
