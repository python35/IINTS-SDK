from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union, cast
import warnings

import pandas as pd

from .contracts import ModelReadyContract, load_contract_yaml, parse_contract
from .runner import ContractRunner, ValidationResult, mdmp_grade_meets_minimum


LOGGER = logging.getLogger(__name__)


ContractInput = Union[ModelReadyContract, Dict[str, Any], Path, str]
GateFailMode = str


@dataclass(frozen=True)
class MDMPGateError(RuntimeError):
    message: str
    result: ValidationResult

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _resolve_contract(contract: ContractInput) -> ModelReadyContract:
    if isinstance(contract, ModelReadyContract):
        return contract
    if isinstance(contract, dict):
        return parse_contract(contract)
    path = Path(contract)
    return load_contract_yaml(path)


def _extract_dataframe(
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    dataframe_arg: Optional[str],
) -> pd.DataFrame:
    if dataframe_arg is not None:
        candidate = kwargs.get(dataframe_arg)
        if isinstance(candidate, pd.DataFrame):
            return candidate
        raise TypeError(f"mdmp_gate expected DataFrame keyword argument '{dataframe_arg}'")

    for arg in args:
        if isinstance(arg, pd.DataFrame):
            return arg
    for value in kwargs.values():
        if isinstance(value, pd.DataFrame):
            return value
    raise TypeError("mdmp_gate could not locate a pandas.DataFrame argument")


def mdmp_gate(
    contract: ContractInput,
    *,
    min_grade: str = "research_grade",
    fail_mode: GateFailMode = "raise",
    dataframe_arg: Optional[str] = None,
    apply_builtin_transforms: bool = True,
    transform_hooks: Optional[Iterable[Callable[[pd.DataFrame], pd.DataFrame]]] = None,
    on_result: Optional[Callable[[ValidationResult], None]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that enforces MDMP compliance before function execution.

    The wrapped function must receive at least one pandas DataFrame argument
    (or explicitly provide `dataframe_arg`).
    """
    normalized_grade = min_grade.strip().lower()
    normalized_fail_mode = fail_mode.strip().lower()
    if normalized_fail_mode not in {"raise", "warn", "log"}:
        raise ValueError("fail_mode must be one of: raise, warn, log")

    contract_obj = _resolve_contract(contract)
    runner = ContractRunner(contract_obj)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            df = _extract_dataframe(args, kwargs, dataframe_arg=dataframe_arg)
            result = runner.run(
                df,
                apply_builtin_transforms=apply_builtin_transforms,
                transform_hooks=transform_hooks,
            )
            setattr(wrapped, "__mdmp_last_result__", result)

            is_grade_ok = mdmp_grade_meets_minimum(result.mdmp_grade, normalized_grade)
            gate_ok = bool(result.is_compliant and is_grade_ok)

            if on_result is not None:
                on_result(result)

            if not gate_ok:
                message = (
                    f"MDMP gate blocked execution for '{func.__name__}': "
                    f"grade={result.mdmp_grade}, required>={normalized_grade}, "
                    f"compliance={result.compliance_score:.2f}%"
                )
                if normalized_fail_mode == "raise":
                    raise MDMPGateError(message=message, result=result)
                if normalized_fail_mode == "warn":
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                else:
                    LOGGER.warning(message)

            return func(*args, **kwargs)

        return cast(Callable[..., Any], wrapped)

    return decorator
