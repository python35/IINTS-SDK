"""Small runtime unit contracts for safety-relevant numeric boundaries."""

from __future__ import annotations

import math
from typing import Any


def finite_value(value: Any, *, name: str, unit: str) -> float:
    """Return a finite float or fail with a unit-aware error."""

    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric ({unit})") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite ({unit})")
    return resolved


def nonnegative_value(value: Any, *, name: str, unit: str) -> float:
    """Return a finite non-negative value with explicit units."""

    resolved = finite_value(value, name=name, unit=unit)
    if resolved < 0.0:
        raise ValueError(f"{name} must be >= 0 ({unit})")
    return resolved


def bounded_value(
    value: Any,
    *,
    name: str,
    unit: str,
    minimum: float,
    maximum: float,
) -> float:
    """Validate a finite number against an inclusive unit-aware interval."""

    resolved = finite_value(value, name=name, unit=unit)
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}] ({unit})")
    return resolved
