"""Reusable physiology math helpers for patient models.

These helpers keep model assumptions explicit and testable. They are research
abstractions, not personalized clinical physiology estimators.
"""
from __future__ import annotations

import math
from typing import Any


PICOGRAMS_PER_MILLIGRAM = 1_000_000_000.0


def validated_snapshot_scalar(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a finite snapshot scalar inside explicitly declared bounds."""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"snapshot {name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"snapshot {name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"snapshot {name} must be at least {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"snapshot {name} must be at most {maximum}")
    return numeric


def validated_snapshot_bool(value: Any, *, name: str) -> bool:
    """Reject truthy strings and numbers when restoring boolean state."""

    if not isinstance(value, bool):
        raise ValueError(f"snapshot {name} must be a boolean")
    return value


def validated_activity_events(
    value: Any,
    *,
    name: str,
    age_key: str,
) -> list[dict[str, float]]:
    """Validate dose/meal bookkeeping events restored from a snapshot."""

    if not isinstance(value, list):
        raise ValueError(f"snapshot {name} must be a list")
    validated: list[dict[str, float]] = []
    for index, event in enumerate(value):
        if not isinstance(event, dict):
            raise ValueError(f"snapshot {name}[{index}] must be an object")
        if "amount" not in event or age_key not in event:
            raise ValueError(
                f"snapshot {name}[{index}] requires amount and {age_key}"
            )
        amount = validated_snapshot_scalar(
            event["amount"], name=f"{name}[{index}].amount", minimum=0.0
        )
        age = validated_snapshot_scalar(
            event[age_key], name=f"{name}[{index}].{age_key}", minimum=0.0
        )
        validated.append({"amount": amount, age_key: age})
    return validated


def glucagon_mg_to_pg(dose_mg: float) -> float:
    """Convert glucagon mass from milligrams to picograms."""

    dose = float(dose_mg)
    if not math.isfinite(dose) or dose < 0.0:
        raise ValueError("dose_mg must be finite and non-negative")
    return dose * PICOGRAMS_PER_MILLIGRAM


def dawn_glucose_rate_mgdl_min(
    current_time_minutes: float,
    *,
    peak_strength_mgdl_per_hour: float,
    start_hour: float,
    end_hour: float,
) -> float:
    """Return a smooth, explicitly phenomenological dawn glucose-rate term.

    ``peak_strength_mgdl_per_hour`` has the same public meaning in every
    patient backend.  A raised-cosine window avoids discontinuous rates at the
    configured start and end.  This is a scenario input, not an inferred
    cortisol, growth-hormone, or endogenous-glucose-production measurement.
    """

    values = {
        "current_time_minutes": float(current_time_minutes),
        "peak_strength_mgdl_per_hour": float(peak_strength_mgdl_per_hour),
        "start_hour": float(start_hour),
        "end_hour": float(end_hour),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("dawn-profile inputs must all be finite")
    if values["peak_strength_mgdl_per_hour"] < 0.0:
        raise ValueError("peak_strength_mgdl_per_hour must be non-negative")
    if not 0.0 <= values["start_hour"] < values["end_hour"] <= 24.0:
        raise ValueError("dawn hours must satisfy 0 <= start < end <= 24")

    minute_of_day = values["current_time_minutes"] % 1_440.0
    start_minute = values["start_hour"] * 60.0
    end_minute = values["end_hour"] * 60.0
    if minute_of_day < start_minute or minute_of_day > end_minute:
        return 0.0

    midpoint = 0.5 * (start_minute + end_minute)
    half_width = 0.5 * (end_minute - start_minute)
    phase = math.pi * (minute_of_day - midpoint) / half_width
    window = 0.5 * (1.0 + math.cos(phase))
    return values["peak_strength_mgdl_per_hour"] * window / 60.0


def antecedent_hypoglycemia_memory_derivative(
    glucose_mgdl: float,
    memory: float,
    *,
    awareness_threshold_mgdl: float = 70.0,
    severe_threshold_mgdl: float = 54.0,
    build_time_constant_minutes: float = 360.0,
    recovery_time_constant_minutes: float = 4_320.0,
) -> float:
    """Return a bounded antecedent-hypoglycemia exposure-memory derivative.

    This is a research abstraction, not a diagnostic HAAF model. Exposure
    accumulates gradually during hypoglycemia and recovers over several days.
    The time constants are deliberately explicit so they can be calibrated
    against clamp-study data instead of being hidden in model code.
    """

    values = {
        "glucose_mgdl": float(glucose_mgdl),
        "memory": float(memory),
        "awareness_threshold_mgdl": float(awareness_threshold_mgdl),
        "severe_threshold_mgdl": float(severe_threshold_mgdl),
        "build_time_constant_minutes": float(build_time_constant_minutes),
        "recovery_time_constant_minutes": float(recovery_time_constant_minutes),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("HAAF abstraction inputs must all be finite")
    if values["awareness_threshold_mgdl"] <= values["severe_threshold_mgdl"]:
        raise ValueError("awareness threshold must exceed severe threshold")
    if values["build_time_constant_minutes"] <= 0.0:
        raise ValueError("build_time_constant_minutes must be positive")
    if values["recovery_time_constant_minutes"] <= 0.0:
        raise ValueError("recovery_time_constant_minutes must be positive")

    current_memory = min(1.0, max(0.0, values["memory"]))
    severity_width = (
        values["awareness_threshold_mgdl"] - values["severe_threshold_mgdl"]
    )
    severity = min(
        1.5,
        max(
            0.0,
            (values["awareness_threshold_mgdl"] - values["glucose_mgdl"])
            / severity_width,
        ),
    )
    build = severity * (1.0 - current_memory) / values["build_time_constant_minutes"]
    recovery = current_memory / values["recovery_time_constant_minutes"]
    return build - recovery


def counterregulatory_rescue_multiplier(
    glucose_mgdl: float,
    memory: float,
    *,
    threshold_mgdl: float = 70.0,
    half_activation_mgdl: float = 16.0,
    maximum_fractional_increase: float = 1.0,
) -> float:
    """Return a saturating endogenous counter-regulatory EGP multiplier.

    The response is bounded and attenuated by the exposure-memory state. It
    captures directionality only; it is not a patient-specific hormone model.
    """

    glucose = float(glucose_mgdl)
    current_memory = float(memory)
    threshold = float(threshold_mgdl)
    half_activation = float(half_activation_mgdl)
    maximum_increase = float(maximum_fractional_increase)
    if not all(
        math.isfinite(value)
        for value in (
            glucose,
            current_memory,
            threshold,
            half_activation,
            maximum_increase,
        )
    ):
        raise ValueError("counter-regulation inputs must all be finite")
    if half_activation <= 0.0:
        raise ValueError("half_activation_mgdl must be positive")
    if maximum_increase < 0.0:
        raise ValueError("maximum_fractional_increase must be non-negative")

    deficit = max(0.0, threshold - glucose)
    activation = deficit / (half_activation + deficit)
    retained_response = 1.0 - min(1.0, max(0.0, current_memory))
    return 1.0 + maximum_increase * activation * retained_response


def smooth_threshold_excess(value: float, *, threshold: float, splay: float = 10.0) -> float:
    """Smooth positive excess above a biological threshold.

    Uses a numerically stable softplus. Below the threshold this approaches
    zero; above it approaches ``value - threshold``. The ``splay`` parameter
    controls the smooth transition width.
    """

    observed = float(value)
    cutoff = float(threshold)
    width = float(splay)
    if not all(math.isfinite(item) for item in (observed, cutoff, width)):
        raise ValueError("smooth threshold inputs must all be finite")
    if width <= 0.0:
        raise ValueError("splay must be positive")
    z = (observed - cutoff) / width
    if z > 50.0:
        return width * z
    if z < -50.0:
        return width * math.exp(z)
    return width * math.log1p(math.exp(z))


def renal_glucose_clearance_concentration(
    glucose_mgdl: float,
    *,
    threshold_mgdl: float = 180.0,
    gain: float = 0.05,
    splay_mgdl: float = 10.0,
) -> float:
    """Return a smooth concentration-domain renal glucose loss term.

    The term models renal-threshold/splay behavior for research simulations:
    negligible below threshold and increasing smoothly above it.
    """

    renal_gain = float(gain)
    if not math.isfinite(renal_gain) or renal_gain < 0.0:
        raise ValueError("gain must be finite and non-negative")
    return renal_gain * smooth_threshold_excess(
        glucose_mgdl,
        threshold=float(threshold_mgdl),
        splay=float(splay_mgdl),
    )
