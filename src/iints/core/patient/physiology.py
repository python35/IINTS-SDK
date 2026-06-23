"""Reusable physiology math helpers for patient models.

These helpers keep model assumptions explicit and testable. They are research
abstractions, not personalized clinical physiology estimators.
"""
from __future__ import annotations

import math


def smooth_threshold_excess(value: float, *, threshold: float, splay: float = 10.0) -> float:
    """Smooth positive excess above a biological threshold.

    Uses a numerically stable softplus. Below the threshold this approaches
    zero; above it approaches ``value - threshold``. The ``splay`` parameter
    controls the smooth transition width.
    """

    width = max(float(splay), 1e-6)
    z = (float(value) - float(threshold)) / width
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

    return max(0.0, float(gain)) * smooth_threshold_excess(
        glucose_mgdl,
        threshold=float(threshold_mgdl),
        splay=float(splay_mgdl),
    )
