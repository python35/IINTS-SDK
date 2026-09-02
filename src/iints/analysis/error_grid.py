#!/usr/bin/env python3
"""Clarke Error Grid Analysis computed from paired measurements.

Zone percentages must be counted from real reference/prediction pairs. A grid
drawn from simulated scatter with the percentages written in as text is not an
error grid analysis - it is an illustration, and reporting it as a clinical
accuracy result is a fabrication.

This module therefore takes pairs and returns counts. There is no code path
that produces zone percentages without data.

Reference
---------
Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. Evaluating clinical
accuracy of systems for self-monitoring of blood glucose. Diabetes Care.
1987;10(5):622-628.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ZONES",
    "HAZARDOUS_ZONES",
    "ClarkeResult",
    "clarke_zones",
    "clarke_error_grid",
    "iso15197_agreement_rate",
]

ZONES = ("A", "B", "C", "D", "E")

#: Zones whose treatment consequence is erroneous or dangerous. Defined here,
#: beside the grid itself, so that callers computing a hazardous rate cannot
#: drift from :attr:`ClarkeResult.hazardous_pct` by keeping a private copy.
HAZARDOUS_ZONES = ("C", "D", "E")


@dataclass(frozen=True)
class ClarkeResult:
    """Zone counts and percentages for a set of paired measurements."""

    counts: Dict[str, int]
    percentages: Dict[str, float]
    n_pairs: int

    @property
    def clinically_acceptable_pct(self) -> float:
        """Zones A + B: no or benign treatment consequence."""
        return self.percentages["A"] + self.percentages["B"]

    @property
    def hazardous_pct(self) -> float:
        """Zones C + D + E: erroneous or dangerous treatment consequence."""
        return sum(self.percentages[z] for z in ("C", "D", "E"))

    def summary_line(self) -> str:
        return (f"n={self.n_pairs}  "
                + "  ".join(f"{z}={self.percentages[z]:.1f}%" for z in ZONES))


def clarke_zones(reference: ArrayLike,
                 predicted: ArrayLike) -> np.ndarray:
    """Classify each (reference, predicted) pair into Clarke zone A-E.

    Both arrays are in mg/dL. Pairs containing a non-finite or non-positive
    reference are dropped by :func:`clarke_error_grid` before classification;
    this function assumes already-clean input.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    pred = np.asarray(predicted, dtype=float).ravel()
    if ref.shape != pred.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {pred.shape}")

    zones = np.full(ref.shape, "B", dtype="<U1")

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.abs(pred - ref) / ref

    # Zone A: within 20% of the reference, or both in the hypo range where a
    # 20% band is not meaningful.
    zone_a = ((ref <= 70) & (pred <= 70)) | (rel_error <= 0.2)

    # Zone E: treatment would be the opposite of what is required.
    zone_e = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))

    # Zone C: would prompt over-correction of a value that needed none.
    zone_c = (((ref >= 70) & (ref <= 290) & (pred >= ref + 110))
              | ((ref >= 130) & (ref <= 180) & (pred <= (7.0 / 5.0) * ref - 182)))

    # Zone D: dangerous failure to detect a true excursion.
    zone_d = (((ref >= 240) & (pred >= 70) & (pred <= 180))
              | ((ref <= 175.0 / 3.0) & (pred >= 70) & (pred <= 180))
              | ((ref >= 175.0 / 3.0) & (ref <= 70) & (pred >= (6.0 / 5.0) * ref)))

    # Precedence matters: A first, then the hazard zones by severity.
    zones[zone_d] = "D"
    zones[zone_c] = "C"
    zones[zone_e] = "E"
    zones[zone_a] = "A"
    return zones


def clarke_error_grid(reference: ArrayLike,
                      predicted: ArrayLike) -> ClarkeResult:
    """Count Clarke zones over paired measurements.

    Raises
    ------
    ValueError
        If no valid pairs remain. Callers must supply real data; there is
        deliberately no synthetic fallback.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    pred = np.asarray(predicted, dtype=float).ravel()
    if ref.shape != pred.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {pred.shape}")

    valid = np.isfinite(ref) & np.isfinite(pred) & (ref > 0)
    ref, pred = ref[valid], pred[valid]
    if ref.size == 0:
        raise ValueError(
            "Clarke EGA requires at least one valid (reference, predicted) "
            "pair; none supplied."
        )

    zones = clarke_zones(ref, pred)
    counts = {z: int((zones == z).sum()) for z in ZONES}
    n = int(ref.size)
    percentages = {z: 100.0 * counts[z] / n for z in ZONES}
    return ClarkeResult(counts=counts, percentages=percentages, n_pairs=n)


def iso15197_agreement_rate(reference: ArrayLike,
                            measured: ArrayLike) -> float:
    """Percentage of pairs within ISO 15197:2013 system accuracy limits.

    The limits are +/-15 mg/dL below 100 mg/dL and +/-15% at or above it.

    This is a descriptive agreement rate only. ISO 15197 is a bench standard
    for blood-glucose meters evaluated against a laboratory reference method
    on real capillary samples; it does not apply to model predictions, and a
    high rate here is not certification or compliance. Do not label the output
    "ISO 15197 PASS".
    """
    ref = np.asarray(reference, dtype=float).ravel()
    meas = np.asarray(measured, dtype=float).ravel()
    valid = np.isfinite(ref) & np.isfinite(meas) & (ref > 0)
    ref, meas = ref[valid], meas[valid]
    if ref.size == 0:
        raise ValueError("no valid pairs supplied")
    diff = np.abs(meas - ref)
    within = np.where(ref < 100.0, diff <= 15.0, diff <= 0.15 * ref)
    return float(100.0 * within.mean())
