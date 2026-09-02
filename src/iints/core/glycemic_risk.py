#!/usr/bin/env python3
"""Canonical blood-glucose risk-space transforms (Kovatchev symmetrization).

Single source of truth for LBGI, HBGI and BGRI. Both
:mod:`iints.core.clinical_metrics` and :mod:`iints.analysis.diabetes_metrics`
delegate here so the SDK cannot ship two disagreeing definitions of the same
published index.

Definition
----------
Kovatchev et al. symmetrize the (highly skewed) blood-glucose scale with

    f(BG) = 1.509 * (ln(BG)^1.084 - 5.381)          BG in mg/dL
    r(BG) = 10 * f(BG)^2

The factor 10 is not cosmetic: it is the scaling that makes r(BG) span
[0, 100] over the clinically meaningful range BG in [20, 600] mg/dL, which is
what every published LBGI/HBGI reference range assumes. Dropping it shifts
every value by an order of magnitude and therefore across the published risk
categories (see :func:`lbgi_risk_category`). ``test_glycemic_risk.py`` pins
r(20) = r(600) = 100 for exactly this reason.

LBGI averages r(BG) over the readings where f(BG) < 0 (hypoglycaemic side),
HBGI over those where f(BG) > 0, in both cases dividing by the total number of
valid readings - not by the number of readings on that side.

References
----------
Kovatchev BP, Cox DJ, Gonder-Frederick LA, Clarke WL. Symmetrization of the
blood glucose measurement scale and its applications. Diabetes Care.
1997;20(11):1655-1658.

Kovatchev BP, Cox DJ, Gonder-Frederick LA, Young-Hyman D, Schlundt D,
Clarke WL. Assessment of risk for severe hypoglycemia among adults with IDDM.
Diabetes Care. 1998;21(11):1870-1875.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import pandas as pd

__all__ = [
    "RISK_SCALE",
    "RISK_EXPONENT",
    "RISK_OFFSET",
    "RISK_AMPLITUDE",
    "EUGLYCEMIC_CENTER_MGDL",
    "bg_risk_transform",
    "bg_risk",
    "lbgi",
    "hbgi",
    "bgri",
    "lbgi_risk_category",
]

#: Kovatchev symmetrization constants, mg/dL scale.
RISK_SCALE = 1.509
RISK_EXPONENT = 1.084
RISK_OFFSET = 5.381

#: Amplitude that maps r(BG) onto [0, 100] over BG in [20, 600] mg/dL.
RISK_AMPLITUDE = 10.0

#: Glucose value at which f(BG) = 0, i.e. the split between the low and high
#: risk branches. Derived, not assumed: exp(RISK_OFFSET ** (1 / RISK_EXPONENT)).
EUGLYCEMIC_CENTER_MGDL = float(np.exp(RISK_OFFSET ** (1.0 / RISK_EXPONENT)))

GlucoseLike = Union[pd.Series, np.ndarray, Iterable[float]]


def _clean(glucose: GlucoseLike) -> np.ndarray:
    """Return finite, strictly positive glucose values as a float array.

    Non-positive and non-finite readings are dropped rather than clipped: the
    logarithm is undefined there, and silently substituting a value would
    invent data.
    """
    values = np.asarray(glucose, dtype=float).ravel()
    return values[np.isfinite(values) & (values > 0.0)]


def bg_risk_transform(glucose: GlucoseLike) -> np.ndarray:
    """Symmetrized glucose scale f(BG). Negative = hypo side, positive = hyper."""
    values = _clean(glucose)
    if values.size == 0:
        return np.empty(0, dtype=float)
    return RISK_SCALE * (np.log(values) ** RISK_EXPONENT - RISK_OFFSET)


def bg_risk(glucose: GlucoseLike) -> np.ndarray:
    """Risk function r(BG) = 10 * f(BG)^2, bounded to [0, 100] on [20, 600] mg/dL."""
    return RISK_AMPLITUDE * bg_risk_transform(glucose) ** 2


def _branch_mean(glucose: GlucoseLike, low_branch: bool) -> float:
    values = _clean(glucose)
    if values.size == 0:
        return 0.0
    transformed = RISK_SCALE * (np.log(values) ** RISK_EXPONENT - RISK_OFFSET)
    side = transformed < 0.0 if low_branch else transformed > 0.0
    risk = np.where(side, RISK_AMPLITUDE * transformed**2, 0.0)
    # Denominator is the full count of valid readings, per the published
    # definition - averaging over only one branch would inflate both indices.
    return float(risk.sum() / values.size)


def lbgi(glucose: GlucoseLike) -> float:
    """Low Blood Glucose Index. 0 when no reading falls on the hypo side."""
    return _branch_mean(glucose, low_branch=True)


def hbgi(glucose: GlucoseLike) -> float:
    """High Blood Glucose Index. 0 when no reading falls on the hyper side."""
    return _branch_mean(glucose, low_branch=False)


def bgri(glucose: GlucoseLike) -> float:
    """Blood Glucose Risk Index, the total risk LBGI + HBGI."""
    values = _clean(glucose)
    if values.size == 0:
        return 0.0
    return float(bg_risk(values).sum() / values.size)


def lbgi_risk_category(value: float) -> str:
    """Published LBGI bands for risk of severe hypoglycaemia (Kovatchev 1998).

    Returned as a label only; the bands are population-level descriptors and
    are not a clinical decision rule for an individual.
    """
    if value < 1.1:
        return "minimal"
    if value < 2.5:
        return "low"
    if value < 5.0:
        return "moderate"
    return "high"
