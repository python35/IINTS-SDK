"""Two-compartment C-peptide kinetics for research graft simulations.

Why this module exists
----------------------
Graft function is not measured in insulin units. Exogenous insulin makes plasma
insulin useless as a read-out, so clinical islet and stem-cell-islet studies
report **C-peptide**: fasting, stimulated, or as an area under the curve from a
mixed-meal or intravenous glucose tolerance test. C-peptide is co-secreted with
insulin in equimolar amounts and is cleared with near-constant kinetics, which
is exactly why it is the endpoint of choice.

Exposing C-peptide therefore turns a graft simulation from an internal
abstraction into something comparable with published endpoints, without
claiming any clinical validity for the graft model itself.

Model
-----
The standard two-compartment representation (Eaton et al. 1980), parameterised
the way Van Cauter et al. (1992, DOI 10.2337/diab.41.3.368) standardised it:

    dC1/dt = S(t) / V  - (k1 + k3) * C1 + k2 * C2
    dC2/dt =              k1 * C1       - k2 * C2

with C1 the plasma (central) concentration in pmol/L, C2 the peripheral
compartment, S the secretion rate in pmol/min and V the central distribution
volume in L. The three rate constants follow from a bi-exponential decay with a
short half-life, a long half-life, and the fraction of the decay amplitude
attributable to the short component:

    a = ln2 / short_half_life,  b = ln2 / long_half_life
    k2 = F*b + (1-F)*a,  k3 = a*b / k2,  k1 = a + b - k2 - k3

``verify_half_lives()`` recovers the half-lives from the eigenvalues of the
resulting system, so the conversion is checked rather than trusted.

Parameter provenance
--------------------
* central volume ``V = 1.92 * BSA + 0.64`` L and
  ``long_half_life = 0.14 * age + 29.2`` min are the Van Cauter regressions;
* ``short_half_life`` uses their class values: 4.95 min for BMI <= 30,
  4.55 min for BMI > 30, 4.52 min for type 2 diabetes;
* ``short_fraction`` is **not yet sourced** in this implementation. It is the
  third degree of freedom of any two-compartment fit and it shapes the early
  decay. The default below is a commonly used adult value, but until it is
  checked against the original table, treat **absolute** C-peptide magnitudes
  as provisional. Relative comparisons at a fixed parameter set - primary
  versus stem-cell-derived islets, or the same graft over time - do not depend
  on it nearly as strongly. This is recorded in docs/DEFECT_REGISTER_NL.md.

Research and education only. No clinical interpretation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple, cast

import numpy as np

from iints.research.provenance import warn_unsourced

__all__ = [
    "CPEPTIDE_PMOL_PER_INSULIN_UNIT",
    "CPeptideModel",
    "CPeptideParameters",
    "body_surface_area_m2",
]

# 1 U of insulin is 0.0347 mg; with a molar mass of 5808 g/mol that is
# 5.97 nmol, i.e. ~6000 pmol. C-peptide is secreted equimolar with insulin.
CPEPTIDE_PMOL_PER_INSULIN_UNIT = 6000.0

# Unsourced third degree of freedom; see "Parameter provenance" above.
DEFAULT_SHORT_FRACTION = 0.76


def body_surface_area_m2(body_weight_kg: float, height_cm: float) -> float:
    """Du Bois body surface area, the input to the Van Cauter volume regression."""
    if body_weight_kg <= 0.0 or height_cm <= 0.0:
        raise ValueError("body weight and height must be positive")
    return 0.007184 * (height_cm ** 0.725) * (body_weight_kg ** 0.425)


@dataclass(frozen=True)
class CPeptideParameters:
    """Subject-level C-peptide kinetics. Unset fields use the regressions."""

    body_weight_kg: float = 70.0
    height_cm: float = 170.0
    age_years: float = 35.0
    type_2_diabetes: bool = False
    short_half_life_minutes: Optional[float] = None
    long_half_life_minutes: Optional[float] = None
    short_fraction: float = DEFAULT_SHORT_FRACTION
    distribution_volume_l: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("body_weight_kg", "height_cm", "age_years"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < float(self.short_fraction) < 1.0:
            raise ValueError("short_fraction must lie strictly between 0 and 1")
        for name in ("short_half_life_minutes", "long_half_life_minutes",
                     "distribution_volume_l"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive when given")

    # ---- derived quantities -------------------------------------------------
    @property
    def body_mass_index(self) -> float:
        return float(self.body_weight_kg) / ((float(self.height_cm) / 100.0) ** 2)

    @property
    def effective_short_half_life_minutes(self) -> float:
        if self.short_half_life_minutes is not None:
            return float(self.short_half_life_minutes)
        if self.type_2_diabetes:
            return 4.52
        return 4.55 if self.body_mass_index > 30.0 else 4.95

    @property
    def effective_long_half_life_minutes(self) -> float:
        if self.long_half_life_minutes is not None:
            return float(self.long_half_life_minutes)
        return 0.14 * float(self.age_years) + 29.2

    @property
    def effective_distribution_volume_l(self) -> float:
        if self.distribution_volume_l is not None:
            return float(self.distribution_volume_l)
        bsa = body_surface_area_m2(self.body_weight_kg, self.height_cm)
        return 1.92 * bsa + 0.64

    @property
    def rate_constants(self) -> Tuple[float, float, float]:
        """(k1, k2, k3) in 1/min for the two-compartment system."""
        fast = math.log(2.0) / self.effective_short_half_life_minutes
        slow = math.log(2.0) / self.effective_long_half_life_minutes
        if slow >= fast:
            raise ValueError(
                "the long half-life must exceed the short half-life")
        fraction = float(self.short_fraction)
        k2 = fraction * slow + (1.0 - fraction) * fast
        k3 = fast * slow / k2
        k1 = fast + slow - k2 - k3
        if min(k1, k2, k3) <= 0.0:
            raise ValueError(
                "half-life and fraction combination gives a non-physical "
                f"parameterisation (k1={k1:.5g}, k2={k2:.5g}, k3={k3:.5g})")
        return k1, k2, k3

    def with_subject(self, **changes: float) -> "CPeptideParameters":
        return replace(self, **cast(Any, changes))


class CPeptideModel:
    """Plasma C-peptide driven by an insulin-equivalent secretion rate."""

    def __init__(self, parameters: Optional[CPeptideParameters] = None) -> None:
        self.parameters = parameters or CPeptideParameters()
        self.central_pmol_l = 0.0
        self.peripheral_pmol_l = 0.0
        if self.parameters.short_fraction == DEFAULT_SHORT_FRACTION:
            # Absolute concentrations rest on this split; relative
            # comparisons at a fixed parameter set do not.
            warn_unsourced("DEFAULT_SHORT_FRACTION")
        self._k1, self._k2, self._k3 = self.parameters.rate_constants
        self._volume_l = self.parameters.effective_distribution_volume_l

    def reset(self, central_pmol_l: float = 0.0,
              peripheral_pmol_l: float = 0.0) -> None:
        if central_pmol_l < 0.0 or peripheral_pmol_l < 0.0:
            raise ValueError("C-peptide concentrations must be non-negative")
        self.central_pmol_l = float(central_pmol_l)
        self.peripheral_pmol_l = float(peripheral_pmol_l)

    def verify_half_lives(self) -> Tuple[float, float]:
        """Recover (short, long) half-life in minutes from the system eigenvalues.

        A self-check on the parameter conversion: this must reproduce the
        requested half-lives, otherwise the rate constants are wrong.
        """
        matrix = np.array([[-(self._k1 + self._k3), self._k2],
                           [self._k1, -self._k2]], dtype=float)
        eigenvalues = np.linalg.eigvals(matrix).real
        rates = np.sort(-eigenvalues)[::-1]          # fast first
        return (float(math.log(2.0) / rates[0]), float(math.log(2.0) / rates[1]))

    def step(self, secretion_units: float, dt_minutes: float,
             substeps: Optional[int] = None) -> float:
        """Advance by ``dt_minutes`` given insulin-equivalent secretion in units.

        The secretion is treated as a constant rate across the interval. The
        fast compartment has a ~5 min half-life, so the integration is
        sub-stepped to keep the explicit scheme accurate at a 5- or 15-minute
        caller step; the default keeps internal steps at or below 0.5 min.
        """
        if not np.isfinite(secretion_units) or secretion_units < 0.0:
            raise ValueError("secretion_units must be finite and non-negative")
        if not np.isfinite(dt_minutes) or dt_minutes <= 0.0:
            raise ValueError("dt_minutes must be finite and positive")
        steps = int(substeps) if substeps else max(1, int(math.ceil(dt_minutes / 0.5)))
        step_minutes = float(dt_minutes) / steps
        rate_pmol_per_min = (float(secretion_units) * CPEPTIDE_PMOL_PER_INSULIN_UNIT
                             / float(dt_minutes))
        for _ in range(steps):
            central, peripheral = self.central_pmol_l, self.peripheral_pmol_l
            d_central = (rate_pmol_per_min / self._volume_l
                         - (self._k1 + self._k3) * central + self._k2 * peripheral)
            d_peripheral = self._k1 * central - self._k2 * peripheral
            self.central_pmol_l = max(0.0, central + step_minutes * d_central)
            self.peripheral_pmol_l = max(0.0, peripheral + step_minutes * d_peripheral)
        if not np.isfinite(self.central_pmol_l):
            raise RuntimeError("C-peptide integration produced a non-finite value")
        return self.central_pmol_l

    @property
    def plasma_pmol_l(self) -> float:
        return float(self.central_pmol_l)

    @property
    def plasma_nmol_l(self) -> float:
        return float(self.central_pmol_l) / 1000.0

    def steady_state_pmol_l(self, secretion_units_per_min: float) -> float:
        """Plasma concentration a constant secretion rate converges to."""
        if secretion_units_per_min < 0.0:
            raise ValueError("secretion_units_per_min must be non-negative")
        rate = secretion_units_per_min * CPEPTIDE_PMOL_PER_INSULIN_UNIT
        # At steady state k1*C1 = k2*C2, so the central balance reduces to
        # S/V = k3*C1.
        return float(rate / (self._volume_l * self._k3))

    def get_state(self) -> dict:
        return {
            "c_peptide_central_pmol_l": float(self.central_pmol_l),
            "c_peptide_peripheral_pmol_l": float(self.peripheral_pmol_l),
            "c_peptide_volume_l": float(self._volume_l),
            "c_peptide_k1_per_min": float(self._k1),
            "c_peptide_k2_per_min": float(self._k2),
            "c_peptide_k3_per_min": float(self._k3),
        }
