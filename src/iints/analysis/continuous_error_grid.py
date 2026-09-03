#!/usr/bin/env python3
"""Continuous Glucose-Error Grid Analysis (CG-EGA) from paired series.

CG-EGA judges a sensor or a predictive model on two axes at once: the
point accuracy of each reading (P-EGA) and the accuracy of the *rate* of
change it implies (R-EGA). Each pair gets a grade on both grids, and the
pair of grades is then combined -- separately per glycemic region -- into
one of accurate reading (AP), benign error (BE), or erroneous reading (EP).

Why this module exists next to :mod:`iints.analysis.error_grid`: the plain
Clarke grid has no notion of time, so a sensor that reads 120 mg/dL while
the reference is falling through 120 mg/dL scores identically to one that
reads 120 while the reference is flat. The second is accurate, the first
is dangerously late. Only the rate-aware grids separate them.

Scope of what is implemented here
---------------------------------
P-EGA (rate-expanded point grid), the rate classification it depends on,
the rate-deviation metric, and the published P-EGA x R-EGA combination
matrix are implemented and tested.

R-EGA zone *boundaries* are deliberately NOT implemented. The numeric
boundary lines of the rate grid are published only in closed-access
sources and in figures; they could not be obtained from an open source at
the time of writing. Guessing them would produce zone percentages that
look like a clinical accuracy result and are not one, which this package
does not do (see the module docstring of :mod:`iints.analysis.error_grid`).
:func:`rega_zones` therefore raises with an explicit reason, and
:func:`cgega` requires R-EGA grades to be supplied by the caller. Once the
boundaries are available, implement :func:`rega_zones` and nothing else in
this module needs to change.

References
----------
Kovatchev BP, Gonder-Frederick LA, Cox DJ, Clarke WL. Evaluating the
accuracy of continuous glucose-monitoring sensors: continuous
glucose-error grid analysis illustrated by TheraSense Freestyle Navigator
data. Diabetes Care. 2004;27(8):1922-1928.

Clarke WL, Kovatchev BP. Continuous glucose sensors: continuing questions
about clinical accuracy. J Diabetes Sci Technol. 2007;1(6):669-675.

Kovatchev BP, Clarke WL. Method, system and computer program product for
evaluating the accuracy of blood glucose monitoring sensors/devices.
US Patent 7,815,569 B2, 2010. -- source of the rate classification bounds
and of the P-EGA boundary expansion amounts encoded below.

De Bois M, El Yacoubi MA, Ammi M. Integration of clinical criteria into
the training of deep models: application to glucose prediction for
diabetic people. arXiv:2009.10514. -- source of the combination matrix
transcribed in :data:`CGEGA_MATRIX`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .error_grid import clarke_zones

__all__ = [
    "CGEGA_MATRIX",
    "CGEGA_LABELS",
    "GLYCEMIC_REGIONS",
    "HYPO_CUTOFF_MGDL",
    "HYPER_CUTOFF_MGDL",
    "PEGA_EXPANSION_MGDL",
    "RATE_CLASSES",
    "RATE_MODERATE_MGDL_MIN",
    "RATE_RAPID_MGDL_MIN",
    "REGA_ZONES",
    "CGEGAResult",
    "cgega",
    "classify_rate",
    "combine_cgega",
    "glycemic_region",
    "pega_zones",
    "rate_deviation",
    "rate_of_change",
    "rega_zones",
]

# --------------------------------------------------------------------------
# Glycemic regions. CG-EGA is reported per region because the clinical
# consequence of the same error differs between them; a single pooled
# percentage hides exactly the cases that matter.
# --------------------------------------------------------------------------

#: Below this reference value the pair is scored in the hypoglycemia region.
HYPO_CUTOFF_MGDL = 70.0

#: Above this reference value the pair is scored in the hyperglycemia region.
HYPER_CUTOFF_MGDL = 180.0

GLYCEMIC_REGIONS = ("hypoglycemia", "euglycemia", "hyperglycemia")

# --------------------------------------------------------------------------
# Rate classification. Bounds are on the *reference* rate of change: the
# expansion answers "how far could the reference have moved while this
# reading was being produced", which is a property of the reference, not of
# the device under test.
# --------------------------------------------------------------------------

#: |rate| at or below this is a slow change; above it, a moderate one.
RATE_MODERATE_MGDL_MIN = 1.0

#: |rate| above this is a rapid change.
RATE_RAPID_MGDL_MIN = 2.0

RATE_CLASSES = (
    "rapid_fall",
    "moderate_fall",
    "slow",
    "moderate_rise",
    "rapid_rise",
)

#: mg/dL by which the relevant P-EGA boundaries move, per rate magnitude.
PEGA_EXPANSION_MGDL: Mapping[str, float] = {
    "slow": 0.0,
    "moderate": 10.0,
    "rapid": 20.0,
}

REGA_ZONES = ("A", "B", "uC", "lC", "uD", "lD", "uE", "lE")

CGEGA_LABELS = ("AP", "BE", "EP")

# --------------------------------------------------------------------------
# The published combination matrix, transcribed cell by cell.
#
# Read as CGEGA_MATRIX[region][r_zone][p_zone]. Only the (region, p_zone)
# combinations the source defines are present; a combination that is absent
# raises in combine_cgega() rather than falling back to a default, so that
# an unexpected zone becomes visible instead of being scored silently.
# --------------------------------------------------------------------------

_HYPO_COLUMNS = ("A", "D", "E")
_EU_COLUMNS = ("A", "B", "C")
_HYPER_COLUMNS = ("A", "B", "C", "D", "E")

_HYPO_ROWS = {
    "A": ("AP", "EP", "EP"),
    "B": ("AP", "EP", "EP"),
    "uC": ("BE", "EP", "EP"),
    "lC": ("BE", "EP", "EP"),
    "uD": ("EP", "EP", "EP"),
    "lD": ("BE", "EP", "EP"),
    "uE": ("EP", "EP", "EP"),
    "lE": ("BE", "EP", "EP"),
}

_EU_ROWS = {
    "A": ("AP", "AP", "EP"),
    "B": ("AP", "AP", "EP"),
    "uC": ("BE", "BE", "EP"),
    "lC": ("BE", "BE", "EP"),
    "uD": ("BE", "BE", "EP"),
    "lD": ("BE", "BE", "EP"),
    "uE": ("EP", "EP", "EP"),
    "lE": ("EP", "EP", "EP"),
}

_HYPER_ROWS = {
    "A": ("AP", "AP", "EP", "EP", "EP"),
    "B": ("AP", "AP", "EP", "EP", "EP"),
    "uC": ("BE", "BE", "EP", "EP", "EP"),
    "lC": ("BE", "BE", "EP", "EP", "EP"),
    "uD": ("BE", "BE", "EP", "EP", "EP"),
    "lD": ("EP", "EP", "EP", "EP", "EP"),
    "uE": ("EP", "EP", "EP", "EP", "EP"),
    "lE": ("EP", "EP", "EP", "EP", "EP"),
}


def _build_matrix() -> Dict[str, Dict[str, Dict[str, str]]]:
    matrix: Dict[str, Dict[str, Dict[str, str]]] = {}
    for region, columns, rows in (
        ("hypoglycemia", _HYPO_COLUMNS, _HYPO_ROWS),
        ("euglycemia", _EU_COLUMNS, _EU_ROWS),
        ("hyperglycemia", _HYPER_COLUMNS, _HYPER_ROWS),
    ):
        if set(rows) != set(REGA_ZONES):
            raise AssertionError(f"{region}: R-EGA rows incomplete")
        region_map: Dict[str, Dict[str, str]] = {}
        for r_zone, cells in rows.items():
            if len(cells) != len(columns):
                raise AssertionError(
                    f"{region}/{r_zone}: {len(cells)} cells for "
                    f"{len(columns)} P-EGA columns"
                )
            for cell in cells:
                if cell not in CGEGA_LABELS:
                    raise AssertionError(f"{region}/{r_zone}: bad cell {cell!r}")
            region_map[r_zone] = dict(zip(columns, cells))
        matrix[region] = region_map
    return matrix


#: Published CG-EGA combination matrix, indexed [region][r_zone][p_zone].
CGEGA_MATRIX = _build_matrix()


@dataclass(frozen=True)
class CGEGAResult:
    """CG-EGA counts and percentages, overall and per glycemic region.

    ``by_region`` percentages are within-region: they answer "of the pairs
    whose reference fell in this region, what fraction were erroneous",
    which is how CG-EGA is reported. They do not sum to 100 across regions.
    """

    counts: Dict[str, int]
    percentages: Dict[str, float]
    n_pairs: int
    by_region_counts: Dict[str, Dict[str, int]]
    by_region_percentages: Dict[str, Dict[str, float]]
    n_by_region: Dict[str, int]

    @property
    def erroneous_pct(self) -> float:
        """Percentage of pairs scored as an erroneous reading."""
        return self.percentages["EP"]

    def summary_line(self) -> str:
        parts = [f"n={self.n_pairs}"]
        parts += [f"{label}={self.percentages[label]:.1f}%" for label in CGEGA_LABELS]
        return "  ".join(parts)

    def region_line(self, region: str) -> str:
        if region not in GLYCEMIC_REGIONS:
            raise ValueError(f"unknown region {region!r}")
        n = self.n_by_region[region]
        if n == 0:
            return f"{region}: no pairs"
        pct = self.by_region_percentages[region]
        body = "  ".join(f"{label}={pct[label]:.1f}%" for label in CGEGA_LABELS)
        return f"{region}: n={n}  {body}"


def glycemic_region(reference: ArrayLike) -> np.ndarray:
    """Assign each reference value to its glycemic region."""
    ref = np.asarray(reference, dtype=float).ravel()
    out = np.full(ref.shape, "euglycemia", dtype="<U13")
    out[ref < HYPO_CUTOFF_MGDL] = "hypoglycemia"
    out[ref > HYPER_CUTOFF_MGDL] = "hyperglycemia"
    return out


def rate_of_change(values: ArrayLike, minutes: ArrayLike) -> np.ndarray:
    """Rate of change in mg/dL/min between consecutive samples.

    The result is aligned with ``values``: element *i* is the rate over the
    interval ending at sample *i*. Element 0 has no preceding sample and is
    NaN -- it is not zero, because "no rate could be computed" and "the
    glucose was flat" are different statements and only one of them is true.

    ``minutes`` must be strictly increasing.
    """
    vals = np.asarray(values, dtype=float).ravel()
    mins = np.asarray(minutes, dtype=float).ravel()
    if vals.shape != mins.shape:
        raise ValueError(f"shape mismatch: {vals.shape} vs {mins.shape}")
    if vals.size < 2:
        raise ValueError(
            "a rate of change needs at least two samples; "
            f"{vals.size} supplied"
        )
    dt = np.diff(mins)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("minutes must be finite and strictly increasing")
    out = np.full(vals.shape, np.nan, dtype=float)
    out[1:] = np.diff(vals) / dt
    return out


def classify_rate(rate: ArrayLike) -> np.ndarray:
    """Classify rates of change into the five CG-EGA rate classes.

    A non-finite rate -- the first sample of a series, for instance -- is
    classified ``"slow"``, which applies no boundary expansion and so
    reduces P-EGA to the plain Clarke grid for that pair. That is the
    conservative direction: it never excuses an error on the strength of a
    rate that was not measured.
    """
    r = np.asarray(rate, dtype=float).ravel()
    out = np.full(r.shape, "slow", dtype="<U13")
    finite = np.isfinite(r)
    mag = np.abs(r)
    moderate = finite & (mag > RATE_MODERATE_MGDL_MIN) & (mag <= RATE_RAPID_MGDL_MIN)
    rapid = finite & (mag > RATE_RAPID_MGDL_MIN)
    out[moderate & (r < 0)] = "moderate_fall"
    out[moderate & (r > 0)] = "moderate_rise"
    out[rapid & (r < 0)] = "rapid_fall"
    out[rapid & (r > 0)] = "rapid_rise"
    return out


def rate_deviation(reference_rate: ArrayLike,
                   estimated_rate: ArrayLike) -> np.ndarray:
    """Rate deviation (RD) in mg/dL/min: reference rate minus estimated rate.

    RD is to the rate what the plain error is to the reading, so its mean is
    a bias on the rate axis and its magnitude is a rate error.

    Note on the definition: the review that introduces RD prints it as a
    difference of rates *divided by the time interval*, while giving its
    unit as mg/dL/min. Those two statements are inconsistent -- dividing a
    rate by a time yields mg/dL/min^2. This implementation follows the
    stated unit and returns the difference of rates, so that ``mean(RD)``
    is comparable to a mean error as the source describes it.
    """
    ref = np.asarray(reference_rate, dtype=float).ravel()
    est = np.asarray(estimated_rate, dtype=float).ravel()
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {est.shape}")
    return ref - est


def pega_zones(reference: ArrayLike,
               predicted: ArrayLike,
               reference_rate: ArrayLike) -> np.ndarray:
    """Classify pairs into rate-expanded point-EGA zones A-E.

    P-EGA is the Clarke grid with its boundaries moved outward according to
    how fast the *reference* was changing, because a reading that lags a
    fast excursion is a smaller clinical error than the same reading taken
    while glucose was flat:

    * reference falling at a moderate rate -- the upper limits of the upper
      A, B and D zones move up by 10 mg/dL; falling rapidly -- by 20 mg/dL.
    * reference rising at a moderate rate -- the lower limits of the lower
      A, B and D zones move down by 10 mg/dL; rising rapidly -- by 20 mg/dL.
    * reference changing slowly -- the zones are the plain Clarke zones.

    Implementation note: moving every upper boundary outward by the same
    amount is a translation of the whole upper boundary set, so classifying
    a point against the expanded grid is the same as classifying the point
    moved *toward* the diagonal by that amount against the unexpanded grid.
    That is what this function does, which keeps a single definition of the
    zone geometry in :func:`~iints.analysis.error_grid.clarke_zones` instead
    of a second copy that could drift from it. The shift is clamped at the
    diagonal so that expansion can never carry a reading across the
    reference and into a hazard zone on the far side.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    pred = np.asarray(predicted, dtype=float).ravel()
    rate = np.asarray(reference_rate, dtype=float).ravel()
    if not (ref.shape == pred.shape == rate.shape):
        raise ValueError(
            f"shape mismatch: reference {ref.shape}, predicted "
            f"{pred.shape}, reference_rate {rate.shape}"
        )

    classes = classify_rate(rate)
    delta = np.zeros(ref.shape, dtype=float)
    delta[np.isin(classes, ("moderate_fall", "moderate_rise"))] = \
        PEGA_EXPANSION_MGDL["moderate"]
    delta[np.isin(classes, ("rapid_fall", "rapid_rise"))] = \
        PEGA_EXPANSION_MGDL["rapid"]

    falling = np.isin(classes, ("moderate_fall", "rapid_fall"))
    rising = np.isin(classes, ("moderate_rise", "rapid_rise"))

    effective = pred.copy()
    # Falling reference excuses reading high: pull high readings down.
    high = falling & (pred > ref)
    effective[high] = np.maximum(ref[high], pred[high] - delta[high])
    # Rising reference excuses reading low: pull low readings up.
    low = rising & (pred < ref)
    effective[low] = np.minimum(ref[low], pred[low] + delta[low])

    return clarke_zones(ref, effective)


def rega_zones(reference_rate: ArrayLike,
               estimated_rate: ArrayLike) -> np.ndarray:
    """Classify rate pairs into R-EGA zones. Not implemented.

    The rate grid's numeric boundary lines are the one part of CG-EGA that
    could not be obtained from an accessible source: the methodology
    publication that defines them is closed access, and the papers that use
    R-EGA show the grid as a figure without tabulating its boundaries. The
    axes (reference rate against estimated rate, both mg/dL/min, plotted
    over -4 to 4) and the zone names (:data:`REGA_ZONES`) are established;
    the lines between them are not.

    Supplying invented boundaries here would make :func:`cgega` return zone
    percentages that read as a clinical accuracy result while resting on
    guessed geometry, so this raises instead. To finish CG-EGA, implement
    this function from the boundary definitions and pass nothing else --
    :func:`cgega` will then be able to compute its own R-EGA grades.
    """
    raise NotImplementedError(
        "R-EGA zone boundaries are not available in this package. The "
        "numeric boundary lines are published only in closed-access "
        "sources (Kovatchev et al., Diabetes Care 2004;27:1922-1928 and "
        "the CG-EGA methodology review) and in figures. Pass R-EGA grades "
        "explicitly to cgega(r_zones=...), or implement this function once "
        "the boundary definitions are at hand. Grades must be drawn from "
        f"{REGA_ZONES}."
    )


def combine_cgega(p_zone: str, r_zone: str, region: str) -> str:
    """Combine a P-EGA and an R-EGA grade into AP, BE or EP for one region."""
    if region not in CGEGA_MATRIX:
        raise ValueError(
            f"unknown glycemic region {region!r}; expected one of "
            f"{GLYCEMIC_REGIONS}"
        )
    rows = CGEGA_MATRIX[region]
    if r_zone not in rows:
        raise ValueError(
            f"unknown R-EGA zone {r_zone!r}; expected one of {REGA_ZONES}"
        )
    row = rows[r_zone]
    if p_zone not in row:
        raise ValueError(
            f"the published CG-EGA matrix defines no cell for P-EGA zone "
            f"{p_zone!r} in the {region} region (defined: "
            f"{tuple(row)}). This combination is not scored rather than "
            f"being given a default."
        )
    return row[p_zone]


def cgega(reference: ArrayLike,
          predicted: ArrayLike,
          minutes: ArrayLike,
          r_zones: ArrayLike | None = None) -> CGEGAResult:
    """Run CG-EGA over paired series and count AP / BE / EP per region.

    Parameters
    ----------
    reference, predicted
        Paired glucose values in mg/dL, in time order.
    minutes
        Sample times in minutes, strictly increasing, same length as the
        series. Used to derive the reference rate of change that P-EGA needs.
    r_zones
        R-EGA grade per pair, from :data:`REGA_ZONES`. Required, because
        :func:`rega_zones` cannot derive them (see its docstring). The first
        sample has no rate and is dropped, so ``r_zones`` may be given
        either for every sample or for the samples from the second onward.

    Raises
    ------
    ValueError
        If no valid pairs remain, or if the inputs disagree in length. There
        is deliberately no synthetic fallback.
    NotImplementedError
        If ``r_zones`` is omitted.
    """
    # Annotated as the shape-erased NDArray, not whatever precise shape mypy
    # would infer from .ravel() alone: the reassignments below (boolean-mask
    # and slice indexing) return that same broader type, and mypy otherwise
    # flags them as incompatible with a narrower inferred type.
    ref: NDArray[np.float64] = np.asarray(reference, dtype=float).ravel()
    pred: NDArray[np.float64] = np.asarray(predicted, dtype=float).ravel()
    mins: NDArray[np.float64] = np.asarray(minutes, dtype=float).ravel()
    if not (ref.shape == pred.shape == mins.shape):
        raise ValueError(
            f"shape mismatch: reference {ref.shape}, predicted "
            f"{pred.shape}, minutes {mins.shape}"
        )

    rate = rate_of_change(ref, mins)

    if r_zones is None:
        rega_zones(rate, rate)  # raises with the explanation
    r: NDArray[np.str_] = np.asarray(r_zones, dtype=str).ravel()
    if r.size == ref.size:
        r = r[1:]
    elif r.size != ref.size - 1:
        raise ValueError(
            f"r_zones has {r.size} entries; expected {ref.size} (one per "
            f"sample) or {ref.size - 1} (one per rate)"
        )
    unknown = sorted(set(r.tolist()) - set(REGA_ZONES))
    if unknown:
        raise ValueError(f"unknown R-EGA zones {unknown}; expected {REGA_ZONES}")

    # Drop the first sample: it has no rate, so it has no R-EGA grade.
    ref, pred, rate = ref[1:], pred[1:], rate[1:]

    valid = np.isfinite(ref) & np.isfinite(pred) & (ref > 0)
    ref, pred, rate, r = ref[valid], pred[valid], rate[valid], r[valid]
    if ref.size == 0:
        raise ValueError(
            "CG-EGA requires at least one valid (reference, predicted) pair "
            "with a computable rate of change; none supplied."
        )

    p = pega_zones(ref, pred, rate)
    regions = glycemic_region(ref)

    labels = np.array(
        [combine_cgega(pz, rz, reg) for pz, rz, reg in zip(p, r, regions)],
        dtype="<U2",
    )

    n = int(labels.size)
    counts = {label: int((labels == label).sum()) for label in CGEGA_LABELS}
    percentages = {label: 100.0 * counts[label] / n for label in CGEGA_LABELS}

    by_region_counts: Dict[str, Dict[str, int]] = {}
    by_region_percentages: Dict[str, Dict[str, float]] = {}
    n_by_region: Dict[str, int] = {}
    for region in GLYCEMIC_REGIONS:
        mask = regions == region
        n_r = int(mask.sum())
        n_by_region[region] = n_r
        region_counts = {
            label: int((labels[mask] == label).sum()) for label in CGEGA_LABELS
        }
        by_region_counts[region] = region_counts
        by_region_percentages[region] = {
            label: (100.0 * region_counts[label] / n_r) if n_r else 0.0
            for label in CGEGA_LABELS
        }

    return CGEGAResult(
        counts=counts,
        percentages=percentages,
        n_pairs=n,
        by_region_counts=by_region_counts,
        by_region_percentages=by_region_percentages,
        n_by_region=n_by_region,
    )
