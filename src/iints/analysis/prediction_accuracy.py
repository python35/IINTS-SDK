"""Rate-aware accuracy analysis for glucose *predictions*.

Why this module exists
----------------------
The Clarke error grid (Clarke et al., Diabetes Care 1987) was designed to grade
**measurement** error: a meter reading against a reference value taken at the
same instant. ``iints.analysis.error_grid`` implements it faithfully, and it is
the right tool for that job.

It is *not* sufficient for grading a forecast. A prediction that is numerically
close to the reference but points the wrong way is clinically dangerous in a way
Clarke cannot see by construction, because Clarke only ever looks at one pair of
scalars. Example: reference 95 mg/dL and falling 2.5 mg/dL/min (heading for a
severe hypo within minutes) versus a prediction of 105 mg/dL and rising. Both
values sit deep in Clarke zone A; the clinical implication is inverted.

Design principle taken from the literature
------------------------------------------
Sivananthan et al., "Assessment of Blood Glucose Predictors: The
Prediction-Error Grid Analysis", Diabetes Technol Ther 2011
(doi:10.1089/dia.2011.0033) established the load-bearing insight: the
continuous glucose-error grid analysis (CG-EGA, Kovatchev et al., Diabetes Care
2004, doi:10.2337/diacare.27.8.1922) estimates the rate of change from *past*
readings, which is wrong for a predictor, because a predictor produces an
estimate ahead of time. Their PRED-EGA estimates the rate of change **on the
predicted profile itself**. That principle is what this module implements: the
reference rate comes from the reference trajectory and the predicted rate comes
from the predicted trajectory, never from history.

HONESTY NOTE — READ BEFORE RENAMING ANYTHING HERE
-------------------------------------------------
This module is an *independent implementation of that principle*. It is
deliberately **not** called PRED-EGA and it does not claim to reproduce that
paper's numbers. The published zone tables are behind a paywall and were not
available when this was written, so the category boundaries below are defined
locally, in this file, in explicit terms. Re-deriving a named method's internals
from memory and publishing the result under that name would misrepresent both
the method and this project.

Therefore:

* Do not rename ``directional_report`` to PRED-EGA, CG-EGA or P-EGA unless you
  have the paper in front of you and have verified this implementation against
  the zone tables printed in it.
* Report Clarke as the secondary, comparable-to-the-literature metric and this
  analysis as the primary safety-relevant one, and say which is which.
* Every threshold in this file is a named module constant, not a literal buried
  in a branch, so that a reviewer can audit the definition in one place.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numpy.typing import ArrayLike

from .error_grid import clarke_zones

__all__ = [
    "RATE_BIN_EDGES",
    "RATE_BIN_LABELS",
    "HYPO_THRESHOLD_MGDL",
    "HYPER_THRESHOLD_MGDL",
    "TREND_REVERSAL_MGDL_MIN",
    "RATE_WINDOW_MINUTES",
    "ACCEPTABLE_CLARKE_ZONES",
    "rate_bin",
    "profile_rate_of_change",
    "glycemic_range",
    "classify_predictions",
    "directional_report",
]

# --- Definitions. Every one of these is a deliberate, citable choice. ---------

#: Rate-of-change categories in mg/dL/min. These are the conventional CGM trend
#: bands used on pump and CGM displays (the single/double arrow convention):
#: falling fast, falling, steady, rising, rising fast.
RATE_BIN_EDGES = (-2.0, -1.0, 1.0, 2.0)
RATE_BIN_LABELS = ("falling_fast", "falling", "steady", "rising", "rising_fast")

#: Glycemic ranges, on the *reference* value. Consensus thresholds
#: (Battelino et al., Diabetes Care 2019, time-in-range consensus).
HYPO_THRESHOLD_MGDL = 70.0
HYPER_THRESHOLD_MGDL = 180.0

#: A predicted trend is treated as *reversed* only when both the reference and
#: the prediction move at least this fast in opposite directions. Below this
#: speed the sign of the rate is noise rather than a clinical trend, so
#: penalising a sign flip there would manufacture failures.
TREND_REVERSAL_MGDL_MIN = 1.0

#: Rate is estimated over the final part of each trajectory. 15 minutes is the
#: interval over which CGM trend arrows are conventionally computed, and it is
#: three samples at a 5-minute cadence, which damps single-sample noise.
RATE_WINDOW_MINUTES = 15.0

#: Clarke zones that do not lead to a harmful treatment decision.
ACCEPTABLE_CLARKE_ZONES = frozenset({"A", "B"})


def rate_bin(rate_mgdl_min: ArrayLike) -> np.ndarray:
    """Map a rate of change in mg/dL/min onto :data:`RATE_BIN_LABELS`.

    Bin index is monotone in the rate, so ``|bin(a) - bin(b)|`` is a meaningful
    distance between two trends.
    """
    rates = np.asarray(rate_mgdl_min, dtype=float)
    return np.digitize(rates, RATE_BIN_EDGES, right=False)


def profile_rate_of_change(profile: np.ndarray,
                           step_minutes: float,
                           window_minutes: float = RATE_WINDOW_MINUTES) -> np.ndarray:
    """Rate of change (mg/dL/min) at the END of each trajectory in ``profile``.

    ``profile`` is ``[n_samples, n_steps]``. The rate is a finite difference
    across the last ``window_minutes`` of the trajectory.

    This function is the whole point of the module: call it on the *predicted*
    trajectory to get the predicted rate, and on the *reference* trajectory to
    get the reference rate. Never substitute a rate computed from pre-forecast
    history for the predicted rate — that is precisely the misapplication that
    Sivananthan et al. (2011) identified.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"profile must be 2-D [n_samples, n_steps], got {arr.shape}")
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")

    lag = int(round(window_minutes / step_minutes))
    if lag < 1:
        raise ValueError(
            f"window_minutes={window_minutes} is shorter than one step of "
            f"{step_minutes} min; cannot estimate a rate"
        )
    if arr.shape[1] <= lag:
        raise ValueError(
            f"trajectory has {arr.shape[1]} steps but a {window_minutes}-min window "
            f"needs at least {lag + 1}; shorten window_minutes or lengthen the horizon"
        )
    span = lag * step_minutes
    return (arr[:, -1] - arr[:, -1 - lag]) / span


def glycemic_range(reference: ArrayLike) -> np.ndarray:
    """Label each reference value ``hypo`` / ``target`` / ``hyper``."""
    ref = np.asarray(reference, dtype=float)
    out = np.full(ref.shape, "target", dtype=object)
    out[ref < HYPO_THRESHOLD_MGDL] = "hypo"
    out[ref > HYPER_THRESHOLD_MGDL] = "hyper"
    return out


def classify_predictions(reference_profile: np.ndarray,
                         predicted_profile: np.ndarray,
                         step_minutes: float,
                         window_minutes: float = RATE_WINDOW_MINUTES) -> Dict[str, np.ndarray]:
    """Classify each forecast as ``accurate`` / ``benign`` / ``erroneous``.

    The rules, stated in full so a reviewer never has to read the code:

    ``erroneous``
        The point pair falls outside Clarke zones A/B (it would drive a harmful
        treatment decision), OR the predicted trend is reversed relative to the
        reference trend while both move faster than
        :data:`TREND_REVERSAL_MGDL_MIN`. The second clause is what Clarke alone
        cannot express.
    ``accurate``
        Point pair in Clarke A/B, and the predicted rate falls in the same
        :data:`RATE_BIN_LABELS` band as the reference rate.
    ``benign``
        Everything else: the point value is clinically acceptable and the trend
        is not reversed, but the predicted speed of change lands in a
        neighbouring band.

    Returns the per-sample arrays used to build :func:`directional_report`, so
    the caller can audit any single decision.
    """
    ref = np.asarray(reference_profile, dtype=float)
    pred = np.asarray(predicted_profile, dtype=float)
    if ref.shape != pred.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs predicted {pred.shape}")

    ref_end, pred_end = ref[:, -1], pred[:, -1]
    ref_rate = profile_rate_of_change(ref, step_minutes, window_minutes)
    pred_rate = profile_rate_of_change(pred, step_minutes, window_minutes)

    zones = clarke_zones(ref_end, pred_end)
    point_ok = np.isin(zones, list(ACCEPTABLE_CLARKE_ZONES))

    same_band = rate_bin(ref_rate) == rate_bin(pred_rate)
    fast_enough = (np.abs(ref_rate) >= TREND_REVERSAL_MGDL_MIN) & (
        np.abs(pred_rate) >= TREND_REVERSAL_MGDL_MIN
    )
    reversed_trend = fast_enough & (np.sign(ref_rate) != np.sign(pred_rate))

    label = np.full(ref_end.shape, "benign", dtype=object)
    label[point_ok & same_band & ~reversed_trend] = "accurate"
    label[~point_ok | reversed_trend] = "erroneous"

    return {
        "label": label,
        "clarke_zone": zones,
        "reference_mgdl": ref_end,
        "predicted_mgdl": pred_end,
        "reference_rate_mgdl_min": ref_rate,
        "predicted_rate_mgdl_min": pred_rate,
        "reversed_trend": reversed_trend,
        "glycemic_range": glycemic_range(ref_end),
    }


def trend_dynamics(reference_profile: np.ndarray,
                   predicted_profile: np.ndarray,
                   step_minutes: float,
                   window_minutes: float = RATE_WINDOW_MINUTES) -> Dict[str, Any]:
    """Does the forecast move at all, and does it move the right way?

    A model trained on a squared-error-like objective at a long horizon can
    minimise its loss by predicting the conditional mean and almost no dynamics.
    Such a forecast scores *well* on MAE and on the Clarke grid — both of which
    only ever see one number per window — while carrying no usable information
    about where glucose is heading. For a closed-loop system that is the
    difference between a useful signal and a flat line with good statistics.

    This function measures that directly:

    ``rate_attenuation``
        SD of the predicted rate divided by SD of the reference rate. A value
        near 1 means the forecast reproduces the observed dynamics; a value near
        0 means it is flat.
    ``sign_concordance_pct``
        Share of windows where the predicted direction matches the reference
        direction, computed only over windows in which the reference is actually
        moving (at least :data:`TREND_REVERSAL_MGDL_MIN`). 50% is a coin flip.
    ``flat_forecast``
        True when the predicted rate never reaches
        :data:`TREND_REVERSAL_MGDL_MIN` anywhere in the dataset. When this is
        True the trend-reversal clause in :func:`classify_predictions` cannot
        fire by construction, so ``erroneous`` collapses onto the plain Clarke
        criterion. Reporting the rate-aware result without this flag would
        overstate what the analysis actually tested.
    """
    ref_rate = profile_rate_of_change(reference_profile, step_minutes, window_minutes)
    pred_rate = profile_rate_of_change(predicted_profile, step_minutes, window_minutes)

    ref_sd = float(np.std(ref_rate))
    pred_sd = float(np.std(pred_rate))
    moving = np.abs(ref_rate) >= TREND_REVERSAL_MGDL_MIN
    # A forecast that does not move has no direction; counting it as the WRONG
    # direction would flatter a genuinely wrong-way forecast by comparison and
    # would make a perfectly flat baseline look maximally discordant. Windows
    # without a predicted direction are reported separately instead.
    has_direction = moving & (pred_rate != 0.0)
    concordance = (
        float(100.0 * np.mean(
            np.sign(ref_rate[has_direction]) == np.sign(pred_rate[has_direction])
        )) if has_direction.any() else None
    )
    flat = bool(np.max(np.abs(pred_rate)) < TREND_REVERSAL_MGDL_MIN)

    return {
        "reference_rate_sd_mgdl_min": ref_sd,
        "predicted_rate_sd_mgdl_min": pred_sd,
        "rate_attenuation": None if ref_sd == 0 else float(pred_sd / ref_sd),
        "reference_max_abs_rate_mgdl_min": float(np.max(np.abs(ref_rate))),
        "predicted_max_abs_rate_mgdl_min": float(np.max(np.abs(pred_rate))),
        "reference_pct_moving": float(100.0 * moving.mean()),
        "predicted_pct_moving": float(
            100.0 * np.mean(np.abs(pred_rate) >= TREND_REVERSAL_MGDL_MIN)
        ),
        "sign_concordance_pct": concordance,
        "sign_concordance_basis": "reference moving and forecast has a direction",
        "predicted_no_direction_pct": float(100.0 * np.mean(pred_rate == 0.0)),
        "n_reference_moving": int(moving.sum()),
        "n_sign_concordance_basis": int(has_direction.sum()),
        "flat_forecast": flat,
        "caveat": (
            "The predicted rate never reaches the trend-reversal threshold, so the "
            "reversal clause cannot fire and 'erroneous' reduces to the Clarke "
            "criterion. Read rate_attenuation and sign_concordance_pct instead: this "
            "forecast carries level information but little or no trend information."
            if flat else None
        ),
    }


def directional_report(reference_profile: np.ndarray,
                       predicted_profile: np.ndarray,
                       step_minutes: float,
                       window_minutes: float = RATE_WINDOW_MINUTES) -> Dict[str, Any]:
    """Stratified accuracy report, overall and per glycemic range.

    Stratification is not decoration. Errors in the hypoglycemic range carry
    asymmetric harm, and they are rare, so a pooled percentage hides them behind
    the euglycemic majority. ``hypo`` is the number to read first.

    Also reports ``overestimation_pct`` per range: the share of forecasts that
    sit *above* the reference. In the hypo range that is the dangerous
    direction — it is the failure mode that leads to insulin being delivered
    into a fall.
    """
    detail = classify_predictions(
        reference_profile, predicted_profile, step_minutes, window_minutes
    )
    label, rng = detail["label"], detail["glycemic_range"]
    residual = detail["predicted_mgdl"] - detail["reference_mgdl"]

    def _block(mask: np.ndarray) -> Dict[str, Any]:
        n = int(mask.sum())
        if n == 0:
            return {"n_pairs": 0, "counts": {k: 0 for k in ("accurate", "benign", "erroneous")},
                    "percentages": {k: None for k in ("accurate", "benign", "erroneous")},
                    "reversed_trend_pct": None, "overestimation_pct": None,
                    "mean_signed_error_mgdl": None}
        counts = {k: int(np.sum(label[mask] == k)) for k in ("accurate", "benign", "erroneous")}
        return {
            "n_pairs": n,
            "counts": counts,
            "percentages": {k: float(100.0 * v / n) for k, v in counts.items()},
            "reversed_trend_pct": float(100.0 * detail["reversed_trend"][mask].mean()),
            "overestimation_pct": float(100.0 * (residual[mask] > 0).mean()),
            "mean_signed_error_mgdl": float(residual[mask].mean()),
        }

    all_mask = np.ones(label.shape, dtype=bool)
    return {
        "definition": (
            "Rate-aware classification. Rate of change is estimated on the predicted "
            "trajectory for the prediction and on the reference trajectory for the "
            "reference, following the design principle of Sivananthan et al. 2011 "
            "(doi:10.1089/dia.2011.0033). Category boundaries are defined locally in "
            "iints.analysis.prediction_accuracy and are NOT the published PRED-EGA "
            "zone tables."
        ),
        "parameters": {
            "step_minutes": float(step_minutes),
            "rate_window_minutes": float(window_minutes),
            "rate_bin_edges_mgdl_min": list(RATE_BIN_EDGES),
            "trend_reversal_mgdl_min": TREND_REVERSAL_MGDL_MIN,
            "acceptable_clarke_zones": sorted(ACCEPTABLE_CLARKE_ZONES),
        },
        "trend_dynamics": trend_dynamics(
            reference_profile, predicted_profile, step_minutes, window_minutes
        ),
        "overall": _block(all_mask),
        "by_range": {name: _block(rng == name) for name in ("hypo", "target", "hyper")},
    }
