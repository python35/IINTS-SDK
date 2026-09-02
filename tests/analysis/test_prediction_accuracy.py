"""Tests for the rate-aware prediction accuracy analysis.

The load-bearing test in this file is
``test_trend_reversal_inside_clarke_zone_a_is_erroneous``: it encodes the entire
scientific reason the module exists. If that test is ever deleted or weakened,
the project has silently gone back to grading forecasts with a metric designed
for measurements.
"""

from __future__ import annotations

import numpy as np
import pytest

from iints.analysis.error_grid import clarke_zones
from iints.analysis.prediction_accuracy import (
    RATE_BIN_EDGES,
    RATE_BIN_LABELS,
    TREND_REVERSAL_MGDL_MIN,
    classify_predictions,
    directional_report,
    glycemic_range,
    profile_rate_of_change,
    rate_bin,
    trend_dynamics,
)

STEP = 5.0


def _ramp(start: float, rate_mgdl_min: float, n_steps: int = 24) -> np.ndarray:
    """One trajectory that ENDS at ``start``, arriving at ``rate_mgdl_min``."""
    t = np.arange(n_steps, dtype=float) * STEP
    return (start + (t - t[-1]) * rate_mgdl_min)[None, :]


# --- rate estimation ---------------------------------------------------------

def test_rate_of_change_recovers_a_known_slope():
    for slope in (-3.0, -1.5, 0.0, 0.8, 2.5):
        got = profile_rate_of_change(_ramp(120.0, slope), STEP)
        assert got.shape == (1,)
        assert got[0] == pytest.approx(slope, abs=1e-9), f"slope {slope} -> {got[0]}"


def test_rate_window_must_fit_the_horizon():
    with pytest.raises(ValueError, match="needs at least"):
        profile_rate_of_change(_ramp(120.0, 1.0, n_steps=3), STEP, window_minutes=60.0)


def test_rate_window_shorter_than_one_step_is_rejected():
    with pytest.raises(ValueError, match="shorter than one step"):
        profile_rate_of_change(_ramp(120.0, 1.0), STEP, window_minutes=2.0)


def test_rate_bins_are_monotone_and_cover_the_edges():
    rates = [-5.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 5.0]
    bins = rate_bin(rates)
    assert list(bins) == sorted(bins), "bin index must be monotone in the rate"
    assert bins.min() == 0 and bins.max() == len(RATE_BIN_LABELS) - 1
    # Edges are left-inclusive: exactly -2.0 is 'falling', not 'falling_fast'.
    assert RATE_BIN_LABELS[rate_bin(-2.0)] == "falling"
    assert RATE_BIN_LABELS[rate_bin(-2.0001)] == "falling_fast"
    assert len(RATE_BIN_LABELS) == len(RATE_BIN_EDGES) + 1


# --- the reason this module exists -------------------------------------------

def test_trend_reversal_inside_clarke_zone_a_is_erroneous():
    """Numerically close, clinically inverted: Clarke says A, this must not.

    Reference is 95 mg/dL falling 2.5 mg/dL/min — heading into a severe hypo.
    The forecast says 105 mg/dL and rising. Clarke grades the point pair A,
    because 95 vs 105 is a 10 mg/dL error in the euglycemic range.
    """
    ref = _ramp(95.0, -2.5)
    pred = _ramp(105.0, +2.5)

    assert clarke_zones(ref[:, -1], pred[:, -1])[0] == "A", (
        "precondition: Clarke must consider this pair acceptable, "
        "otherwise the test is not testing what it claims"
    )
    out = classify_predictions(ref, pred, STEP)
    assert out["label"][0] == "erroneous"
    assert bool(out["reversed_trend"][0]) is True


def test_slow_sign_flip_is_not_penalised():
    """Below the reversal threshold the sign of the rate is noise, not a trend."""
    slow = TREND_REVERSAL_MGDL_MIN / 4.0
    out = classify_predictions(_ramp(120.0, -slow), _ramp(120.0, +slow), STEP)
    assert bool(out["reversed_trend"][0]) is False
    assert out["label"][0] != "erroneous"


def test_perfect_forecast_is_accurate():
    ref = _ramp(140.0, -1.5)
    out = classify_predictions(ref, ref.copy(), STEP)
    assert out["label"][0] == "accurate"
    assert out["clarke_zone"][0] == "A"


def test_neighbouring_speed_band_is_benign_not_erroneous():
    """Same direction, one band too slow: acceptable point, imperfect trend."""
    out = classify_predictions(_ramp(150.0, -2.5), _ramp(150.0, -1.5), STEP)
    assert out["label"][0] == "benign"


def test_point_error_outside_ab_is_erroneous_even_with_perfect_trend():
    ref = _ramp(60.0, -1.5)
    pred = _ramp(200.0, -1.5)
    assert clarke_zones(ref[:, -1], pred[:, -1])[0] not in ("A", "B")
    out = classify_predictions(ref, pred, STEP)
    assert out["label"][0] == "erroneous"


# --- stratification ----------------------------------------------------------

def test_glycemic_range_thresholds():
    got = list(glycemic_range([50.0, 69.9, 70.0, 180.0, 180.1, 300.0]))
    assert got == ["hypo", "hypo", "target", "target", "hyper", "hyper"]


def test_report_partitions_every_pair_exactly_once():
    rng = np.random.default_rng(11)
    n = 500
    ref = np.cumsum(rng.normal(0, 2, (n, 24)), axis=1) + rng.uniform(50, 300, (n, 1))
    pred = ref + rng.normal(0, 25, (n, 24))
    rep = directional_report(ref, pred, STEP)

    assert rep["overall"]["n_pairs"] == n
    assert sum(rep["by_range"][r]["n_pairs"] for r in ("hypo", "target", "hyper")) == n
    for block in [rep["overall"]] + [rep["by_range"][r] for r in ("hypo", "target", "hyper")]:
        if block["n_pairs"] == 0:
            continue
        assert sum(block["counts"].values()) == block["n_pairs"]
        assert sum(block["percentages"].values()) == pytest.approx(100.0, abs=1e-9)


def test_report_records_its_own_parameters():
    """A number without its definition is not a result."""
    rep = directional_report(_ramp(120.0, 0.0), _ramp(120.0, 0.0), STEP)
    p = rep["parameters"]
    assert p["step_minutes"] == STEP
    assert p["trend_reversal_mgdl_min"] == TREND_REVERSAL_MGDL_MIN
    assert p["rate_bin_edges_mgdl_min"] == list(RATE_BIN_EDGES)


def test_report_does_not_claim_to_be_the_published_grid():
    """Guard against a future rename silently misattributing the method.

    The category boundaries here were defined locally because the PRED-EGA zone
    tables were not accessible. Anyone who obtains the paper and verifies this
    implementation against it may relax this test — deliberately, not by
    accident.
    """
    text = directional_report(_ramp(120.0, 0.0), _ramp(120.0, 0.0), STEP)["definition"]
    assert "NOT the published PRED-EGA" in text
    assert "doi:10.1089/dia.2011.0033" in text


# --- flat-forecast detection -------------------------------------------------

def test_flat_forecast_is_flagged_and_explained():
    """A forecast with no dynamics must not pass as 'zero trend reversals'.

    This is the degeneracy that a point-wise metric cannot express: predicting
    the conditional mean scores well on MAE and on Clarke while carrying no
    information about direction.
    """
    rng = np.random.default_rng(3)
    n = 300
    slopes = rng.uniform(-3.0, 3.0, n)
    ref = np.stack([_ramp(150.0, s)[0] for s in slopes])
    flat = np.full_like(ref, 150.0)

    dyn = directional_report(ref, flat, STEP)["trend_dynamics"]
    assert dyn["flat_forecast"] is True
    assert dyn["caveat"] is not None and "reduces to the Clarke criterion" in dyn["caveat"]
    assert dyn["rate_attenuation"] == pytest.approx(0.0, abs=1e-12)
    assert dyn["predicted_pct_moving"] == 0.0
    assert dyn["reference_pct_moving"] > 0.0


def test_faithful_forecast_is_not_flagged_flat():
    rng = np.random.default_rng(4)
    slopes = rng.uniform(-3.0, 3.0, 300)
    ref = np.stack([_ramp(150.0, s)[0] for s in slopes])
    dyn = directional_report(ref, ref.copy(), STEP)["trend_dynamics"]
    assert dyn["flat_forecast"] is False
    assert dyn["caveat"] is None
    assert dyn["rate_attenuation"] == pytest.approx(1.0, rel=1e-9)
    assert dyn["sign_concordance_pct"] == pytest.approx(100.0)


def test_motionless_forecast_has_no_direction_rather_than_a_wrong_one():
    """A flat baseline must not be scored as 0% concordant; it has no direction."""
    ref = np.stack([_ramp(150.0, s)[0] for s in (2.0, -2.0, 3.0, -3.0)])
    flat = np.full_like(ref, 150.0)
    dyn = trend_dynamics(ref, flat, STEP)
    assert dyn["predicted_no_direction_pct"] == pytest.approx(100.0)
    assert dyn["n_sign_concordance_basis"] == 0
    assert dyn["sign_concordance_pct"] is None


def test_sign_concordance_is_measured_only_where_reference_moves():
    """Direction is undefined when the reference is flat; those windows are excluded."""
    still = np.stack([_ramp(150.0, 0.0)[0]] * 50)          # reference not moving
    moving = np.stack([_ramp(150.0, 2.0)[0]] * 50)         # reference rising fast
    ref = np.concatenate([still, moving])
    pred = np.concatenate([_ramp(150.0, -2.0)[0][None, :].repeat(50, 0),  # wrong way
                           moving.copy()])                                # right way
    dyn = trend_dynamics(ref, pred, STEP)
    assert dyn["n_reference_moving"] == 50
    assert dyn["sign_concordance_pct"] == pytest.approx(100.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        classify_predictions(_ramp(120.0, 0.0), _ramp(120.0, 0.0)[:, :-1], STEP)
