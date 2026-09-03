#!/usr/bin/env python3
"""Tests for CG-EGA: rate classification, P-EGA expansion, combination matrix."""

from __future__ import annotations

import numpy as np
import pytest

from iints.analysis.continuous_error_grid import (
    CGEGA_LABELS,
    CGEGA_MATRIX,
    GLYCEMIC_REGIONS,
    PEGA_EXPANSION_MGDL,
    REGA_ZONES,
    cgega,
    classify_rate,
    combine_cgega,
    glycemic_region,
    pega_zones,
    rate_deviation,
    rate_of_change,
    rega_zones,
)
from iints.analysis.error_grid import clarke_zones

#: Clinical severity order, used to assert that the rate expansion can only
#: ever be a relaxation.
_SEVERITY = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


# ---------------------------------------------------------------- matrix ---

def test_matrix_columns_per_region_match_the_published_table():
    assert set(CGEGA_MATRIX) == set(GLYCEMIC_REGIONS)
    expected_columns = {
        "hypoglycemia": {"A", "D", "E"},
        "euglycemia": {"A", "B", "C"},
        "hyperglycemia": {"A", "B", "C", "D", "E"},
    }
    for region, columns in expected_columns.items():
        for r_zone in REGA_ZONES:
            assert set(CGEGA_MATRIX[region][r_zone]) == columns


def test_matrix_rows_cover_every_rega_zone_with_valid_labels():
    for region in GLYCEMIC_REGIONS:
        assert set(CGEGA_MATRIX[region]) == set(REGA_ZONES)
        for row in CGEGA_MATRIX[region].values():
            assert set(row.values()) <= set(CGEGA_LABELS)


@pytest.mark.parametrize(
    "region,r_zone,p_zone,expected",
    [
        # Both grids accurate is an accurate reading in every region.
        ("hypoglycemia", "A", "A", "AP"),
        ("euglycemia", "A", "A", "AP"),
        ("hyperglycemia", "A", "A", "AP"),
        # A missed rate in hypoglycemia: predicting a fall while glucose is
        # rising is benign there, because it prompts no dangerous action.
        ("hypoglycemia", "lE", "A", "BE"),
        # The mirror case is not benign: missing a fall in hypoglycemia.
        ("hypoglycemia", "uD", "A", "EP"),
        ("hypoglycemia", "uE", "A", "EP"),
        # In hyperglycemia the asymmetry reverses.
        ("hyperglycemia", "uD", "A", "BE"),
        ("hyperglycemia", "lD", "A", "EP"),
        # A hazardous point zone is erroneous regardless of the rate grade.
        ("euglycemia", "A", "C", "EP"),
        ("hyperglycemia", "A", "E", "EP"),
    ],
)
def test_published_cells(region, r_zone, p_zone, expected):
    assert combine_cgega(p_zone, r_zone, region) == expected


def test_undefined_region_and_zone_combinations_raise_rather_than_default():
    # The published table has no B column in hypoglycemia. A silent default
    # here would score a pair the source does not score.
    with pytest.raises(ValueError, match="no cell for P-EGA zone"):
        combine_cgega("B", "A", "hypoglycemia")
    with pytest.raises(ValueError, match="unknown R-EGA zone"):
        combine_cgega("A", "Z", "euglycemia")
    with pytest.raises(ValueError, match="unknown glycemic region"):
        combine_cgega("A", "A", "normoglycemia")


# ------------------------------------------------------------- regions ----

def test_glycemic_region_boundaries_are_inclusive_of_euglycemia():
    regions = glycemic_region([69.9, 70.0, 120.0, 180.0, 180.1])
    assert list(regions) == [
        "hypoglycemia",
        "euglycemia",
        "euglycemia",
        "euglycemia",
        "hyperglycemia",
    ]


# ---------------------------------------------------------------- rates ---

def test_rate_of_change_leaves_the_first_sample_undefined():
    rate = rate_of_change([100.0, 110.0, 95.0], [0.0, 5.0, 10.0])
    assert np.isnan(rate[0])
    assert rate[1] == pytest.approx(2.0)
    assert rate[2] == pytest.approx(-3.0)


def test_rate_of_change_rejects_non_increasing_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        rate_of_change([100.0, 110.0], [5.0, 5.0])
    with pytest.raises(ValueError, match="at least two samples"):
        rate_of_change([100.0], [0.0])


@pytest.mark.parametrize(
    "rate,expected",
    [
        (0.0, "slow"),
        (1.0, "slow"),        # at the bound, still slow
        (-1.0, "slow"),
        (1.5, "moderate_rise"),
        (-1.5, "moderate_fall"),
        (2.0, "moderate_rise"),   # at the bound, still moderate
        (-2.0, "moderate_fall"),
        (2.5, "rapid_rise"),
        (-2.5, "rapid_fall"),
    ],
)
def test_rate_classes_at_and_around_the_bounds(rate, expected):
    assert classify_rate([rate])[0] == expected


def test_unmeasurable_rate_is_treated_as_slow():
    # The conservative direction: no expansion, so no error gets excused on
    # the strength of a rate that was never measured.
    assert classify_rate([np.nan])[0] == "slow"


def test_rate_deviation_is_a_signed_rate_difference():
    rd = rate_deviation([2.0, -1.0], [1.5, -1.5])
    assert rd == pytest.approx([0.5, 0.5])


# ----------------------------------------------------------------- P-EGA --

def _dense_grid():
    ref = np.arange(40.0, 401.0, 5.0)
    pred = np.arange(40.0, 401.0, 5.0)
    rr, pp = np.meshgrid(ref, pred, indexing="ij")
    return rr.ravel(), pp.ravel()


def test_slow_change_reduces_pega_to_the_plain_clarke_grid():
    ref, pred = _dense_grid()
    slow = np.full(ref.shape, 0.5)
    assert np.array_equal(pega_zones(ref, pred, slow), clarke_zones(ref, pred))


@pytest.mark.parametrize("rate", [-1.5, -2.5, 1.5, 2.5])
def test_expansion_is_only_ever_a_relaxation(rate):
    # The whole point of the rate expansion is to excuse readings that a
    # fast reference movement explains. It must never make a zone worse.
    ref, pred = _dense_grid()
    rates = np.full(ref.shape, rate)
    expanded = pega_zones(ref, pred, rates)
    plain = clarke_zones(ref, pred)
    sev_expanded = np.vectorize(_SEVERITY.__getitem__)(expanded)
    sev_plain = np.vectorize(_SEVERITY.__getitem__)(plain)
    assert np.all(sev_expanded <= sev_plain)


def test_falling_reference_excuses_reading_high_by_the_published_amount():
    # Zone A against a reference of 100 reaches to 120 mg/dL. A reading of
    # 130 is outside it on the plain grid, and a moderate fall moves the
    # limit by 10 mg/dL, which brings it exactly to the bound.
    ref = np.array([100.0, 100.0, 100.0])
    rates = np.array([-0.5, -1.5, -2.5])
    assert list(pega_zones(ref, np.full(3, 130.0), rates)) == ["B", "A", "A"]

    # A reading of 140 separates the two amounts: 10 mg/dL leaves it outside
    # zone A, 20 mg/dL brings it to the bound.
    assert list(pega_zones(ref, np.full(3, 140.0), rates)) == ["B", "B", "A"]
    assert PEGA_EXPANSION_MGDL["moderate"] == 10.0
    assert PEGA_EXPANSION_MGDL["rapid"] == 20.0


def test_rising_reference_excuses_reading_low_but_falling_does_not():
    # A reading of 70 against a reference of 100 is 30% low. Only a rising
    # reference explains it; a falling reference makes it a real error.
    ref = np.array([100.0, 100.0])
    pred = np.array([70.0, 70.0])
    rising = pega_zones(ref, pred, np.array([2.5, 2.5]))
    falling = pega_zones(ref, pred, np.array([-2.5, -2.5]))
    assert rising[0] == "A"
    assert falling[0] == "B"


def test_expansion_never_carries_a_reading_across_the_reference():
    # A 20 mg/dL shift applied to a reading only 5 mg/dL above the reference
    # must stop at the reference, not overshoot into a zone on the far side.
    ref = np.array([100.0])
    pred = np.array([105.0])
    assert pega_zones(ref, pred, np.array([-2.5]))[0] == "A"


def test_pega_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        pega_zones([100.0, 110.0], [100.0], [0.0, 0.0])


# ----------------------------------------------------------------- R-EGA --

def test_rega_zones_refuses_to_guess_its_boundaries():
    with pytest.raises(NotImplementedError, match="boundaries are not available"):
        rega_zones([1.0], [1.0])


# ---------------------------------------------------------------- CG-EGA --

def _series():
    minutes = np.arange(0.0, 30.0, 5.0)
    reference = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])
    predicted = np.array([100.0, 104.0, 112.0, 113.0, 121.0, 126.0])
    return reference, predicted, minutes


def test_cgega_requires_rega_grades_and_says_why():
    reference, predicted, minutes = _series()
    with pytest.raises(NotImplementedError, match="boundaries are not available"):
        cgega(reference, predicted, minutes)


def test_cgega_drops_the_first_sample_because_it_has_no_rate():
    reference, predicted, minutes = _series()
    result = cgega(reference, predicted, minutes, r_zones=["A"] * 5)
    assert result.n_pairs == 5
    # The same grades supplied per sample rather than per rate must agree.
    per_sample = cgega(reference, predicted, minutes, r_zones=["A"] * 6)
    assert per_sample.counts == result.counts


def test_cgega_accurate_series_scores_as_accurate_readings():
    reference, predicted, minutes = _series()
    result = cgega(reference, predicted, minutes, r_zones=["A"] * 5)
    assert result.counts["AP"] == 5
    assert result.percentages["AP"] == pytest.approx(100.0)
    assert result.erroneous_pct == pytest.approx(0.0)
    assert "n=5" in result.summary_line()


def test_cgega_reports_within_region_percentages():
    # Two pairs in hypoglycemia, two in hyperglycemia; the hypoglycemic
    # pairs carry an R-EGA grade that is benign there and erroneous when
    # the reference is high, so pooling would hide the difference.
    minutes = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    reference = np.array([60.0, 62.0, 64.0, 300.0, 302.0])
    predicted = np.array([60.0, 62.0, 64.0, 300.0, 302.0])
    result = cgega(reference, predicted, minutes, r_zones=["lE"] * 4)

    assert result.n_by_region["hypoglycemia"] == 2
    assert result.n_by_region["hyperglycemia"] == 2
    assert result.by_region_percentages["hypoglycemia"]["BE"] == pytest.approx(100.0)
    assert result.by_region_percentages["hyperglycemia"]["EP"] == pytest.approx(100.0)
    # Within-region percentages are not a partition of the whole.
    assert result.by_region_percentages["hypoglycemia"]["BE"] != \
        result.percentages["BE"]
    assert "hypoglycemia: n=2" in result.region_line("hypoglycemia")


def test_cgega_region_line_reports_empty_regions_honestly():
    reference, predicted, minutes = _series()
    result = cgega(reference, predicted, minutes, r_zones=["A"] * 5)
    assert result.n_by_region["hypoglycemia"] == 0
    assert result.region_line("hypoglycemia") == "hypoglycemia: no pairs"


def test_cgega_rejects_unknown_rega_grades_and_bad_lengths():
    reference, predicted, minutes = _series()
    with pytest.raises(ValueError, match="unknown R-EGA zones"):
        cgega(reference, predicted, minutes, r_zones=["A", "A", "A", "A", "Q"])
    with pytest.raises(ValueError, match="expected 6"):
        cgega(reference, predicted, minutes, r_zones=["A", "A"])


def test_cgega_refuses_when_no_valid_pairs_remain():
    minutes = np.array([0.0, 5.0, 10.0])
    reference = np.array([100.0, np.nan, -5.0])
    predicted = np.array([100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="none supplied"):
        cgega(reference, predicted, minutes, r_zones=["A", "A"])
