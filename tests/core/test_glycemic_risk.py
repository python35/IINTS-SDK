"""Regression tests for the canonical LBGI/HBGI implementation.

These pin the properties that a previous implementation violated: the risk
function was missing the factor 10, so LBGI/HBGI came out exactly one order of
magnitude too small and crossed the published risk-category boundaries. Two
implementations of the same index also coexisted and disagreed.
"""

import numpy as np
import pandas as pd
import pytest

from iints.analysis.diabetes_metrics import DiabetesMetrics
from iints.core import glycemic_risk as gr
from iints.core.clinical_metrics import ClinicalMetricsCalculator

# Fixed vector spanning severe hypo to severe hyper.
REFERENCE_VECTOR = pd.Series([40.0, 55.0, 70.0, 90.0, 112.5,
                              140.0, 180.0, 250.0, 350.0, 400.0])
REFERENCE_LBGI = 6.345390
REFERENCE_HBGI = 13.445030


def test_risk_function_spans_zero_to_one_hundred():
    """r(BG) must reach ~100 at both ends of [20, 600] mg/dL.

    This is the property that fixes the amplitude at 10. If the factor is
    dropped the endpoints land at ~10 instead, and every published LBGI
    reference range becomes meaningless.
    """
    assert gr.bg_risk(20.0)[0] == pytest.approx(100.0, abs=0.1)
    assert gr.bg_risk(600.0)[0] == pytest.approx(100.0, abs=0.1)
    assert gr.RISK_AMPLITUDE == 10.0


def test_branch_boundary_is_the_zero_of_the_risk_transform():
    """The low/high split is the sign of f(BG), i.e. ~112.5 mg/dL."""
    assert gr.EUGLYCEMIC_CENTER_MGDL == pytest.approx(112.5, abs=0.05)
    assert gr.bg_risk_transform(gr.EUGLYCEMIC_CENTER_MGDL)[0] == pytest.approx(0.0, abs=1e-9)


def test_branches_are_exclusive():
    all_high = pd.Series([150.0, 200.0, 300.0])
    all_low = pd.Series([45.0, 60.0, 70.0])
    assert gr.lbgi(all_high) == 0.0
    assert gr.hbgi(all_low) == 0.0
    assert gr.lbgi(all_low) > 0.0
    assert gr.hbgi(all_high) > 0.0


def test_lbgi_plus_hbgi_equals_bgri():
    assert gr.lbgi(REFERENCE_VECTOR) + gr.hbgi(REFERENCE_VECTOR) == pytest.approx(
        gr.bgri(REFERENCE_VECTOR)
    )


def test_reference_vector_is_pinned():
    assert gr.lbgi(REFERENCE_VECTOR) == pytest.approx(REFERENCE_LBGI, rel=1e-6)
    assert gr.hbgi(REFERENCE_VECTOR) == pytest.approx(REFERENCE_HBGI, rel=1e-6)


def test_invalid_readings_are_dropped_not_imputed():
    dirty = pd.Series([100.0, np.nan, 0.0, -5.0, 100.0])
    clean = pd.Series([100.0, 100.0])
    assert gr.lbgi(dirty) == pytest.approx(gr.lbgi(clean))
    assert gr.lbgi(pd.Series([], dtype=float)) == 0.0
    assert gr.hbgi(pd.Series([], dtype=float)) == 0.0


def test_the_two_public_entry_points_agree():
    """Guard against the duplicate implementations drifting apart again."""
    calculator = ClinicalMetricsCalculator()
    assert calculator.calculate_lbgi(REFERENCE_VECTOR) == pytest.approx(
        DiabetesMetrics.blood_glucose_risk_index(REFERENCE_VECTOR, "low")
    )
    assert calculator.calculate_hbgi(REFERENCE_VECTOR) == pytest.approx(
        DiabetesMetrics.blood_glucose_risk_index(REFERENCE_VECTOR, "high")
    )


def test_risk_categories_match_published_bands():
    assert gr.lbgi_risk_category(0.9) == "minimal"
    assert gr.lbgi_risk_category(1.59) == "low"
    assert gr.lbgi_risk_category(3.0) == "moderate"
    assert gr.lbgi_risk_category(6.0) == "high"


def test_magnitude_is_not_off_by_an_order_of_magnitude():
    """A realistic T1D trace must land in the published LBGI range.

    Regression guard: the pre-fix implementation returned ~0.16 for a series
    of this shape, which reads as 'minimal' risk instead of 'low'.
    """
    rng = np.random.default_rng(0)
    trace = pd.Series(np.clip(rng.normal(155.0, 60.0, 5000), 40.0, 400.0))
    value = gr.lbgi(trace)
    assert 0.5 < value < 6.0
    assert gr.lbgi_risk_category(value) in {"low", "moderate"}
