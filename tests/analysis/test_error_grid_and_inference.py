"""Tests for the computed Clarke error grid and the clustered intervals."""

import numpy as np
import pandas as pd
import pytest

from iints.analysis.clustered_inference import (
    cluster_t_ci,
    compare_algorithms,
    naive_ci,
    paired_block_differences,
)
from iints.analysis.error_grid import (
    ZONES,
    clarke_error_grid,
    clarke_zones,
    iso15197_agreement_rate,
)


# --------------------------------------------------------------------------
# Clarke error grid
# --------------------------------------------------------------------------

def test_perfect_prediction_is_all_zone_a():
    values = np.array([60.0, 90.0, 150.0, 250.0, 350.0])
    result = clarke_error_grid(values, values)
    assert result.percentages["A"] == 100.0
    assert result.hazardous_pct == 0.0


def test_known_hazard_pairs_land_in_the_right_zones():
    # Reference high, predicted hypo -> opposite treatment -> zone E.
    assert clarke_zones([250.0], [50.0])[0] == "E"
    assert clarke_zones([50.0], [250.0])[0] == "E"
    # True severe hyper reported as near-normal -> missed excursion -> zone D.
    assert clarke_zones([300.0], [150.0])[0] == "D"


def test_percentages_sum_to_one_hundred():
    rng = np.random.default_rng(1)
    ref = rng.uniform(40.0, 400.0, 500)
    pred = ref * rng.normal(1.0, 0.25, 500)
    result = clarke_error_grid(ref, pred)
    assert sum(result.percentages[z] for z in ZONES) == pytest.approx(100.0)
    assert result.n_pairs == 500


def test_zone_counts_come_from_data_not_constants():
    """Two different datasets must not produce the same zone percentages."""
    good = np.linspace(80.0, 300.0, 200)
    accurate = clarke_error_grid(good, good * 1.02)
    inaccurate = clarke_error_grid(good, good * 1.60)
    assert accurate.percentages["A"] > inaccurate.percentages["A"]


def test_empty_input_raises_instead_of_fabricating():
    with pytest.raises(ValueError):
        clarke_error_grid([], [])


def test_iso15197_band_switches_at_one_hundred():
    # Below 100 mg/dL the band is absolute (+/-15 mg/dL).
    assert iso15197_agreement_rate([80.0], [94.0]) == 100.0
    assert iso15197_agreement_rate([80.0], [96.0]) == 0.0
    # At or above 100 mg/dL it is relative (+/-15%).
    assert iso15197_agreement_rate([200.0], [229.0]) == 100.0
    assert iso15197_agreement_rate([200.0], [231.0]) == 0.0


# --------------------------------------------------------------------------
# Clustered inference
# --------------------------------------------------------------------------

def _clustered_frame():
    """Six clusters with a real between-cluster offset."""
    rng = np.random.default_rng(0)
    rows = []
    for cluster, offset in enumerate([-6.0, -3.0, -1.0, 1.0, 3.0, 6.0]):
        for scenario in range(4):
            for seed in range(10):
                for algo, effect in (("reference", 0.0), ("candidate", 1.0)):
                    rows.append({
                        "profile_id": f"p{cluster}",
                        "scenario_slug": f"s{scenario}",
                        "seed": seed,
                        "algorithm_id": algo,
                        "tir_70_180": 80.0 + offset + effect + rng.normal(0, 1.0),
                    })
    return pd.DataFrame(rows)


def test_clustering_widens_the_interval():
    frame = _clustered_frame()
    diffs = paired_block_differences(frame, "tir_70_180", "algorithm_id",
                                     "candidate", "reference")
    naive = naive_ci(diffs["difference"])
    clustered = cluster_t_ci(diffs["difference"], diffs["profile_id"])
    assert clustered.n_clusters == 6
    assert clustered.half_width > naive.half_width
    assert clustered.estimate == pytest.approx(naive.estimate, abs=1e-9)


def test_incomplete_blocks_are_dropped_not_imputed():
    frame = _clustered_frame()
    n_before = len(paired_block_differences(frame, "tir_70_180", "algorithm_id",
                                            "candidate", "reference"))
    truncated = frame.drop(frame.index[0])  # remove one arm of one block
    n_after = len(paired_block_differences(truncated, "tir_70_180", "algorithm_id",
                                           "candidate", "reference"))
    assert n_after == n_before - 1


def test_compare_algorithms_reports_both_methods():
    frame = _clustered_frame()
    table = compare_algorithms(frame, reference="reference", n_boot=200)
    assert list(table["treatment"]) == ["candidate"]
    row = table.iloc[0]
    assert row["n_blocks"] == 240
    assert row["n_clusters"] == 6
    assert row["inflation_factor"] > 1.0
    assert row["cluster_t_ci_low"] < row["naive_ci_low"]
