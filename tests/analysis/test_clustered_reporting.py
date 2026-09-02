"""The reported intervals must count clusters, not runs.

The study matrix runs every algorithm on every (profile, scenario, seed) block,
so runs sharing a profile are repeated measurements of one virtual patient.
These tests pin the three behaviours that make the reported numbers honest:
the interval is taken over clusters, it is refused when the design cannot carry
one, and value/label alignment survives missing metrics.
"""

from __future__ import annotations

from iints.analysis.clustered_inference import MIN_CLUSTERS_FOR_INTERVAL
from iints.analysis.study_analysis import _difference_ci95, _stats


def _clustered_values(n_clusters: int, runs_per_cluster: int) -> tuple[list[float], list[str]]:
    """Between-cluster spread only, so the correct interval is easy to reason about."""
    values: list[float] = []
    clusters: list[str] = []
    for index in range(n_clusters):
        for _ in range(runs_per_cluster):
            values.append(float(index))
            clusters.append(f"profile_{index}")
    return values, clusters


def test_interval_is_taken_over_clusters_not_runs():
    values, clusters = _clustered_values(6, 100)
    clustered = _stats(values, clusters)
    pooled = _stats(values)

    assert clustered["ci_method"] == "cluster_t"
    assert clustered["n_clusters"] == 6
    assert clustered["cluster_level"] == "profile_id"
    # Every run within a cluster is identical here, so all of the spread is
    # between clusters: the pooled interval divides it by sqrt(600) and the
    # cluster interval by sqrt(6). The pooled one is therefore far too narrow.
    half_width = (clustered["ci95_high"] - clustered["ci95_low"]) / 2.0
    assert half_width > 10 * clustered["pseudoreplicated_ci95_half_width"]
    # The pooled call must not smuggle an interval out under the same key.
    assert pooled["ci95_low"] is None
    assert "pseudo-replicated" in pooled["ci95_omitted_because"]


def test_interval_is_refused_below_the_cluster_minimum():
    values, clusters = _clustered_values(MIN_CLUSTERS_FOR_INTERVAL - 1, 50)
    stats = _stats(values, clusters)

    assert stats["ci95_low"] is None
    assert stats["n_clusters"] == MIN_CLUSTERS_FOR_INTERVAL - 1
    assert str(MIN_CLUSTERS_FOR_INTERVAL) in stats["ci95_omitted_because"]
    # The point estimate is still reported; only the interval is withheld.
    assert stats["mean"] is not None
    assert stats["count"] == (MIN_CLUSTERS_FOR_INTERVAL - 1) * 50


def test_missing_metric_does_not_shift_cluster_labels():
    stats = _stats([1.0, None, 2.0], ["profile_a", "profile_b", "profile_c"])
    # profile_b's run is dropped with its value, leaving two clusters, which is
    # below the minimum. Filtering values and labels separately would leave
    # three labels on two values and silently mislabel the second run.
    assert stats["count"] == 2
    assert stats["n_clusters"] == 2
    assert stats["ci95_low"] is None


def test_difference_is_paired_over_shared_clusters():
    profiles = [f"profile_{index}" for index in range(6)]
    left = [10.0 + index for index, _ in enumerate(profiles)]
    right = [8.0 + index for index, _ in enumerate(profiles)]

    estimate = _difference_ci95(left, right, profiles, profiles)

    assert estimate["ci_method"] == "paired_cluster_t"
    assert estimate["n_clusters"] == 6
    assert estimate["difference_in_means"] == 2.0
    # Pairing removes the between-profile trend entirely, so every per-cluster
    # difference is the same and the interval collapses onto the estimate.
    assert set(estimate["per_cluster_differences"].values()) == {2.0}
    assert estimate["ci95_low"] == estimate["ci95_high"] == 2.0


def test_difference_is_refused_without_shared_clusters():
    estimate = _difference_ci95(
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        ["profile_a", "profile_b", "profile_c"],
        ["profile_x", "profile_y", "profile_z"],
    )

    assert estimate["difference_in_means"] == -3.0
    assert estimate["ci95_low"] is None
    assert estimate["n_clusters"] == 0
    assert "cannot be paired" in estimate["ci95_omitted_because"]
