"""Pin the numbers in EUCYS_REPORT.md to the exported statistics.

The report is hand-written prose, and its intervals once described a method the
SDK had already replaced. This test recomputes the statistics from the archived
run records and checks that the intervals the report prints are the ones the
current analysis produces, so the two cannot drift apart again unnoticed.

The test is skipped when the 3600-run bundle is not present (it is a large
results directory, not part of a fresh clone).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "eucys_2026"
REPORT = REPO_ROOT / "research" / "EUCYS_REPORT.md"

pytestmark = pytest.mark.skipif(
    not (RESULTS_ROOT / "study_summary.json").is_file() or not REPORT.is_file(),
    reason="EUCYS benchmark bundle or report not present in this checkout",
)


@pytest.fixture(scope="module")
def statistics() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "research"))
    from export_eucys_statistics import build_statistics

    return build_statistics(RESULTS_ROOT)


@pytest.fixture(scope="module")
def report_text() -> str:
    return REPORT.read_text(encoding="utf-8")


def _interval_appears(text: str, low: float, high: float) -> bool:
    """True when the report prints this interval in any of its formats."""
    patterns = [
        rf"{low:.2f}\s*to\s*{high:.2f}",
        rf"{low:+.2f}\s*to\s*{high:+.2f}",
        rf"{low:.2f}\s*to\s*{high:+.2f}",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def test_overall_tir_interval_is_current(statistics, report_text) -> None:
    block = statistics["full_bundle"]["aggregate"]["tir_70_180"]
    assert block["ci_method"] == "cluster_t"
    assert _interval_appears(report_text, block["ci95_low"], block["ci95_high"]), (
        f"report does not print the current overall TIR interval "
        f"{block['ci95_low']:.2f} to {block['ci95_high']:.2f}"
    )


def test_per_algorithm_intervals_are_current(statistics, report_text) -> None:
    missing = [
        label
        for label, row in statistics["full_bundle"]["by_algorithm"].items()
        if row["ci95_low"] is not None
        and not _interval_appears(report_text, row["ci95_low"], row["ci95_high"])
    ]
    assert not missing, f"stale or absent per-algorithm intervals in the report: {missing}"


def test_arm_intervals_are_current(statistics, report_text) -> None:
    missing = [
        label
        for label, row in statistics["full_bundle"]["by_arm"].items()
        if row["ci95_low"] is not None
        and not _interval_appears(report_text, row["ci95_low"], row["ci95_high"])
    ]
    assert not missing, f"stale or absent per-arm intervals in the report: {missing}"


def test_undecided_contrast_is_not_claimed_as_established(statistics, report_text) -> None:
    """The comparison whose interval includes zero must be reported as undecided.

    This is the claim the pseudo-replicated interval used to support, so it is
    the one worth guarding: if a future edit re-states it as an improvement, the
    report is overclaiming again.
    """
    contrasts = statistics["arms"]["clean_certified"]["paired_contrasts_tir"]["baselines"]
    undecided = [name for name, row in contrasts.items() if row["excludes_zero"] is False]
    assert undecided, "expected at least one clean-arm contrast whose interval includes zero"
    for name in undecided:
        row = contrasts[name]
        assert _interval_appears(report_text, row["ci95_low"], row["ci95_high"]), (
            f"report omits the interval for the undecided contrast against {name}"
        )
        assert re.search(
            r"not distinguishable|includes zero|not established|stays undecided|does not \(95",
            report_text,
        ), "report prints the interval but never states that the comparison is undecided"


def test_no_pseudoreplicated_interval_method_described(report_text) -> None:
    """The old method must not be presented as the method in use.

    It may still be named where the report explains what changed, so the check
    is that it is not offered as the current computation.
    """
    assert not re.search(
        r"the 95% confidence intervals are computed as mean \+/- `1\.96 \* standard error`",
        report_text,
    ), "report still describes the run-level interval as its method"
    assert "cluster" in report_text.lower(), "report never states the clustering level"
