from __future__ import annotations

from iints.data.evidence import rank_real_data_sources
from iints.data.realism_governance import review_real_data_realism
from iints.data.realism_reference import get_realism_reference
from iints.data.realism_validator import RealismCheck, RealismReport


def _strict_pass_report() -> RealismReport:
    return RealismReport(
        verdict="likely_realistic",
        realism_score=0.96,
        summary="likely realistic",
        metrics={
            "duration_hours": 24.0,
            "rapid_change_count": 0,
            "long_gap_count": 0,
            "impossible_value_count": 0,
            "tir_below_54_pct": 0.0,
            "tir_below_70_pct": 2.0,
            "meal_count": 3,
            "insulin_event_count": 3,
        },
        checks=[
            RealismCheck("quality_basics", "quality", "passed", "info", "ok"),
            RealismCheck("glucose_variability", "variability", "passed", "info", "ok"),
            RealismCheck("event_balance", "events", "passed", "info", "ok"),
            RealismCheck("meal_response", "meals", "passed", "info", "ok"),
            RealismCheck("causal_alignment", "causal", "passed", "info", "ok"),
            RealismCheck("overnight_shape", "overnight", "passed", "info", "ok"),
            RealismCheck(
                "reference_envelope",
                "reference",
                "passed",
                "info",
                "ok",
                metrics={"failed_metrics": 0, "warning_metrics": 0},
            ),
        ],
        meal_responses=[],
        warnings=[],
        reference_profile=get_realism_reference("free_living_t1d"),
    )


def test_strict_real_data_gate_passes_clean_external_reference_report() -> None:
    result = review_real_data_realism(_strict_pass_report())

    assert result.passed is True
    assert result.status == "passed"
    assert result.critical_failures == []


def test_strict_real_data_gate_blocks_review_verdict_and_sensor_artifacts() -> None:
    report = _strict_pass_report()
    bad = RealismReport(
        verdict="needs_review",
        realism_score=0.7,
        summary="needs review",
        metrics={
            **report.metrics,
            "rapid_change_count": 2,
            "tir_below_54_pct": 3.0,
        },
        checks=report.checks,
        meal_responses=[],
        warnings=[],
        reference_profile=report.reference_profile,
    )

    result = review_real_data_realism(bad)

    assert result.passed is False
    assert result.status == "blocked"
    assert any("needs_review" in item for item in result.critical_failures)
    assert any("rapid-change" in item for item in result.critical_failures)


def test_real_data_source_ranking_prioritizes_calibration_references() -> None:
    ranked = rank_real_data_sources()
    by_id = {row["id"]: row for row in ranked}

    assert by_id["azt1d"]["tier"] == "tier_1_calibration_reference"
    assert by_id["hupa_ucm"]["tier"] == "tier_1_calibration_reference"
    assert by_id["sample"]["tier"] == "demo_only"
