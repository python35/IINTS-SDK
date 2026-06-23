from __future__ import annotations

from iints.analysis.explainable_ai import ClinicalAuditTrail


def test_metabolic_assessment_uses_fixed_flags_and_refuses_diagnosis() -> None:
    assessment = ClinicalAuditTrail._deterministic_metabolic_assessment(
        glucose=[280.0, 290.0],
        ffa=[0.5, 0.8],
        ketones=[2.8, 3.2],
        insulin=[0.0, 0.0],
    )

    assert "fixed_glucose_flag=marked_hyperglycemia_range" in assessment
    assert "fixed_ketone_flag=high_simulated_ketones" in assessment
    assert "fixed_ffa_trend_flag=rising" in assessment
    assert "diagnostic_authority=none" in assessment
    assert "DKA cannot be diagnosed" in assessment
