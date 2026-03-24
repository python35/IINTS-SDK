from __future__ import annotations

from typing import Any, Dict

from mdmp_core.runner import MDMP_GRADE_ORDER


def _grade_rank(value: str) -> int:
    try:
        return MDMP_GRADE_ORDER.index(value)
    except ValueError:
        return -1


def _read_grade(payload: Dict[str, Any]) -> str:
    return str(payload.get("effective_grade", payload.get("grade", "raw")))


def compare_reports(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_grade = _read_grade(baseline)
    candidate_grade = _read_grade(candidate)
    baseline_score = float(baseline.get("compliance_score", 0.0))
    candidate_score = float(candidate.get("compliance_score", 0.0))

    dataset_fp_a = str(baseline.get("dataset_fingerprint_sha256", ""))
    dataset_fp_b = str(candidate.get("dataset_fingerprint_sha256", ""))
    same_dataset = dataset_fp_a == dataset_fp_b and dataset_fp_a != ""

    return {
        "baseline": {
            "grade": baseline_grade,
            "grade_reason": baseline.get("grade_reason"),
            "compliance_score": baseline_score,
            "dataset_fingerprint_sha256": dataset_fp_a,
            "schema_version": baseline.get("schema_version"),
        },
        "candidate": {
            "grade": candidate_grade,
            "grade_reason": candidate.get("grade_reason"),
            "compliance_score": candidate_score,
            "dataset_fingerprint_sha256": dataset_fp_b,
            "schema_version": candidate.get("schema_version"),
        },
        "delta": {
            "compliance_score": round(candidate_score - baseline_score, 4),
            "grade_rank": _grade_rank(candidate_grade) - _grade_rank(baseline_grade),
        },
        "summary": {
            "same_dataset_fingerprint": same_dataset,
            "better_grade": _grade_rank(candidate_grade) > _grade_rank(baseline_grade),
            "better_score": candidate_score > baseline_score,
        },
    }
