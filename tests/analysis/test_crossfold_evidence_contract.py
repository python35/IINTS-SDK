"""Invariants the cross-fold evidence file must satisfy to be quotable.

These are not schema tests. Each one encodes a claim that, if it silently
stopped holding, would turn the published numbers into a misstatement:

* subjects contribute once, and never to a fold that trained on them;
* every interval is clustered on subjects, never on pairs;
* the effective sample size printed beside a number is the subject count;
* the primary safety outcome is present and stratified;
* the flat-forecast caveat travels with the rate-aware numbers.

The file is optional: it is a build product, not source. When it is absent the
tests skip rather than fail, so a fresh clone is green — but when it exists it
must be honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "apps/iints-tauri/frontend/evidence/crossfold_evidence.json"

pytestmark = pytest.mark.skipif(
    not EVIDENCE.exists(),
    reason="crossfold_evidence.json not built; run research/export_crossfold_evidence.py",
)


@pytest.fixture(scope="module")
def evidence() -> dict:
    return json.loads(EVIDENCE.read_text())


def test_every_subject_is_held_out_exactly_once(evidence):
    seen: dict[str, str] = {}
    for fold in evidence["folds"]:
        for subject in fold["test_subjects"]:
            assert subject not in seen, (
                f"subject {subject} is held out by both {seen[subject]} and "
                f"{fold['name']}; it would be counted twice"
            )
            seen[subject] = fold["name"]
    assert set(seen) == set(evidence["per_subject"]), (
        "the reported subjects differ from the held-out subjects of the folds"
    )
    assert evidence["n_subjects"] == len(seen)


def test_no_fold_reports_on_a_subject_it_trained_or_validated_on(evidence):
    for fold in evidence["folds"]:
        used = set(fold["train_subjects"]) | set(fold["val_subjects"])
        leaked = sorted(set(fold["test_subjects"]) & used)
        assert not leaked, (
            f"{fold['name']} reports on subjects {leaked} that it saw during "
            "training or model selection; validation subjects were consumed by "
            "early stopping and are not out-of-sample"
        )


def test_folds_are_comparable_before_being_pooled(evidence):
    prints = [f["config_fingerprint"] for f in evidence["folds"]]
    assert all(p is not None for p in prints), "a fold was pooled without its config"
    assert all(p == prints[0] for p in prints), (
        "folds differ in training settings; pooling them measures the difference "
        "between recipes rather than between subjects"
    )
    for fold in evidence["folds"]:
        best, cap = fold.get("best_epoch"), fold.get("epoch_cap")
        if best is not None and cap is not None:
            assert best < cap, (
                f"{fold['name']} hit its epoch cap, so the cap chose the model"
            )


def test_intervals_are_clustered_on_subjects_not_pairs(evidence):
    n_subjects = evidence["n_subjects"]
    checked = 0
    for arm in ("model", "persistence"):
        for name, block in evidence["summary"][arm].items():
            if not isinstance(block, dict) or "n_subjects" not in block:
                continue
            assert block["n_subjects"] == n_subjects, (
                f"{arm}.{name} reports n_subjects={block['n_subjects']}"
            )
            assert block["n_pairs"] > block["n_subjects"], (
                f"{arm}.{name} appears to be clustered on pairs"
            )
            assert "cluster" in block["method"].lower(), (
                f"{arm}.{name} was not computed with a cluster-level estimator "
                f"(method={block['method']!r})"
            )
            checked += 1
    assert checked >= 2 * 5, "too few clustered intervals found to trust this check"


def test_primary_safety_outcome_is_present_and_paired(evidence):
    hypo = evidence["primary_outcome_hypoglycemia_detection"]
    assert hypo["n_hypo_windows"] > 0
    assert hypo["n_subjects_with_hypo"] >= 2, (
        "a subject-level interval on the primary outcome needs at least two subjects"
    )
    paired = hypo["paired_model_minus_persistence"]
    for key in ("estimate", "ci_low", "ci_high", "n_subjects", "model_better_at_95pct"):
        assert key in paired, f"primary outcome contrast is missing {key}"
    assert paired["ci_low"] <= paired["estimate"] <= paired["ci_high"]


def test_hypoglycemia_is_reported_separately_from_the_pooled_average(evidence):
    """A pooled percentage hides the hypo range behind the euglycemic majority."""
    for arm in ("model", "persistence"):
        by_range = evidence["summary"][arm]["directional"]["by_range"]
        assert set(by_range) == {"hypo", "target", "hyper"}
        assert by_range["hypo"]["n_pairs"] > 0
        assert "overestimation_pct" in by_range["hypo"], (
            "the dangerous direction must be reported for the hypo range"
        )


def test_flat_forecast_caveat_travels_with_the_rate_aware_numbers(evidence):
    """If the reversal clause cannot fire, the file must say so.

    Otherwise a reader sees 'zero trend reversals' and concludes the forecast
    gets direction right, when in fact it never expressed a direction.
    """
    for arm in ("model", "persistence"):
        dyn = evidence["summary"][arm]["directional"]["trend_dynamics"]
        assert "rate_attenuation" in dyn and "flat_forecast" in dyn
        if dyn["flat_forecast"]:
            assert dyn["caveat"], f"{arm} is a flat forecast but carries no caveat"
            reversed_pct = evidence["summary"][arm]["directional"]["overall"][
                "reversed_trend_pct"
            ]
            assert reversed_pct == 0.0, (
                "a flat forecast cannot produce trend reversals; a non-zero count "
                "means the flag and the classification disagree"
            )


def test_paired_contrast_direction_is_declared_not_inferred(evidence):
    for name, block in evidence["paired_model_minus_persistence"].items():
        assert "higher_is_better" in block, f"{name} does not declare its polarity"
        if block["model_better_at_95pct"] is True:
            if block["higher_is_better"]:
                assert block["ci_low"] > 0, f"{name} claims a win the interval denies"
            else:
                assert block["ci_high"] < 0, f"{name} claims a win the interval denies"


def test_estimand_and_pair_definition_are_stated(evidence):
    assert "subject" in evidence["estimand"].lower()
    assert evidence["pair_definition"]
    assert evidence["horizon_minutes"] > 0
    for fold in evidence["folds"]:
        assert fold["data_sha256"], "evidence without a data checksum is not traceable"
        assert fold["model_sha256"]
