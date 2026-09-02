"""Head-to-head comparison of training arms on one shared held-out subject set.

This is the counterpart of ``export_crossfold_evidence.py``, and the two answer
different questions. Cross-fold pooling asks *how well does this recipe
generalise to unseen people*, and therefore demands that every fold be trained
identically and differ only in seed and split. An arm comparison asks *does
changing one ingredient of the recipe change the outcome*, and therefore demands
the exact opposite: the same subjects, the same windows, and a difference
confined to the ingredient under test.

Both demands are enforced rather than assumed:

* every arm must hold out the same subjects, and use the same validation
  subjects — otherwise one arm was model-selected on people the other was not;
* the evaluation windows must be identical arrays, so every contrast is paired
  window by window and no arm is scored on an easier slice;
* the config fingerprints must agree on everything except the keys under test.
  A loss comparison in which the horizon also changed measures neither.

The estimand is the same as everywhere else in this codebase: the mean over
*subjects*, not over pairs. With a shared split the subject count is whatever
that split held out, which is typically small. Rather than print a confidence
interval that a reader would take at face value, this exporter refuses to
compute one below :data:`MIN_SUBJECTS_FOR_INTERVAL` and reports each subject
separately instead. A difference that holds in two of two people is worth
reporting; calling it a 95% interval is not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _evidence_common import (  # noqa: E402
    HIGHER_IS_BETTER,
    HYPO_THRESHOLD_MGDL,
    Fold,
    hypo_sensitivity_by_subject,
    interval,
    load_fold,
    per_pair_indicators,
)
from iints.analysis.prediction_accuracy import directional_report, trend_dynamics  # noqa: E402
from iints.research.evaluation import hypoglycemia_detection_report  # noqa: E402

#: Below this many subjects a cluster-level t interval is reported as absent
#: rather than computed. Two subjects give one degree of freedom and an interval
#: so wide it is uninformative, but it still *looks* like evidence in a table.
MIN_SUBJECTS_FOR_INTERVAL = 3

#: Fingerprint keys that together describe "the loss function". Varying any of
#: them is one experiment; everything else must match, or the contrast is
#: confounded.
LOSS_KEYS = frozenset({
    "loss",
    "band_weighted_low_threshold", "band_weighted_low_weight",
    "band_weighted_high_threshold", "band_weighted_high_weight",
    "band_weighted_max_weight",
    "safety_weighted_alpha", "safety_weighted_low_threshold",
    "safety_weighted_max_weight",
})

#: What an arm comparison may be about. The caller names the ingredient under
#: test, and every other fingerprinted setting must be held constant. This is a
#: named set rather than a free-form flag so that "loss" cannot quietly grow to
#: include the target parameterization: an arm that changed both the loss and
#: predict_delta would answer neither question.
UNDER_TEST_PRESETS: dict[str, frozenset[str]] = {
    "loss": LOSS_KEYS,
    "target_parameterization": frozenset({"predict_delta"}),
}


def _subject_means(values: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    return {
        s: float(np.mean(values[subjects == s]))
        for s in sorted(set(subjects.tolist()))
    }


def _estimate(values: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    """Subject-level point estimate, with an interval only when one is honest."""
    per_subject = _subject_means(values, subjects)
    n = len(per_subject)
    out: dict[str, Any] = {
        "estimate": float(np.mean(list(per_subject.values()))),
        "per_subject": per_subject,
        "n_subjects": n,
        "n_pairs": int(values.size),
        "estimand": "mean over subjects of the per-subject mean",
    }
    if n >= MIN_SUBJECTS_FOR_INTERVAL:
        out["interval"] = interval(values, subjects)
    else:
        out["interval"] = None
        out["interval_omitted_because"] = (
            f"{n} subjects is below the minimum of {MIN_SUBJECTS_FOR_INTERVAL}; "
            "a cluster-level interval here would be uninformative and would read "
            "as more evidence than the design contains"
        )
    return out


def _arm_summary(fold: Fold, predicted: np.ndarray) -> dict[str, Any]:
    ref, subjects, step = fold.reference, fold.subjects, fold.step_minutes
    indicators = per_pair_indicators(ref, predicted, step)
    hypo = hypoglycemia_detection_report(
        ref[:, -1], predicted[:, -1], threshold_mgdl=HYPO_THRESHOLD_MGDL
    )
    return {
        "metrics": {k: _estimate(v, subjects) for k, v in indicators.items()},
        "hypoglycemia_detection": {
            "sensitivity_pct": hypo["sensitivity_pct"],
            "n_hypo_windows": hypo["counts"]["true_positive"] + hypo["counts"]["false_negative"],
            "specificity_pct": hypo["specificity_pct"],
            "precision_pct": hypo["precision_pct"],
            "by_subject": hypo_sensitivity_by_subject(ref, predicted, subjects),
            "report": hypo,
        },
        "directional": directional_report(ref, predicted, step),
        "trend_dynamics": trend_dynamics(ref, predicted, step),
    }


def _paired_contrast(fold: Fold,
                     treatment: np.ndarray,
                     control: np.ndarray,
                     step: float) -> dict[str, Any]:
    """Per-subject difference treatment - control, on identical windows."""
    subjects = fold.subjects
    t_ind = per_pair_indicators(fold.reference, treatment, step)
    c_ind = per_pair_indicators(fold.reference, control, step)
    out: dict[str, Any] = {}
    for metric in t_ind:
        diff = t_ind[metric] - c_ind[metric]
        block = _estimate(diff, subjects)
        block["higher_is_better"] = metric in HIGHER_IS_BETTER
        favourable = (
            block["estimate"] > 0 if block["higher_is_better"] else block["estimate"] < 0
        )
        block["favours_treatment"] = bool(favourable)
        block["consistent_across_subjects"] = bool(
            all((v > 0) == (block["estimate"] > 0) for v in block["per_subject"].values())
        )
        iv = block["interval"]
        block["established_at_95pct"] = bool(
            iv is not None
            and (iv["ci_low"] > 0 if block["higher_is_better"] else iv["ci_high"] < 0)
        )
        out[metric] = block
    return out


def _hypo_contrast(fold: Fold,
                   treatment: np.ndarray,
                   control: np.ndarray) -> dict[str, Any]:
    """Paired difference in hypo detection, restricted to hypoglycemic windows."""
    ref, subjects = fold.reference, fold.subjects
    mask = ref[:, -1] < HYPO_THRESHOLD_MGDL
    if not mask.any():
        return {"n_hypo_windows": 0, "note": "no hypoglycemic endpoints in this split"}
    detected = {
        "treatment": (treatment[mask, -1] < HYPO_THRESHOLD_MGDL).astype(float) * 100.0,
        "control": (control[mask, -1] < HYPO_THRESHOLD_MGDL).astype(float) * 100.0,
    }
    hypo_subjects = subjects[mask]
    block = _estimate(detected["treatment"] - detected["control"], hypo_subjects)
    block["higher_is_better"] = True
    block["favours_treatment"] = bool(block["estimate"] > 0)
    # With a two-subject split this is the only replication statement available,
    # so it must be in the payload rather than left for a reader to eyeball.
    block["consistent_across_subjects"] = bool(
        all((v > 0) == (block["estimate"] > 0) for v in block["per_subject"].values())
    )
    iv = block["interval"]
    block["established_at_95pct"] = bool(iv is not None and iv["ci_low"] > 0)
    return {
        "n_hypo_windows": int(mask.sum()),
        "n_subjects_with_hypo": len(set(hypo_subjects.tolist())),
        "treatment_sensitivity": _estimate(detected["treatment"], hypo_subjects),
        "control_sensitivity": _estimate(detected["control"], hypo_subjects),
        "paired_treatment_minus_control": block,
        "definition": (
            f"windows whose reference endpoint is below {HYPO_THRESHOLD_MGDL:g} mg/dL "
            "for which the forecast endpoint is also below it"
        ),
    }


def _offset_control(control: np.ndarray, treatment: np.ndarray) -> tuple[np.ndarray, float]:
    """The control arm shifted by one constant, matched to the treatment's bias.

    A loss that penalises hypoglycemic errors can improve hypo detection two
    ways: by forecasting the descent, or by predicting lower everywhere. The
    second is a bias, and a bias needs no training at all — subtracting a
    constant at inference time achieves it. This control separates the two.

    The constant is the mean difference between the two arms *on the held-out
    data itself*, which is the most favourable choice possible for the control.
    That is deliberate: if even an optimally shifted control matches the
    treatment, the treatment's advantage is not a better forecast.
    """
    offset = float(np.mean(treatment - control))
    return control + offset, offset


def _check_arms_are_comparable(
    folds: dict[str, Fold],
    under_test: str = "loss",
) -> dict[str, Any]:
    """Refuse the comparison unless it is actually a comparison.

    ``under_test`` names the ingredient the contrast is about (see
    ``UNDER_TEST_PRESETS``). Only its keys may differ between arms.
    """
    names = list(folds)
    first = folds[names[0]]

    splits = {n: tuple(f.provenance["test_subjects"]) for n, f in folds.items()}
    if len(set(splits.values())) != 1:
        raise SystemExit(
            "arms hold out different subjects, so a paired contrast is impossible:\n  "
            + "\n  ".join(f"{n}: {s}" for n, s in splits.items())
        )
    vals = {n: tuple(f.provenance["val_subjects"]) for n, f in folds.items()}
    if len(set(vals.values())) != 1:
        raise SystemExit(
            "arms were model-selected on different validation subjects; the winner "
            "would partly reflect an easier early-stopping set:\n  "
            + "\n  ".join(f"{n}: {v}" for n, v in vals.items())
        )

    for name, fold in folds.items():
        if not np.array_equal(fold.subjects, first.subjects):
            raise SystemExit(f"{name}: evaluation windows differ from {names[0]}")
        if not np.allclose(fold.reference, first.reference, equal_nan=True):
            raise SystemExit(f"{name}: reference trajectories differ from {names[0]}")
        if not np.allclose(fold.persistence, first.persistence, equal_nan=True):
            raise SystemExit(f"{name}: persistence baseline differs from {names[0]}")
        best, cap = fold.provenance.get("best_epoch"), fold.provenance.get("epoch_cap")
        if best is not None and cap is not None and best >= cap:
            raise SystemExit(
                f"{name}: training stopped at the epoch cap ({best} of {cap}), so the "
                "cap chose the model rather than early stopping"
            )

    # The config fingerprint records the data *path*, which stays equal while the
    # file behind it is rebuilt. Arms trained months apart on the same path are
    # then a data contrast wearing the label of the ingredient under test.
    trained_on = {
        n: f.provenance.get("data_sha256_at_training")
        for n, f in folds.items()
        if f.provenance.get("data_sha256_at_training") is not None
    }
    if len(set(trained_on.values())) > 1:
        raise SystemExit(
            "arms were trained on different data, so the contrast would confound "
            f"{under_test} with the training set:\n  "
            + "\n  ".join(f"{n}: {s[:12]}" for n, s in trained_on.items())
            + "\nRetrain every arm on one data pack before comparing."
        )

    prints = {n: f.provenance["config_fingerprint"] for n, f in folds.items()}
    keys = set().union(*(set(p) for p in prints.values()))
    varied = {
        k for k in keys
        if len({json.dumps(p.get(k), sort_keys=True) for p in prints.values()}) > 1
    }
    try:
        allowed = UNDER_TEST_PRESETS[under_test]
    except KeyError:
        raise SystemExit(
            f"unknown ingredient under test {under_test!r}; "
            f"choose one of {sorted(UNDER_TEST_PRESETS)}"
        )
    confounded = sorted(varied - allowed)
    if confounded:
        raise SystemExit(
            "arms differ in settings that are not under test, so the contrast would "
            f"confound them with {under_test}: {confounded}"
        )
    if not varied:
        raise SystemExit(
            f"arms are identical in every fingerprinted setting, including {under_test}. "
            "A null result here would measure run-to-run noise, not the ingredient."
        )
    return {
        "under_test": under_test,
        "under_test_keys": sorted(allowed),
        "shared_test_subjects": list(splits[names[0]]),
        "shared_val_subjects": list(vals[names[0]]),
        "varied_fingerprint_keys": sorted(varied),
        "held_constant": sorted(keys - varied),
        "per_arm_settings_under_test": {
            n: {k: p.get(k) for k in sorted(varied)} for n, p in prints.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=MODEL:CONFIG",
                    help="repeatable; NAME is the label used in the output")
    ap.add_argument("--control", required=True,
                    help="name of the arm every other arm is contrasted against")
    ap.add_argument("--under-test", default="loss", choices=sorted(UNDER_TEST_PRESETS),
                    help="which ingredient the arms differ in; every other "
                         "fingerprinted setting must be held constant")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    folds: dict[str, Fold] = {}
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        model, _, config = rest.partition(":")
        if not (name and model and config):
            raise SystemExit(f"--arm must be NAME=MODEL:CONFIG, got {spec!r}")
        folds[name] = load_fold(Path(model), args.data, Path(config), name=name)
        print(f"loaded {name}: {folds[name].reference.shape[0]} windows, "
              f"subjects {folds[name].provenance['test_subjects']}")

    if args.control not in folds:
        raise SystemExit(f"--control {args.control!r} is not one of {list(folds)}")

    design = _check_arms_are_comparable(folds, under_test=args.under_test)
    print(f"comparable: varying {design['varied_fingerprint_keys']}")

    control_fold = folds[args.control]
    step = control_fold.step_minutes

    arms: dict[str, Any] = {
        "persistence": _arm_summary(control_fold, control_fold.persistence)
    }
    for name, fold in folds.items():
        arms[name] = _arm_summary(fold, fold.predicted)

    # One shifted control per non-control arm: does a constant offset of the
    # control reproduce whatever that arm gained?
    offsets: dict[str, float] = {}
    for name, fold in folds.items():
        if name == args.control:
            continue
        shifted, offset = _offset_control(control_fold.predicted, fold.predicted)
        offsets[name] = offset
        arms[f"{args.control}_shifted_to_{name}"] = _arm_summary(control_fold, shifted)

    contrasts: dict[str, Any] = {}
    for name, fold in folds.items():
        if name == args.control:
            continue
        shifted = control_fold.predicted + offsets[name]
        contrasts[f"{name}_vs_{args.control}_shifted"] = {
            "interpretation": (
                "the treatment arm against the control arm moved by a single "
                "constant. A difference near zero means the training change "
                "delivered a bias, which requires no training to obtain."
            ),
            "offset_mgdl": offsets[name],
            "metrics": _paired_contrast(fold, fold.predicted, shifted, step),
            "hypoglycemia_detection": _hypo_contrast(fold, fold.predicted, shifted),
        }
        contrasts[f"{name}_vs_{args.control}"] = {
            "metrics": _paired_contrast(fold, fold.predicted, control_fold.predicted, step),
            "hypoglycemia_detection": _hypo_contrast(
                fold, fold.predicted, control_fold.predicted
            ),
        }
    for name, fold in folds.items():
        contrasts[f"{name}_vs_persistence"] = {
            "metrics": _paired_contrast(fold, fold.predicted, fold.persistence, step),
            "hypoglycemia_detection": _hypo_contrast(
                fold, fold.predicted, fold.persistence
            ),
        }

    n_subjects = len(set(control_fold.subjects.tolist()))
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "question": (
            f"does changing the {args.under_test.replace('_', ' ')} change held-out "
            "forecast behaviour, on subjects no arm was trained or model-selected on"
        ),
        "estimand": "mean over subjects of the per-subject mean; pairs are not the unit",
        "control_arm": args.control,
        "design": design,
        "n_subjects": n_subjects,
        "n_pairs": int(control_fold.reference.shape[0]),
        "horizon_minutes": control_fold.provenance["horizon_minutes"],
        "inference": {
            "min_subjects_for_interval": MIN_SUBJECTS_FOR_INTERVAL,
            "intervals_reported": n_subjects >= MIN_SUBJECTS_FOR_INTERVAL,
            "note": (
                "with a shared split the subject count is fixed by that split; "
                "widening it requires retraining every arm on further splits"
            ),
        },
        "offset_controls_mgdl": offsets,
        "arms": arms,
        "contrasts": contrasts,
        "provenance": {n: f.provenance for n, f in folds.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\nwrote {args.out}  ({n_subjects} subjects, {payload['n_pairs']} pairs)")
    for name, summary in arms.items():
        dyn = summary["trend_dynamics"]
        print(f"  {name:22} hypo_sens={summary['hypoglycemia_detection']['sensitivity_pct']:5.1f}%  "
              f"MAE={summary['metrics']['absolute_error_mgdl']['estimate']:5.1f}  "
              f"zoneA={summary['metrics']['clarke_zone_a']['estimate']:5.1f}%  "
              f"rate_att={dyn['rate_attenuation']:.3f}  flat={dyn['flat_forecast']}")
    for label, block in contrasts.items():
        h = block["hypoglycemia_detection"].get("paired_treatment_minus_control")
        if h:
            print(f"  {label:34} hypo {h['estimate']:+6.2f} pp  "
                  f"consistent={h.get('consistent_across_subjects')}")


if __name__ == "__main__":
    main()
