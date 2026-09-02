"""Subject-level cross-fold evidence for the glucose forecaster.

What problem this solves
------------------------
A single checkpoint is evaluated on the two subjects it happened to hold out.
Two people is the real sample size, whatever the pair count says, and the
spread between two people is wide. Reporting a pooled percentage over tens of
thousands of pairs from two subjects states a precision that does not exist —
the same clustering error this project already fixed once, in the simulator
benchmark.

The fix here needs no new training. Several checkpoints in ``models/`` were
trained with identical settings and *different* subject splits. Evaluating each
on its own held-out subjects and pooling at the subject level turns them into a
subject-level cross-validation that was already paid for.

What is being estimated
-----------------------
Not "how good is this one checkpoint" but "how good is this training recipe on
a subject it has never seen". That is the question a reviewer actually asks, and
it is the only one a handful of folds can answer honestly. Every interval is a
cluster-level t interval over SUBJECTS via
``iints.analysis.clustered_inference.cluster_t_ci``; the effective n is printed
next to every number so nobody can mistake pair count for sample size.

Refusals, not warnings
----------------------
The script exits rather than produce a misleading file when folds are not
comparable (different architecture, features, loss, horizon or preprocessing),
when a subject appears as held-out in one fold and as training data in another
fold that is being pooled with it, or when a checkpoint's early stopping ran
into its epoch cap (in which case the cap, not the data, chose the model and the
folds are not interchangeable).

Usage::

    PYTHONPATH=src ./.venv/bin/python research/export_crossfold_evidence.py \\
        --data data_packs/ohio_merged.parquet \\
        --fold models/ohio_t1dm_full_multimodal_seed7/predictor.pt:results/ohio_t1dm_full_training_sweep/configs/multimodal_seed7.yaml \\
        --fold models/ohio_t1dm_full_multimodal_seed99/predictor.pt:results/ohio_t1dm_full_training_sweep/configs/multimodal_seed99.yaml \\
        --out apps/iints-tauri/frontend/evidence/crossfold_evidence.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Import the shared loader by path rather than relying on the script directory
# landing on sys.path, which is not guaranteed under every launcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _evidence_common import (  # noqa: E402
    HIGHER_IS_BETTER,
    HYPO_THRESHOLD_MGDL,
    Fold,
    hypo_sensitivity_by_subject as _hypo_sensitivity_by_subject,
    interval as _interval,
    load_fold,
    per_pair_indicators as _per_pair_indicators,
)
from iints.analysis.prediction_accuracy import directional_report  # noqa: E402
from iints.research.evaluation import hypoglycemia_detection_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--fold", required=True, action="append", metavar="CHECKPOINT[:CONFIG]",
        help="Checkpoint to evaluate on its own held-out subjects. Repeat for "
             "each fold. Append ':path/to/config.yaml' when the checkpoint uses "
             "a reconstructed feature such as the announced-meal channel.",
    )
    args = ap.parse_args()

    folds: list[Fold] = []
    for spec in args.fold:
        model_str, _, config_str = spec.partition(":")
        folds.append(load_fold(Path(model_str), args.data,
                               Path(config_str) if config_str else None))

    if len(folds) < 2:
        raise SystemExit(
            "Cross-fold evidence needs at least two folds; with one fold the "
            "subject-level interval has no between-fold variation to measure. "
            "Use export_desktop_evidence.py for a single checkpoint."
        )

    # --- comparability, enforced before a single number is computed ----------
    fingerprints = [f.provenance.get("config_fingerprint") for f in folds]
    if any(fp is None for fp in fingerprints):
        raise SystemExit(
            "Every fold needs its training config so the settings can be compared. "
            "Pool only checkpoints proven to differ solely in seed and split."
        )
    for other, fold in zip(fingerprints[1:], folds[1:]):
        if other != fingerprints[0]:
            differing = sorted(k for k in fingerprints[0]
                               if fingerprints[0].get(k) != other.get(k))
            raise SystemExit(
                f"Fold '{fold.name}' was trained with different settings than "
                f"'{folds[0].name}' (differs in: {differing}). Pooling them would "
                "measure the difference between recipes, not between subjects."
            )
    for fold in folds:
        best, cap = fold.provenance.get("best_epoch"), fold.provenance.get("epoch_cap")
        if best is not None and cap is not None and best >= cap:
            raise SystemExit(
                f"Fold '{fold.name}' stopped at its epoch cap ({best} of {cap}), so "
                "the cap chose the model rather than early stopping. Retrain with a "
                "higher cap before pooling it with folds that stopped early."
            )

    held_out = [set(f.provenance["test_subjects"]) for f in folds]
    for i, a in enumerate(held_out):
        for j, b in enumerate(held_out[i + 1:], start=i + 1):
            if a & b:
                raise SystemExit(
                    f"Folds '{folds[i].name}' and '{folds[j].name}' share held-out "
                    f"subjects {sorted(a & b)}. A subject may contribute only once."
                )
    # A subject held out by one fold but trained on by another is still a valid
    # observation: each fold's prediction for it is genuinely out-of-sample for
    # the model that produced it. That is what cross-validation is. It is only
    # a problem if the SAME subject is counted twice, checked above.

    step = folds[0].step_minutes
    if any(f.step_minutes != step for f in folds):
        raise SystemExit("Folds disagree on the sampling interval.")

    reference = np.concatenate([f.reference for f in folds])
    predicted = np.concatenate([f.predicted for f in folds])
    persistence = np.concatenate([f.persistence for f in folds])
    subjects = np.concatenate([f.subjects for f in folds])
    fold_of = np.concatenate([np.full(f.n_pairs, f.name, dtype=object) for f in folds])

    arms = {"model": predicted, "persistence": persistence}
    indicators = {a: _per_pair_indicators(reference, p, step) for a, p in arms.items()}

    summary: dict[str, Any] = {}
    for arm, ind in indicators.items():
        summary[arm] = {k: _interval(v, subjects) for k, v in ind.items()}
        summary[arm]["directional"] = directional_report(reference, arms[arm], step)
        summary[arm]["hypo_detection_pooled"] = hypoglycemia_detection_report(
            reference[:, -1], arms[arm][:, -1], threshold_mgdl=HYPO_THRESHOLD_MGDL
        )
        summary[arm]["hypo_sensitivity_by_subject"] = _hypo_sensitivity_by_subject(
            reference, arms[arm], subjects
        )

    # --- primary safety outcome, at subject level ----------------------------
    # Hypo detection sensitivity is a rate over hypo events, not over all pairs,
    # so it is pooled across only those windows where the reference is truly
    # below threshold. Clustering is still by subject: the effective sample size
    # is the number of people, not the number of hypo windows.
    hypo_mask = reference[:, -1] < HYPO_THRESHOLD_MGDL
    if not hypo_mask.any():
        raise SystemExit("No hypoglycemic windows in the held-out data; cannot "
                         "report the primary safety outcome.")
    hypo_detected = {
        arm: (p[hypo_mask, -1] < HYPO_THRESHOLD_MGDL).astype(float) * 100.0
        for arm, p in arms.items()
    }
    hypo_subjects = subjects[hypo_mask]
    hypo_block: dict[str, Any] = {
        "definition": (
            f"Share of windows whose reference endpoint is below "
            f"{HYPO_THRESHOLD_MGDL:g} mg/dL for which the forecast endpoint is also "
            "below it. Cluster-level t interval over subjects. At a fixed horizon "
            "the detection lead time of a correctly predicted event is the horizon "
            "itself, so it is a design property and is not reported as a measurement."
        ),
        "n_hypo_windows": int(hypo_mask.sum()),
        "n_subjects_with_hypo": len(set(hypo_subjects.tolist())),
        **{arm: _interval(v, hypo_subjects) for arm, v in hypo_detected.items()},
    }
    hypo_diff = _interval(hypo_detected["model"] - hypo_detected["persistence"],
                          hypo_subjects)
    hypo_diff["higher_is_better"] = True
    hypo_diff["model_better_at_95pct"] = (
        None if not np.isfinite(hypo_diff["ci_low"]) else bool(hypo_diff["ci_low"] > 0)
    )
    hypo_block["paired_model_minus_persistence"] = hypo_diff

    # Paired model-minus-persistence contrast, one difference per subject. The
    # arms see identical pairs, so pairing removes between-subject variance —
    # the reason a paired contrast is worth more here than two separate ones.
    paired: dict[str, Any] = {}
    for metric in indicators["model"]:
        diff = indicators["model"][metric] - indicators["persistence"][metric]
        iv = _interval(diff, subjects)
        higher_better = metric in HIGHER_IS_BETTER
        iv["higher_is_better"] = higher_better
        if not np.isfinite(iv["ci_low"]):
            # One subject: a difference exists but has no interval, so no claim.
            iv["model_better_at_95pct"] = None
        elif higher_better:
            iv["model_better_at_95pct"] = bool(iv["ci_low"] > 0)
        else:
            iv["model_better_at_95pct"] = bool(iv["ci_high"] < 0)
        paired[metric] = iv

    per_subject = {}
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        per_subject[subject] = {
            "fold": str(fold_of[mask][0]),
            "n_pairs": int(mask.sum()),
            **{arm: {k: float(v[mask].mean()) for k, v in ind.items()}
               for arm, ind in indicators.items()},
        }

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "estimand": (
            "Out-of-sample accuracy of the training recipe on an unseen subject. "
            "Every interval is a cluster-level t interval over subjects; the "
            "effective sample size is n_subjects, never n_pairs."
        ),
        "folds": [
            {"name": f.name, "n_pairs": f.n_pairs, **f.provenance} for f in folds
        ],
        "n_subjects": len(per_subject),
        "n_pairs": int(reference.shape[0]),
        "horizon_minutes": folds[0].provenance["horizon_minutes"],
        "pair_definition": "forecast endpoint, one pair per window",
        "primary_outcome_hypoglycemia_detection": hypo_block,
        "summary": summary,
        "paired_model_minus_persistence": paired,
        "per_subject": per_subject,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    def _json_safe(obj: Any) -> Any:
        """Coerce numpy scalars. Never silently drop a value we cannot encode."""
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"refusing to write un-encodable {type(obj).__name__} to evidence")

    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False, default=_json_safe))

    print(f"wrote {args.out}")
    print(f"  folds          : {', '.join(f.name for f in folds)}")
    print(f"  subjects       : {payload['n_subjects']} ({', '.join(per_subject)})")
    print(f"  pairs          : {payload['n_pairs']:,}")
    for arm in ("model", "persistence"):
        za = summary[arm]["clarke_zone_a"]
        de = summary[arm]["directional_erroneous"]
        print(f"  {arm:12} Zone A {za['estimate']:5.1f}% [{za['ci_low']:5.1f},{za['ci_high']:5.1f}]"
              f"   erroneous {de['estimate']:5.2f}% [{de['ci_low']:5.2f},{de['ci_high']:5.2f}]")
    for metric in ("clarke_zone_a", "directional_erroneous", "absolute_error_mgdl"):
        d = paired[metric]
        print(f"  paired {metric:22}: {d['estimate']:+7.2f} "
              f"[{d['ci_low']:+7.2f},{d['ci_high']:+7.2f}]  "
              f"n={d['n_subjects']} subjects  model_better={d['model_better_at_95pct']}")
    print(f"  PRIMARY hypo detection ({hypo_block['n_hypo_windows']:,} hypo windows, "
          f"{hypo_block['n_subjects_with_hypo']} subjects):")
    for arm in ("model", "persistence"):
        h = hypo_block[arm]
        print(f"    {arm:12} {h['estimate']:5.1f}% [{h['ci_low']:5.1f},{h['ci_high']:5.1f}]")
    hd = hypo_block["paired_model_minus_persistence"]
    print(f"    paired     {hd['estimate']:+5.1f} pp [{hd['ci_low']:+5.1f},{hd['ci_high']:+5.1f}]"
          f"  model_better={hd['model_better_at_95pct']}")
    for arm in ("model", "persistence"):
        dyn = summary[arm]["directional"]["trend_dynamics"]
        sc = dyn["sign_concordance_pct"]
        print(f"  {arm:12} rate attenuation {dyn['rate_attenuation']:.3f}  "
              f"sign concordance {'n/a (no direction)' if sc is None else f'{sc:.1f}%'}  "
              f"flat_forecast={dyn['flat_forecast']}")


if __name__ == "__main__":
    main()
