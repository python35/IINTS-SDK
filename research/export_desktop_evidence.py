"""Export a real evaluation result for the desktop workbench to display.

The desktop app used to draw its Clarke error grid from random numbers with the
zone percentages written into the label by hand. This script produces the thing
those charts should have been showing all along: paired (reference, predicted)
values from a trained checkpoint evaluated on held-out subjects, with the
provenance needed to defend them.

Scientific choices made here, and why:

* Subjects come from the checkpoint's own ``test_subjects`` split, never the
  training or validation subjects. Validation subjects were used for model
  selection (early stopping), so an error grid over them is optimistic.
* The pair is the *endpoint* of the forecast horizon (one pair per window),
  not every step of every horizon. Reporting all horizon steps inflates n by
  the horizon length while adding almost no independent information, because
  consecutive steps of one forecast are near-perfectly correlated.
* A persistence baseline (carry the last observed glucose forward) is scored
  on exactly the same pairs. At clinically useful horizons persistence is hard
  to beat, so a grid without it cannot show whether the model contributes
  anything.
* Zone percentages are also reported per subject. With two held-out subjects
  the effective sample size is two, not tens of thousands; the pooled number
  alone would overstate the precision.

Usage
-----
    python research/export_desktop_evidence.py \
        --model models/ohio_t1dm_full_multimodal_seed7/predictor.pt \
        --data data_packs/ohio_merged.parquet \
        --out apps/iints-tauri/frontend/evidence/forecast_evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.util
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from iints.analysis.error_grid import clarke_error_grid
from iints.research.config import PredictorConfig, TrainingConfig
from iints.research.dataset import FeatureScaler, build_sequences, load_dataset
from iints.research.predictor import load_predictor

MAX_SCATTER_POINTS = 4000


def _load_meal_announcement_fn():
    """Borrow the evaluator's feature reconstruction rather than re-deriving it.

    The announced-meal feature is built by shifting carbohydrate intake earlier
    in time. Re-implementing that shift here would risk an off-by-one against
    training, which would silently invalidate every number this script prints.
    """
    path = Path(__file__).with_name("evaluate_predictor.py")
    spec = importlib.util.spec_from_file_location("_iints_evaluate_predictor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._apply_meal_announcement


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ega_payload(reference: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    result = clarke_error_grid(reference, predicted)
    pct = {z: float(result.percentages[z]) for z in "ABCDE"}
    return {
        "zone_percentages": pct,
        "zone_counts": {z: int(result.counts[z]) for z in "ABCDE"},
        "n_pairs": int(result.n_pairs),
        "clinically_acceptable_pct": pct["A"] + pct["B"],
        "hazardous_pct": pct["C"] + pct["D"] + pct["E"],
        "mae": float(np.mean(np.abs(predicted - reference))),
        "rmse": float(np.sqrt(np.mean((predicted - reference) ** 2))),
        "bias": float(np.mean(predicted - reference)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--config",
        type=Path,
        help="Training config YAML. Required when the checkpoint uses a "
             "reconstructed feature such as the announced-meal channel.",
    )
    args = ap.parse_args()

    report_path = args.model.parent / "training_report.json"
    if not report_path.exists():
        raise SystemExit(
            f"No training_report.json beside {args.model}. The held-out subject list "
            "is required; refusing to guess a split."
        )
    report = json.loads(report_path.read_text())

    test_subjects = [str(s) for s in report.get("test_subjects") or []]
    if not test_subjects:
        raise SystemExit(
            "Checkpoint reports no test_subjects. An error grid over training or "
            "validation subjects is not an out-of-sample result; refusing to export."
        )

    model, cfg = load_predictor(args.model)
    history_steps = int(cfg["history_steps"])
    horizon_steps = int(cfg["horizon_steps"])
    step_minutes = int(cfg.get("time_step_minutes", 5))
    feature_columns = list(cfg["feature_columns"])
    target_column = cfg["target_column"]
    glucose_idx = feature_columns.index(target_column)

    scaler = FeatureScaler.from_dict(cfg["scaler"]) if cfg.get("scaler") else None

    predictor_cfg = PredictorConfig(
        history_minutes=history_steps * step_minutes,
        horizon_minutes=horizon_steps * step_minutes,
        time_step_minutes=step_minutes,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    training_cfg = TrainingConfig()
    config_sha = None
    if args.config:
        config_sha = _sha256(args.config)
        expected = report.get("config_sha256")
        if expected and expected != config_sha:
            raise SystemExit(
                "Config checksum does not match the one recorded at training "
                f"time ({config_sha[:12]} vs {expected[:12]}). Evaluating with "
                "different preprocessing would invalidate the result."
            )
        raw = yaml.safe_load(args.config.read_text()) or {}
        if isinstance(raw.get("training"), dict):
            training_cfg = TrainingConfig(**raw["training"])

    df = load_dataset(args.data)
    df["subject_id"] = df["subject_id"].astype(str)
    apply_meal_announcement = _load_meal_announcement_fn()
    df = apply_meal_announcement(df, predictor_cfg, training_cfg)

    still_missing = [c for c in feature_columns if c not in df.columns]
    if still_missing:
        raise SystemExit(
            f"Features required by the checkpoint are absent after preprocessing: "
            f"{still_missing}. Pass the training --config used for this model."
        )
    missing = sorted(set(test_subjects) - set(df["subject_id"].unique()))
    if missing:
        raise SystemExit(f"Test subjects absent from {args.data}: {missing}")

    model.eval()
    per_subject: dict[str, dict[str, Any]] = {}
    ref_all, pred_all, persist_all, subj_all = [], [], [], []

    for subject in test_subjects:
        sub = df[df["subject_id"] == subject]
        # Built per subject so a window can never span two people, and with the
        # segment column so it can never span a recording gap either.
        X, y = build_sequences(
            sub,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            feature_columns=feature_columns,
            target_column=target_column,
            subject_column="subject_id",
            segment_column="segment" if "segment" in sub.columns else None,
        )
        if len(X) == 0:
            continue

        X_scaled = scaler.transform(X) if scaler is not None else X
        with torch.no_grad():
            out = model(torch.from_numpy(X_scaled.astype(np.float32))).numpy()

        reference = np.asarray(y, dtype=float)[:, -1]
        predicted = np.asarray(out, dtype=float).reshape(len(X), -1)[:, -1]
        persistence = np.asarray(X, dtype=float)[:, -1, glucose_idx]

        per_subject[subject] = {
            "n_pairs": int(len(reference)),
            "model": _ega_payload(reference, predicted),
            "persistence": _ega_payload(reference, persistence),
        }
        ref_all.append(reference)
        pred_all.append(predicted)
        persist_all.append(persistence)
        subj_all.append(np.full(len(reference), subject, dtype=object))

    if not ref_all:
        raise SystemExit("No usable sequences for the held-out subjects.")

    reference = np.concatenate(ref_all)
    predicted = np.concatenate(pred_all)
    persistence = np.concatenate(persist_all)
    subjects = np.concatenate(subj_all)

    # Thin the scatter for the UI only. Percentages are always computed on the
    # full set above, never on this subsample, so the chart cannot disagree
    # with the numbers printed beside it.
    rng = np.random.default_rng(0)
    if len(reference) > MAX_SCATTER_POINTS:
        keep = np.sort(rng.choice(len(reference), MAX_SCATTER_POINTS, replace=False))
    else:
        keep = np.arange(len(reference))

    subject_zone_a = [per_subject[s]["model"]["zone_percentages"]["A"] for s in per_subject]

    payload = {
        "schema": "iints.desktop.forecast_evidence/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "model_path": str(args.model),
            "model_sha256": _sha256(args.model),
            "data_path": str(args.data),
            "data_sha256": _sha256(args.data),
            "training_data_sha256": report.get("data_sha256"),
            "train_subjects": report.get("train_subjects"),
            "val_subjects": report.get("val_subjects"),
            "test_subjects": report.get("test_subjects"),
            "subject_level_split": report.get("subject_level_split"),
            "horizon_minutes": horizon_steps * step_minutes,
            "history_minutes": history_steps * step_minutes,
            "pair_definition": "forecast endpoint, one pair per window",
            "config_sha256": config_sha,
            # Disclosed because it is future information relative to the
            # prediction instant: the model is told about carbohydrates before
            # they are eaten. This is the standard announced-meal assumption in
            # closed-loop research, but a reader must be able to see it.
            "meal_announcement_minutes": training_cfg.meal_announcement_minutes,
        },
        "pooled": {
            "model": _ega_payload(reference, predicted),
            "persistence": _ega_payload(reference, persistence),
        },
        "per_subject": per_subject,
        "subject_level_zone_a": {
            "n_subjects": len(subject_zone_a),
            "mean": float(np.mean(subject_zone_a)),
            "min": float(np.min(subject_zone_a)),
            "max": float(np.max(subject_zone_a)),
        },
        "scatter": {
            "note": (
                "Subsample for display only; all percentages are computed on the "
                "full set of pairs."
            ),
            "n_shown": int(len(keep)),
            "reference": [round(float(v), 2) for v in reference[keep]],
            "predicted": [round(float(v), 2) for v in predicted[keep]],
            "persistence": [round(float(v), 2) for v in persistence[keep]],
            "subject": [str(s) for s in subjects[keep]],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    pooled = payload["pooled"]
    print(f"Wrote {args.out}")
    print(f"  horizon        : {payload['provenance']['horizon_minutes']} min")
    print(f"  held-out subj  : {test_subjects}")
    print(f"  pairs          : {pooled['model']['n_pairs']:,}")
    print(f"  model    Zone A: {pooled['model']['zone_percentages']['A']:.1f}%  "
          f"MAE {pooled['model']['mae']:.1f}  hazardous {pooled['model']['hazardous_pct']:.1f}%")
    print(f"  persistence     Zone A: {pooled['persistence']['zone_percentages']['A']:.1f}%  "
          f"MAE {pooled['persistence']['mae']:.1f}  "
          f"hazardous {pooled['persistence']['hazardous_pct']:.1f}%")


if __name__ == "__main__":
    main()
