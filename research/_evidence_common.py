"""Shared loading path for every evidence exporter.

One function, :func:`load_fold`, turns a checkpoint plus a dataset into the
held-out forecasts it produced. Both ``export_desktop_evidence.py`` (single
checkpoint, endpoint pairs, feeds the desktop app) and
``export_crossfold_evidence.py`` (several checkpoints, subject-level intervals)
go through it.

They share this module on purpose. Two exporters with their own copies of the
preprocessing would eventually disagree about what "the same evaluation" means,
and the disagreement would show up as two different numbers for one claim with
nothing to say which is right.

Invariants enforced here, so no caller can skip them:

* The held-out subject list comes from the checkpoint's ``training_report.json``
  and nowhere else. Inventing a split invalidates every number downstream.
* Held-out subjects must be disjoint from both the training and the validation
  subjects. Validation subjects were consumed by early stopping, so they are not
  out-of-sample either.
* Preprocessing is the training-time preprocessing, reused from
  ``evaluate_predictor.py`` rather than reimplemented. A one-step shift in a
  private reimplementation would silently invalidate everything.
* If a training config is supplied its checksum must match the one recorded at
  training time.
* Sequences are built per subject and per recording segment, so a window can
  never span two people or a gap in the recording.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from iints.analysis.clustered_inference import cluster_t_ci  # noqa: E402
from iints.analysis.error_grid import HAZARDOUS_ZONES, clarke_zones  # noqa: E402
from iints.analysis.prediction_accuracy import (  # noqa: E402
    HYPO_THRESHOLD_MGDL,
    classify_predictions,
)
from iints.research.config import PredictorConfig, TrainingConfig  # noqa: E402
from iints.research.dataset import FeatureScaler, build_sequences, load_dataset  # noqa: E402
from iints.research.evaluation import hypoglycemia_detection_report  # noqa: E402
from iints.research.predictor import load_predictor  # noqa: E402

#: Metrics where a HIGHER subject mean is the better outcome. Everything else
#: is an error rate, where lower is better. Stated once, here, so the sign of
#: every paired contrast is read from a table instead of re-derived at each use.
HIGHER_IS_BETTER = frozenset({"clarke_zone_a", "directional_accurate"})

__all__ = [
    "Fold",
    "load_fold",
    "sha256",
    "config_fingerprint",
    "interval",
    "per_pair_indicators",
    "hypo_sensitivity_by_subject",
    "HIGHER_IS_BETTER",
    "HYPO_THRESHOLD_MGDL",
]


def interval(values: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    """Cluster-level t interval over subjects, with the effective n attached."""
    iv = cluster_t_ci(values, subjects)
    out = iv.to_dict()
    # n_clusters IS the effective sample size here; name it so in the payload,
    # because 'n_observations' next to a pair count invites the wrong reading.
    out["n_subjects"] = iv.n_clusters
    out["n_pairs"] = iv.n_observations
    return out


def per_pair_indicators(reference: np.ndarray,
                        predicted: np.ndarray,
                        step_minutes: float) -> dict[str, np.ndarray]:
    """Per-pair 0/1 (or continuous) values whose subject means are the metrics.

    Everything is expressed per pair so that ``cluster_t_ci`` can collapse it to
    a subject mean itself. That keeps one tested implementation of the pooling
    instead of a second one written here.
    """
    ref_end, pred_end = reference[:, -1], predicted[:, -1]
    zones = np.asarray(clarke_zones(ref_end, pred_end), dtype=object)
    detail = classify_predictions(reference, predicted, step_minutes)
    return {
        "clarke_zone_a": (zones == "A").astype(float) * 100.0,
        "clarke_hazardous": np.isin(zones, HAZARDOUS_ZONES).astype(float) * 100.0,
        "directional_erroneous": (detail["label"] == "erroneous").astype(float) * 100.0,
        "directional_accurate": (detail["label"] == "accurate").astype(float) * 100.0,
        "reversed_trend": detail["reversed_trend"].astype(float) * 100.0,
        "absolute_error_mgdl": np.abs(pred_end - ref_end),
        "signed_error_mgdl": pred_end - ref_end,
    }


def hypo_sensitivity_by_subject(reference: np.ndarray,
                                predicted: np.ndarray,
                                subjects: np.ndarray) -> dict[str, float | None]:
    """Per-subject hypo detection sensitivity, reusing the SDK's own report.

    At a fixed forecast horizon the detection lead time of a correctly predicted
    event IS the horizon, so it is a property of the design rather than a
    measured quantity and is deliberately not reported as one.
    """
    out: dict[str, float | None] = {}
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        rep = hypoglycemia_detection_report(
            reference[mask, -1], predicted[mask, -1], threshold_mgdl=HYPO_THRESHOLD_MGDL
        )
        out[subject] = rep["sensitivity_pct"]
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _meal_announcement_fn():
    """Reuse the trainer's announced-meal reconstruction; never reimplement it."""
    path = Path(__file__).with_name("evaluate_predictor.py")
    spec = importlib.util.spec_from_file_location("_iints_evaluate_predictor", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._apply_meal_announcement


@dataclass
class Fold:
    """One checkpoint evaluated on the subjects it never saw.

    Profiles are full trajectories ``[n_windows, horizon_steps]``, not just the
    endpoint. The rate-aware analysis needs the shape of the forecast, not one
    number from it.
    """

    name: str
    reference: np.ndarray
    predicted: np.ndarray
    persistence: np.ndarray
    subjects: np.ndarray
    step_minutes: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(self.reference.shape[0])

    def endpoints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The last step of each trajectory: one pair per window."""
        return self.reference[:, -1], self.predicted[:, -1], self.persistence[:, -1]


def config_fingerprint(config_path: Path) -> dict[str, Any]:
    """The settings that must agree before two checkpoints may be pooled.

    Seed and subject split are deliberately absent: those are exactly what is
    allowed to differ between folds. Epoch cap is absent too, because early
    stopping decides the effective number of epochs — the caller should check
    that ``best_epoch`` sits below the cap instead.
    """
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    predictor = raw.get("predictor", {}) or {}
    training = raw.get("training", {}) or {}
    return {
        "history_minutes": predictor.get("history_minutes"),
        "horizon_minutes": predictor.get("horizon_minutes"),
        "feature_columns": list(predictor.get("feature_columns") or []),
        "target_column": predictor.get("target_column"),
        "predict_delta": predictor.get("predict_delta", False),
        "loss": training.get("loss"),
        "hidden_size": training.get("hidden_size"),
        "num_layers": training.get("num_layers"),
        "dropout": training.get("dropout"),
        "normalization": training.get("normalization"),
        "learning_rate": training.get("learning_rate"),
        "batch_size": training.get("batch_size"),
        "meal_announcement_minutes": training.get("meal_announcement_minutes"),
        "meal_announcement_grams": training.get("meal_announcement_grams"),
    }


def load_fold(model_path: Path,
              data_path: Path,
              config_path: Path | None = None,
              *,
              name: str | None = None) -> Fold:
    """Evaluate one checkpoint on its own held-out subjects."""
    model_path, data_path = Path(model_path), Path(data_path)
    report_path = model_path.parent / "training_report.json"
    if not report_path.exists():
        raise SystemExit(
            f"No training_report.json beside {model_path}. The held-out subject "
            "list is required; refusing to guess a split."
        )
    report = json.loads(report_path.read_text())

    test_subjects = [str(s) for s in report.get("test_subjects") or []]
    if not test_subjects:
        raise SystemExit(
            f"{model_path.parent.name} reports no test_subjects. An error grid over "
            "training or validation subjects is not an out-of-sample result."
        )
    train_subjects = {str(s) for s in report.get("train_subjects") or []}
    val_subjects = {str(s) for s in report.get("val_subjects") or []}
    leaked = sorted(set(test_subjects) & (train_subjects | val_subjects))
    if leaked:
        raise SystemExit(
            f"{model_path.parent.name}: subjects {leaked} appear in both the held-out "
            "list and the training or validation list. Validation subjects were "
            "consumed by early stopping and are not out-of-sample either."
        )

    model, cfg = load_predictor(model_path)
    history_steps = int(cfg["history_steps"])
    horizon_steps = int(cfg["horizon_steps"])
    step_minutes = float(cfg.get("time_step_minutes", 5))
    feature_columns = list(cfg["feature_columns"])
    target_column = cfg["target_column"]
    predict_delta = bool(cfg.get("predict_delta", False))
    glucose_idx = feature_columns.index(target_column)
    scaler = FeatureScaler.from_dict(cfg["scaler"]) if cfg.get("scaler") else None

    predictor_cfg = PredictorConfig(
        history_minutes=int(history_steps * step_minutes),
        horizon_minutes=int(horizon_steps * step_minutes),
        time_step_minutes=int(step_minutes),
        feature_columns=feature_columns,
        target_column=target_column,
        predict_delta=predict_delta,
    )
    training_cfg = TrainingConfig()
    config_sha = None
    fingerprint = None
    if config_path is not None:
        config_path = Path(config_path)
        config_sha = sha256(config_path)
        expected = report.get("config_sha256")
        if expected and expected != config_sha:
            raise SystemExit(
                f"{model_path.parent.name}: config checksum does not match the one "
                f"recorded at training time ({config_sha[:12]} vs {expected[:12]}). "
                "Evaluating with different preprocessing would invalidate the result."
            )
        raw = yaml.safe_load(config_path.read_text()) or {}
        if isinstance(raw.get("training"), dict):
            training_cfg = TrainingConfig(**raw["training"])
        fingerprint = config_fingerprint(config_path)

    df = load_dataset(data_path)
    df["subject_id"] = df["subject_id"].astype(str)
    df = _meal_announcement_fn()(df, predictor_cfg, training_cfg)

    still_missing = [c for c in feature_columns if c not in df.columns]
    if still_missing:
        raise SystemExit(
            f"Features required by the checkpoint are absent after preprocessing: "
            f"{still_missing}. Pass the training --config used for this model."
        )
    absent = sorted(set(test_subjects) - set(df["subject_id"].unique()))
    if absent:
        raise SystemExit(f"Held-out subjects absent from {data_path}: {absent}")

    model.eval()
    refs, preds, persists, subs = [], [], [], []
    for subject in test_subjects:
        sub = df[df["subject_id"] == subject]
        X, y = build_sequences(
            sub,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            feature_columns=feature_columns,
            target_column=target_column,
            subject_column="subject_id",
            segment_column="segment" if "segment" in sub.columns else None,
            predict_delta=predict_delta,
        )
        if len(X) == 0:
            continue
        X_scaled = scaler.transform(X) if scaler is not None else X
        with torch.no_grad():
            out = model(torch.from_numpy(X_scaled.astype(np.float32))).numpy()

        last = np.asarray(X, dtype=float)[:, -1, glucose_idx]
        if predict_delta:
            reference = (last[:, None] + np.asarray(y, dtype=float)).reshape(len(X), -1)
            predicted = (last[:, None] + np.asarray(out, dtype=float)).reshape(len(X), -1)
        else:
            reference = np.asarray(y, dtype=float).reshape(len(X), -1)
            predicted = np.asarray(out, dtype=float).reshape(len(X), -1)

        # Persistence carries the last observed glucose forward unchanged. As a
        # trajectory that is a flat line, which is the honest representation:
        # this baseline cannot express a trend, and the rate-aware analysis
        # should be able to see that.
        persistence = np.repeat(last[:, None], predicted.shape[1], axis=1)

        refs.append(reference)
        preds.append(predicted)
        persists.append(persistence)
        subs.append(np.full(len(X), subject, dtype=object))

    if not refs:
        raise SystemExit(f"No usable sequences for the held-out subjects of {model_path}.")

    return Fold(
        name=name or model_path.parent.name,
        reference=np.concatenate(refs),
        predicted=np.concatenate(preds),
        persistence=np.concatenate(persists),
        subjects=np.concatenate(subs),
        step_minutes=step_minutes,
        provenance={
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "data_path": str(data_path),
            "data_sha256": sha256(data_path),
            "data_sha256_at_training": report.get("data_sha256"),
            "config_path": str(config_path) if config_path else None,
            "config_sha256": config_sha,
            "config_fingerprint": fingerprint,
            "seed": report.get("seed"),
            "best_epoch": report.get("best_epoch"),
            "epoch_cap": report.get("epochs"),
            "test_subjects": test_subjects,
            "val_subjects": sorted(val_subjects),
            "train_subjects": sorted(train_subjects),
            "history_minutes": int(history_steps * step_minutes),
            "horizon_minutes": int(horizon_steps * step_minutes),
            "step_minutes": step_minutes,
            "meal_announcement_minutes": getattr(training_cfg, "meal_announcement_minutes", None),
        },
    )
