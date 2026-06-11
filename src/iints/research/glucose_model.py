from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from iints.research.dataset import compute_dataset_lineage, load_dataset, save_dataset
from iints.research.config import PredictorConfig, TrainingConfig
from iints.research.dataset import build_sequences
from iints.research.evaluation import (
    forecast_error_report,
    hypoglycemia_detection_report,
    uncertainty_reliability_report,
)
from iints.research.forecasting import PhysiologyAwareBaseline
from iints.research.predictor import LastValueBaseline, LinearTrendBaseline, load_predictor_service

GLUCOSE_MODEL_ID = "iints-glucose-forecast-v0"
GLUCOSE_MODEL_CARD_VERSION = 1

GLUCOSE_MODEL_FEATURE_COLUMNS = [
    "glucose_actual_mgdl",
    "glucose_trend_mgdl_min",
    "patient_iob_units",
    "patient_cob_grams",
    "delivered_insulin_units",
    "carb_intake_grams",
    "effective_isf",
    "effective_icr",
    "effective_basal_rate_u_per_hr",
    "exercise_intensity",
    "stress_intensity",
    "steps",
    "heart_rate",
    "time_of_day_sin",
    "time_of_day_cos",
    "glucagon_mg",
    "haaf_memory",
]

_ALIAS_COLUMNS: Mapping[str, Sequence[str]] = {
    "glucose_actual_mgdl": (
        "glucose_actual_mgdl",
        "glucose_to_algo_mgdl",
        "glucose_mgdl",
        "sensor_glucose_mgdl",
        "cgm_mgdl",
        "glucose",
        "bg",
    ),
    "time_minutes": ("time_minutes", "minutes", "minute", "t", "timestamp_minutes"),
    "carb_intake_grams": ("carb_intake_grams", "carbs", "carbs_g", "meal_carbs", "carbohydrates"),
    "delivered_insulin_units": (
        "delivered_insulin_units",
        "insulin",
        "bolus_units",
        "insulin_units",
        "total_insulin_units",
    ),
    "patient_iob_units": ("patient_iob_units", "iob", "insulin_on_board"),
    "patient_cob_grams": ("patient_cob_grams", "cob", "carbs_on_board"),
    "effective_isf": ("effective_isf", "isf", "insulin_sensitivity_factor"),
    "effective_icr": ("effective_icr", "icr", "insulin_carb_ratio"),
    "effective_basal_rate_u_per_hr": ("effective_basal_rate_u_per_hr", "basal_rate", "basal_u_hr"),
    "exercise_intensity": ("exercise_intensity", "activity_intensity", "exercise"),
    "stress_intensity": ("stress_intensity", "illness_intensity", "stress"),
    "steps": ("steps", "step_count"),
    "heart_rate": ("heart_rate", "hr", "heartrate"),
    "glucagon_mg": ("glucagon_mg", "delivered_glucagon_mg", "glucagon_dose_mg"),
    "haaf_memory": ("haaf_memory", "haaf_state", "hypo_awareness_failure"),
}

_DEFAULT_FEATURE_VALUES: Mapping[str, float] = {
    "glucose_trend_mgdl_min": 0.0,
    "patient_iob_units": 0.0,
    "patient_cob_grams": 0.0,
    "delivered_insulin_units": 0.0,
    "carb_intake_grams": 0.0,
    "effective_isf": 50.0,
    "effective_icr": 10.0,
    "effective_basal_rate_u_per_hr": 0.8,
    "exercise_intensity": 0.0,
    "stress_intensity": 0.0,
    "steps": 0.0,
    "heart_rate": 0.0,
    "time_of_day_sin": 0.0,
    "time_of_day_cos": 1.0,
    "glucagon_mg": 0.0,
    "haaf_memory": 0.0,
}


@dataclass(frozen=True)
class GlucoseTrainingPack:
    dataset_path: Path
    config_path: Path
    manifest_path: Path
    model_intent_path: Path
    row_count: int
    subject_count: int
    source_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "config_path": str(self.config_path),
            "manifest_path": str(self.manifest_path),
            "model_intent_path": str(self.model_intent_path),
            "row_count": self.row_count,
            "subject_count": self.subject_count,
            "source_count": self.source_count,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    return parsed.replace([np.inf, -np.inf], np.nan).fillna(default)


def _derive_time_minutes(df: pd.DataFrame, *, time_step_minutes: int) -> pd.Series:
    existing = _first_existing(df, _ALIAS_COLUMNS["time_minutes"])
    if existing is not None:
        return _safe_numeric(df[existing], 0.0)
    timestamp_col = _first_existing(df, ("timestamp", "datetime", "time", "date_time", "Timestamp"))
    if timestamp_col is not None:
        parsed = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        if parsed.notna().any():
            first = parsed.dropna().iloc[0]
            minutes = (parsed - first).dt.total_seconds() / 60.0
            return minutes.ffill().fillna(0.0)
    return pd.Series(np.arange(len(df), dtype=float) * float(time_step_minutes), index=df.index)


def _copy_alias_or_default(df: pd.DataFrame, canonical: str, default: float) -> pd.Series:
    source = _first_existing(df, _ALIAS_COLUMNS.get(canonical, (canonical,)))
    if source is None:
        return pd.Series(default, index=df.index, dtype=float)
    return _safe_numeric(df[source], default)


def standardize_glucose_forecast_frame(
    df: pd.DataFrame,
    *,
    source_label: str,
    time_step_minutes: int = 5,
    subject_prefix: Optional[str] = None,
    max_gap_multiplier: float = 2.5,
) -> pd.DataFrame:
    """Normalize one real or simulated glucose dataframe into the v0 forecast schema."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    out = pd.DataFrame(index=df.index)
    out["time_minutes"] = _derive_time_minutes(df, time_step_minutes=time_step_minutes)
    out["glucose_actual_mgdl"] = _copy_alias_or_default(df, "glucose_actual_mgdl", np.nan)
    if out["glucose_actual_mgdl"].isna().all():
        raise ValueError("No glucose column found. Expected one of: " + ", ".join(_ALIAS_COLUMNS["glucose_actual_mgdl"]))
    out["glucose_actual_mgdl"] = out["glucose_actual_mgdl"].interpolate(limit_direction="both").clip(35.0, 450.0)

    subject_col = _first_existing(df, ("subject_id", "patient_id", "subject", "patient", "id"))
    if subject_col is None:
        subjects = pd.Series("subject_000", index=df.index, dtype=str)
    else:
        subjects = df[subject_col].fillna("unknown").astype(str)
    prefix = subject_prefix or source_label
    out["subject_id"] = subjects.map(lambda value: f"{prefix}:{value}")

    source_segment = _first_existing(df, ("segment", "segment_id", "session_id", "day"))
    if source_segment is None:
        gap = out.groupby("subject_id", observed=False)["time_minutes"].diff().fillna(float(time_step_minutes))
        segment_index = gap.gt(float(time_step_minutes) * max_gap_multiplier).groupby(out["subject_id"], observed=False).cumsum()
        out["segment"] = out["subject_id"] + ":seg" + segment_index.astype(int).astype(str)
    else:
        out["segment"] = out["subject_id"] + ":" + df[source_segment].fillna("segment_0").astype(str)

    for column in GLUCOSE_MODEL_FEATURE_COLUMNS:
        if column == "glucose_actual_mgdl":
            continue
        if column == "glucose_trend_mgdl_min":
            existing = _first_existing(df, _ALIAS_COLUMNS.get(column, (column,)))
            if existing is not None:
                out[column] = _safe_numeric(df[existing], 0.0)
            else:
                delta_g = out.groupby(["subject_id", "segment"], observed=False)["glucose_actual_mgdl"].diff()
                delta_t = out.groupby(["subject_id", "segment"], observed=False)["time_minutes"].diff()
                out[column] = (delta_g / delta_t.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            continue
        if column in {"time_of_day_sin", "time_of_day_cos"}:
            minutes_of_day = np.mod(out["time_minutes"].to_numpy(dtype=float), 1440.0)
            angle = (2.0 * np.pi * minutes_of_day) / 1440.0
            out["time_of_day_sin"] = np.sin(angle)
            out["time_of_day_cos"] = np.cos(angle)
            continue
        out[column] = _copy_alias_or_default(df, column, _DEFAULT_FEATURE_VALUES[column])

    out["source_dataset"] = source_label
    out["source_row_index"] = np.arange(len(out), dtype=int)
    out = out.sort_values(["subject_id", "segment", "time_minutes"]).reset_index(drop=True)
    numeric_cols = ["time_minutes", *GLUCOSE_MODEL_FEATURE_COLUMNS]
    for column in numeric_cols:
        out[column] = _safe_numeric(out[column], _DEFAULT_FEATURE_VALUES.get(column, 0.0))
    return out


def glucose_model_config_payload(
    *,
    profile: str = "long",
    history_minutes: int = 360,
    horizon_minutes: int = 120,
    time_step_minutes: int = 5,
    feature_columns: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Return a training config tuned for a dedicated glucose forecast model."""
    normalized = profile.strip().lower().replace("_", "-")
    presets = {
        "smoke": {"epochs": 2, "batch_size": 64, "hidden_size": 32, "num_layers": 1, "dropout": 0.05},
        "quick": {"epochs": 12, "batch_size": 128, "hidden_size": 64, "num_layers": 2, "dropout": 0.10},
        "long": {"epochs": 120, "batch_size": 256, "hidden_size": 128, "num_layers": 2, "dropout": 0.15},
        "paper": {"epochs": 220, "batch_size": 256, "hidden_size": 160, "num_layers": 3, "dropout": 0.18},
    }
    if normalized not in presets:
        raise ValueError(f"Unknown glucose-model profile: {profile!r}. Use smoke, quick, long, or paper.")
    training = presets[normalized]
    return {
        "predictor": {
            "history_minutes": int(history_minutes),
            "horizon_minutes": int(horizon_minutes),
            "time_step_minutes": int(time_step_minutes),
            "feature_columns": list(feature_columns or GLUCOSE_MODEL_FEATURE_COLUMNS),
            "target_column": "glucose_actual_mgdl",
        },
        "training": {
            "epochs": training["epochs"],
            "batch_size": training["batch_size"],
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "hidden_size": training["hidden_size"],
            "num_layers": training["num_layers"],
            "dropout": training["dropout"],
            "subject_level_split": True,
            "validation_split": 0.15,
            "test_split": 0.15,
            "seed": 42,
            "normalization": "robust",
            "loss": "pinn",
            "band_weighted_low_threshold": 70.0,
            "band_weighted_high_threshold": 180.0,
            "band_weighted_low_weight": 2.5,
            "band_weighted_high_weight": 1.6,
            "band_weighted_max_weight": 5.0,
            "pinn_lambda": 0.5,
            "pinn_max_roc": 10.0,
            "early_stopping_patience": 18 if normalized in {"long", "paper"} else 5,
            "early_stopping_min_delta": 0.0005,
        },
        "iints_glucose_model": {
            "model_id": GLUCOSE_MODEL_ID,
            "profile": normalized,
            "task": "multi-horizon glucose forecasting",
            "research_only": True,
            "not_for_treatment": True,
        },
    }


def write_glucose_model_config(path: Path, **kwargs: Any) -> dict[str, Any]:
    payload = glucose_model_config_payload(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return payload


def _source_summary(df: pd.DataFrame, *, label: str, path: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "sha256": _sha256_file(path),
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "segments": int(df["segment"].nunique()) if "segment" in df.columns else 0,
        "glucose_mean_mgdl": float(df["glucose_actual_mgdl"].mean()),
        "glucose_min_mgdl": float(df["glucose_actual_mgdl"].min()),
        "glucose_max_mgdl": float(df["glucose_actual_mgdl"].max()),
    }


def build_glucose_training_pack(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    labels: Optional[Sequence[str]] = None,
    output_format: str = "csv",
    profile: str = "long",
    history_minutes: int = 360,
    horizon_minutes: int = 120,
    time_step_minutes: int = 5,
) -> GlucoseTrainingPack:
    if not input_paths:
        raise ValueError("At least one --input dataset path is required.")
    if labels is not None and len(labels) != len(input_paths):
        raise ValueError("If labels are provided, their count must match input_paths.")
    normalized_format = output_format.strip().lower()
    if normalized_format not in {"csv", "parquet"}:
        raise ValueError("output_format must be csv or parquet")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for index, raw_path in enumerate(input_paths):
        path = raw_path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        label = labels[index] if labels is not None else path.stem.replace(" ", "_")
        raw_df = load_dataset(path)
        frame = standardize_glucose_forecast_frame(
            raw_df,
            source_label=label,
            time_step_minutes=time_step_minutes,
            subject_prefix=label,
        )
        frames.append(frame)
        sources.append(_source_summary(frame, label=label, path=path))

    merged = pd.concat(frames, ignore_index=True).sort_values(["source_dataset", "subject_id", "segment", "time_minutes"])
    merged = merged.reset_index(drop=True)
    dataset_path = output_dir / ("glucose_training_dataset.parquet" if normalized_format == "parquet" else "glucose_training_dataset.csv")
    save_dataset(merged, dataset_path)

    config_path = output_dir / "glucose_model_config.yaml"
    config_payload = write_glucose_model_config(
        config_path,
        profile=profile,
        history_minutes=history_minutes,
        horizon_minutes=horizon_minutes,
        time_step_minutes=time_step_minutes,
    )
    lineage = compute_dataset_lineage(merged, source_path=dataset_path)
    manifest = {
        "schema_version": "iints_glucose_model_dataset_v1",
        "model_id": GLUCOSE_MODEL_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "row_count": int(len(merged)),
        "subject_count": int(merged["subject_id"].nunique()),
        "segment_count": int(merged["segment"].nunique()),
        "source_count": len(sources),
        "sources": sources,
        "feature_columns": list(GLUCOSE_MODEL_FEATURE_COLUMNS),
        "target_column": "glucose_actual_mgdl",
        "history_minutes": int(history_minutes),
        "horizon_minutes": int(horizon_minutes),
        "time_step_minutes": int(time_step_minutes),
        "lineage": lineage,
        "privacy": {
            "raw_private_data_included": False,
            "note": "This manifest may contain local source paths. Do not publish it directly if those paths are private.",
        },
        "training_config": config_payload,
    }
    manifest_path = output_dir / "glucose_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    intent_path = output_dir / "MODEL_INTENT.md"
    intent_path.write_text(_render_model_intent(manifest))
    return GlucoseTrainingPack(
        dataset_path=dataset_path,
        config_path=config_path,
        manifest_path=manifest_path,
        model_intent_path=intent_path,
        row_count=int(len(merged)),
        subject_count=int(merged["subject_id"].nunique()),
        source_count=len(sources),
    )


def _render_model_intent(manifest: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# IINTS Glucose Forecast Model Intent",
            "",
            "This training pack is for a research-only glucose forecasting model.",
            "It is not a medical device, not a treatment recommender, and not a dosing authority.",
            "",
            "## Task",
            "",
            "Predict future glucose trajectories from recent CGM, insulin, carbohydrate, activity, stress, and physiology features.",
            "",
            "## Dataset Summary",
            "",
            f"- Rows: {manifest['row_count']}",
            f"- Subjects: {manifest['subject_count']}",
            f"- Sources: {manifest['source_count']}",
            f"- History window: {manifest['history_minutes']} min",
            f"- Forecast horizon: {manifest['horizon_minutes']} min",
            "",
            "## Safety Boundary",
            "",
            "The model output may be used for simulation, retrospective analysis, uncertainty research, and safety-supervisor experiments only.",
            "Any controller experiment must keep deterministic safety supervision as the final authority.",
        ]
    )


def public_manifest_from_private(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Redact local paths and hashes that should not be published with gated datasets."""
    public_sources = []
    for source in manifest.get("sources", []):
        if not isinstance(source, Mapping):
            continue
        public_sources.append(
            {
                "label": source.get("label"),
                "rows": source.get("rows"),
                "subjects": source.get("subjects"),
                "segments": source.get("segments"),
                "glucose_mean_mgdl": source.get("glucose_mean_mgdl"),
                "glucose_min_mgdl": source.get("glucose_min_mgdl"),
                "glucose_max_mgdl": source.get("glucose_max_mgdl"),
            }
        )
    return {
        "schema_version": manifest.get("schema_version"),
        "model_id": manifest.get("model_id", GLUCOSE_MODEL_ID),
        "created_utc": manifest.get("created_utc"),
        "row_count": manifest.get("row_count"),
        "subject_count": manifest.get("subject_count"),
        "segment_count": manifest.get("segment_count"),
        "source_count": manifest.get("source_count"),
        "sources": public_sources,
        "feature_columns": manifest.get("feature_columns", GLUCOSE_MODEL_FEATURE_COLUMNS),
        "target_column": manifest.get("target_column", "glucose_actual_mgdl"),
        "history_minutes": manifest.get("history_minutes"),
        "horizon_minutes": manifest.get("horizon_minutes"),
        "time_step_minutes": manifest.get("time_step_minutes"),
        "privacy": {
            "raw_private_data_included": False,
            "local_paths_redacted": True,
            "raw_dataset_hashes_redacted": True,
            "note": "Publish model weights and aggregate metadata only unless the dataset license explicitly allows row-level sharing.",
        },
    }


_HF_COMPARISON_ARTIFACTS: Mapping[str, str] = {
    "comparison_report.md": "comparison_report.md",
    "comparison_report.json": "comparison_report.json",
    "horizon_metrics.csv": "horizon_metrics.csv",
    "physiological_violation_metrics.csv": "physiological_violation_metrics.csv",
    "hypo_detection_metrics.csv": "hypo_detection_metrics.csv",
    "model_card_metrics.json": "model_card_metrics.json",
}


def _copy_first_existing(source_paths: Sequence[Path], destination: Path) -> Optional[Path]:
    for source in source_paths:
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return destination
    return None


def _copy_hf_comparison_artifacts(comparison_dir: Optional[Path], output_dir: Path) -> dict[str, str]:
    if comparison_dir is None:
        return {}
    source_dir = comparison_dir.expanduser()
    if not source_dir.exists():
        raise FileNotFoundError(f"Comparison directory not found: {source_dir}")
    copied: dict[str, str] = {}
    for source_name, target_name in _HF_COMPARISON_ARTIFACTS.items():
        source = source_dir / source_name
        if source.exists():
            destination = output_dir / target_name
            shutil.copy2(source, destination)
            copied[target_name] = str(destination)
    figures_dir = source_dir / "figures"
    if figures_dir.exists():
        target_figures = output_dir / "figures"
        if target_figures.exists():
            shutil.rmtree(target_figures)
        shutil.copytree(figures_dir, target_figures)
        copied["figures"] = str(target_figures)
    return copied


def _load_comparison_metrics(comparison_dir: Optional[Path]) -> Optional[dict[str, Any]]:
    if comparison_dir is None:
        return None
    metrics_path = comparison_dir.expanduser() / "model_card_metrics.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text())
    return payload if isinstance(payload, dict) else None


def write_huggingface_export_bundle(
    *,
    model_dir: Path,
    output_dir: Path,
    repo_id: Optional[str] = None,
    dataset_manifest: Optional[Path] = None,
    comparison_dir: Optional[Path] = None,
    model_name: str = GLUCOSE_MODEL_ID,
) -> dict[str, str]:
    model_dir = model_dir.expanduser()
    output_dir = output_dir.expanduser()
    checkpoint = model_dir / "predictor.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_checkpoint = output_dir / "predictor.pt"
    shutil.copy2(checkpoint, copied_checkpoint)

    copied_model_config = _copy_first_existing(
        [
            model_dir / "glucose_model_config.yaml",
            model_dir / "glucose_model_config.resolved.yaml",
            model_dir / "dataset" / "glucose_model_config.yaml",
            model_dir.parent / "dataset" / "glucose_model_config.yaml",
        ],
        output_dir / "glucose_model_config.yaml",
    )

    training_report_path = model_dir / "training_report.json"
    training_report: dict[str, Any] = {}
    if training_report_path.exists():
        training_report = json.loads(training_report_path.read_text())
        (output_dir / "training_report.json").write_text(json.dumps(training_report, indent=2))

    public_manifest: Optional[dict[str, Any]] = None
    if dataset_manifest is not None and dataset_manifest.exists():
        manifest = json.loads(dataset_manifest.read_text())
        public_manifest = public_manifest_from_private(manifest)
        (output_dir / "dataset_manifest.public.json").write_text(json.dumps(public_manifest, indent=2))

    comparison_artifacts = _copy_hf_comparison_artifacts(comparison_dir, output_dir)
    comparison_metrics = _load_comparison_metrics(comparison_dir)
    hf_config = {
        "model_id": model_name,
        "task": "multi-horizon glucose forecasting",
        "framework": "pytorch",
        "checkpoint_file": "predictor.pt",
        "research_only": True,
        "not_for_treatment": True,
        "feature_columns": (public_manifest or {}).get("feature_columns") or training_report.get("feature_columns"),
        "history_steps": training_report.get("history_steps"),
        "horizon_steps": training_report.get("horizon_steps"),
        "source_sdk": "IINTS-SDK",
        "evaluation_artifacts": sorted([*comparison_artifacts, "comparison_interpretation.md"]),
    }
    (output_dir / "config.json").write_text(json.dumps(hf_config, indent=2))
    readme = render_huggingface_model_card(
        model_name=model_name,
        repo_id=repo_id,
        training_report=training_report,
        public_manifest=public_manifest,
        comparison_metrics=comparison_metrics,
        comparison_artifacts=comparison_artifacts,
    )
    (output_dir / "README.md").write_text(readme)
    (output_dir / "privacy.md").write_text(render_hf_privacy_notes(public_manifest=public_manifest))
    (output_dir / "limitations.md").write_text(render_hf_limitations(comparison_metrics=comparison_metrics))
    (output_dir / "comparison_interpretation.md").write_text(
        render_hf_comparison_interpretation(comparison_metrics=comparison_metrics)
    )
    (output_dir / "PUBLISHING.md").write_text(render_hf_publishing_notes(repo_id=repo_id))
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    inference_example = examples_dir / "inference_example.py"
    inference_example.write_text(render_hf_inference_example())
    sample_trace = examples_dir / "sample_glucose_trace.csv"
    sample_trace.write_text(render_hf_sample_trace_csv())

    outputs = {
        "output_dir": str(output_dir),
        "checkpoint": str(copied_checkpoint),
        "readme": str(output_dir / "README.md"),
        "config": str(output_dir / "config.json"),
        "privacy": str(output_dir / "privacy.md"),
        "limitations": str(output_dir / "limitations.md"),
        "comparison_interpretation": str(output_dir / "comparison_interpretation.md"),
        "publishing_notes": str(output_dir / "PUBLISHING.md"),
        "inference_example": str(inference_example),
        "sample_trace": str(sample_trace),
    }
    if copied_model_config is not None:
        outputs["glucose_model_config"] = str(copied_model_config)
    outputs.update({f"artifact_{key}": value for key, value in comparison_artifacts.items()})
    return outputs


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _hf_display_title(model_name: str) -> str:
    if model_name == GLUCOSE_MODEL_ID:
        return "IINTS-AF Glucose Forecast v0"
    return model_name


def _render_metric_table(comparison_metrics: Optional[Mapping[str, Any]]) -> list[str]:
    if not comparison_metrics:
        return [
            "Comparison metrics were not bundled yet.",
            "Run `iints research glucose-model compare` and pass `--comparison-dir` to `export-hf` before public release.",
        ]
    rows = comparison_metrics.get("models", [])
    if not isinstance(rows, list) or not rows:
        return ["Comparison metrics file was bundled, but no model rows were available."]
    lines = [
        "| Model | Kind | MAE | RMSE | Missed hypo % | Physiology violation % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {model} | {kind} | {mae} | {rmse} | {missed} | {viol} |".format(
                model=row.get("model", "unknown"),
                kind=row.get("kind", "unknown"),
                mae=_fmt_metric(row.get("mae")),
                rmse=_fmt_metric(row.get("rmse")),
                missed=_fmt_metric(row.get("missed_hypo_rate_pct")),
                viol=_fmt_metric(row.get("any_physiology_violation_pct")),
            )
        )
    return lines


def _render_artifact_lines(comparison_artifacts: Mapping[str, str]) -> list[str]:
    if not comparison_artifacts:
        return [
            "- `comparison_interpretation.md`: how to interpret MSE, PINN, horizons, and physiology gates",
            "- Comparison artifacts: not bundled yet",
            "- Recommended command: `iints research glucose-model compare ...`",
        ]
    return [
        "- `comparison_interpretation.md`: how to interpret MSE, PINN, horizons, and physiology gates",
        *[f"- `{name}`" for name in sorted(comparison_artifacts)],
    ]


def render_huggingface_model_card(
    *,
    model_name: str,
    repo_id: Optional[str],
    training_report: Mapping[str, Any],
    public_manifest: Optional[Mapping[str, Any]],
    comparison_metrics: Optional[Mapping[str, Any]] = None,
    comparison_artifacts: Optional[Mapping[str, str]] = None,
) -> str:
    repo_line = f"- Repository: `{repo_id}`" if repo_id else "- Repository: not set yet"
    data_lines = ["- Dataset manifest: not provided"]
    if public_manifest:
        data_lines = [
            f"- Rows: {public_manifest.get('row_count', 'unknown')}",
            f"- Subjects: {public_manifest.get('subject_count', 'unknown')}",
            f"- Sources: {public_manifest.get('source_count', 'unknown')}",
            "- Raw/private row-level data: not included",
        ]
        for source in public_manifest.get("sources", []):
            if isinstance(source, Mapping):
                data_lines.append(
                    f"- Source `{source.get('label')}`: {source.get('rows')} rows, {source.get('subjects')} subjects"
                )
    artifact_lines = _render_artifact_lines(comparison_artifacts or {})
    metric_lines = _render_metric_table(comparison_metrics)
    title = _hf_display_title(model_name)
    return "\n".join(
        [
            "---",
            "license: apache-2.0",
            "tags:",
            "- glucose-forecasting",
            "- diabetes-technology",
            "- time-series",
            "- research-only",
            "- iints",
            "pipeline_tag: time-series-forecasting",
            "---",
            "",
            f"# {title}",
            "",
            repo_line,
            "- Source SDK: IINTS-SDK",
            "- Status: research-only, not a medical device",
            "",
            "> This model is not a medical device. It must not be used for insulin dosing, treatment decisions, diagnosis, or real-time patient care.",
            "",
            "## Intended Use",
            "",
            "This model is designed to forecast short-term glucose trajectories from CGM and contextual diabetes-technology features for simulation, retrospective analysis, and safety-supervisor research.",
            "It must not be used for diagnosis, treatment, insulin dosing, glucagon dosing, or real-world medical decision-making.",
            "",
            "## Inputs",
            "",
            "The v0 contract uses recent windows of glucose, trend, insulin-on-board, carbohydrates-on-board, delivered insulin, carbs, activity, stress, circadian features, glucagon, and experimental HAAF/counter-regulation features when available.",
            "Missing optional features are filled with conservative defaults during IINTS dataset preparation.",
            "",
            "## Outputs",
            "",
            "The checkpoint predicts a multi-step future glucose trajectory. IINTS evaluation commands can derive 30/60/120-minute forecasts, hypo/hyper risk, band-wise error, missed-hypoglycemia rate, and uncertainty reliability when MC dropout is enabled.",
            "",
            "## Training Data Summary",
            "",
            *data_lines,
            "",
            "## Evaluation Snapshot",
            "",
            f"- Test MAE: {_fmt_metric(training_report.get('test_mae'))} mg/dL",
            f"- Test RMSE: {_fmt_metric(training_report.get('test_rmse'))} mg/dL",
            f"- Test bias: {_fmt_metric(training_report.get('test_bias'))} mg/dL",
            "",
            "## Physiology-Aware Evaluation",
            "",
            "The model is evaluated not only by numerical error, but also by physiological plausibility:",
            "",
            "- impossible glucose predictions",
            "- excessive glucose rate-of-change",
            "- hypoglycemia detection performance",
            "- horizon-specific accuracy",
            "- insulin-on-board / carbs-on-board consistency checks",
            "",
            *metric_lines,
            "",
            "## Bundled Evaluation Files",
            "",
            *artifact_lines,
            "",
            "## Safety Limitations",
            "",
            "- This model can be wrong, overconfident, distribution-shifted, or biased by the training data.",
            "- Subject-level held-out validation and external dataset validation are required before any research claim.",
            "- Deterministic safety supervision must remain the final authority in any simulated controller experiment.",
            "- Do not upload private OhioT1DM or patient-level raw data unless the dataset license and ethics constraints explicitly permit it.",
            "",
            "See `comparison_interpretation.md`, `limitations.md`, and `privacy.md` for the full publication boundary.",
            "",
            "## Example IINTS Usage",
            "",
            "```bash",
            "iints research forecast-run \\",
            "  --input results/run/results.csv \\",
            "  --predictor models/iints-glucose-forecast-v0/predictor.pt \\",
            "  --output-dir results/forecast_with_iints_glucose_model",
            "```",
            "",
        ]
    )


def render_hf_privacy_notes(*, public_manifest: Optional[Mapping[str, Any]]) -> str:
    lines = [
        "# Privacy And Dataset Boundary",
        "",
        "This repository is intended to contain model artifacts, aggregate metrics, and documentation only.",
        "It should not contain raw private patient rows, local dataset paths, or private dataset hashes.",
        "",
        "## What May Be Included",
        "",
        "- `predictor.pt` model weights",
        "- model configuration",
        "- aggregate evaluation metrics",
        "- public/redacted dataset manifest metadata",
        "- example synthetic or toy traces",
        "",
        "## What Must Not Be Included By Default",
        "",
        "- raw OhioT1DM rows unless the license and ethics constraints explicitly allow it",
        "- private patient identifiers",
        "- local filesystem paths",
        "- non-redacted source hashes for gated/private files",
        "- clinical claims beyond the performed validation",
        "",
    ]
    if public_manifest:
        lines.extend(
            [
                "## Redacted Dataset Summary",
                "",
                f"- Rows: {public_manifest.get('row_count', 'unknown')}",
                f"- Subjects: {public_manifest.get('subject_count', 'unknown')}",
                f"- Sources: {public_manifest.get('source_count', 'unknown')}",
                "- Raw/private row-level data included: no",
                "",
            ]
        )
    lines.extend(
        [
            "## Publication Checklist",
            "",
            "- Verify `dataset_manifest.public.json` has no private paths.",
            "- Verify examples are synthetic/toy or explicitly public.",
            "- Upload privately first and inspect the Hugging Face file list.",
            "- Keep the research-only and not-for-treatment boundary visible.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_hf_limitations(*, comparison_metrics: Optional[Mapping[str, Any]]) -> str:
    best = comparison_metrics.get("best_by_mae") if comparison_metrics else None
    best_line = "- Best-by-MAE model: filled when `model_card_metrics.json` is bundled"
    if isinstance(best, Mapping):
        best_line = f"- Best-by-MAE model: `{best.get('model')}` with MAE `{_fmt_metric(best.get('mae'))}` mg/dL"
    return "\n".join(
        [
            "# Limitations",
            "",
            "This model is a research artifact for glucose forecasting experiments inside IINTS-AF.",
            "It is not a medical device and must not be used for treatment decisions.",
            "",
            "## Model Limitations",
            "",
            "- Forecasts can be wrong during distribution shift, sensor artifacts, illness, exercise, or unusual meals.",
            "- Lower MAE does not automatically mean safer behavior.",
            "- A model can look accurate on average while still missing hypoglycemia or producing implausible rates of change.",
            "- The model does not replace deterministic safety checks, clinical judgment, or validated medical-device software.",
            "",
            "## Evaluation Limitations",
            "",
            best_line,
            "- External held-out validation is required before making strong research claims.",
            "- Physiological gates are screening tools, not regulatory certification.",
            "- Simulator-generated traces can help stress-test behavior but are not a substitute for real-world validation.",
            "",
            "## Intended Boundary",
            "",
            "- Allowed: education, simulation, retrospective research, benchmarking, model-card reporting.",
            "- Not allowed: insulin dosing, glucagon dosing, diagnosis, treatment, real-time patient care.",
        ]
    ) + "\n"


def render_hf_comparison_interpretation(*, comparison_metrics: Optional[Mapping[str, Any]]) -> str:
    best = comparison_metrics.get("best_by_mae") if comparison_metrics else None
    best_line = "- Best-by-MAE model: not bundled yet"
    if isinstance(best, Mapping):
        best_line = f"- Best-by-MAE model: `{best.get('model')}` with MAE `{_fmt_metric(best.get('mae'))}` mg/dL"
    return "\n".join(
        [
            "# Interpreting Glucose Forecast Results",
            "",
            "This note explains how to read IINTS-AF glucose model comparison outputs.",
            "It is written for research, model-card review, and jury discussions. It is not a clinical validation claim.",
            "",
            "## 1. What The Comparison Is Trying To Answer",
            "",
            "The comparison should not answer only: which model has the lowest average error?",
            "For diabetes-technology research, it should also answer:",
            "",
            "- Which model misses hypoglycemia least often?",
            "- Which model creates the fewest physiologically impossible predictions?",
            "- Which model degrades most gracefully at longer horizons?",
            "- Which model remains consistent with insulin-on-board and carbs-on-board context?",
            "- Which model is easiest to explain and audit?",
            "",
            best_line,
            "",
            "## 2. Why MSE Can Look Best",
            "",
            "A standard MSE model minimizes the squared forecast error:",
            "",
            "```text",
            "L_MSE = mean((predicted_glucose - observed_glucose)^2)",
            "```",
            "",
            "Under strong assumptions such as symmetric noise, independent errors, and a squared-error objective, this is a sensible estimator of average behavior. In classical linear settings, least-squares reasoning is related to the Gauss-Markov result: among linear unbiased estimators under homoscedastic uncorrelated errors, ordinary least squares has minimum variance.",
            "",
            "That does not mean the lowest-MSE model is automatically the safest or most useful model for diabetes research. MSE treats errors symmetrically and averages over all windows. A model can have good average MAE/RMSE while still making rare but important errors around hypoglycemia, meals, sensor artifacts, or fast insulin action.",
            "",
            "## 3. Why PINN Is Different",
            "",
            "The IINTS physiological loss keeps the normal forecast-error term, but adds penalties when predictions violate basic physiology:",
            "",
            "```text",
            "L_total = L_MSE + lambda * L_physiology",
            "```",
            "",
            "In the current SDK implementation, the physiological penalty includes:",
            "",
            "- impossible glucose bounds below 20 mg/dL or above 600 mg/dL",
            "- excessive first-step glucose rate-of-change from the last observed glucose",
            "- suspicious fast rise when insulin-on-board is high and carbs-on-board is near zero",
            "- suspicious fast drop when carbs-on-board is present but insulin-on-board is low",
            "",
            "This can make a PINN model trade a small amount of average error for fewer implausible or safety-relevant failures. That tradeoff is intentional: in medical-device research, a model that is slightly less optimal on average but more physiologically conservative can be more useful for simulation and safety-supervisor experiments.",
            "",
            "## 4. Why Longer Horizons Are Harder",
            "",
            "Short horizons, such as 15 or 30 minutes, are often dominated by recent glucose trend and sensor continuity. Longer horizons, such as 60 or 120 minutes, depend much more on delayed meal absorption, insulin pharmacodynamics, activity, stress, circadian effects, and sensor lag.",
            "",
            "Forecast uncertainty normally grows with horizon because each future step depends on uncertain previous state evolution. In simple stochastic systems, error variance can accumulate with the number of steps; in nonlinear glucose physiology, meals, insulin, exercise, and counter-regulation can amplify this growth in a non-linear way.",
            "",
            "For that reason, a strong result should be reported by horizon, not only as one average number. Use `horizon_metrics.csv` to check whether the model remains plausible at 30, 60, and 120 minutes.",
            "",
            "## 5. How To Promote A Model",
            "",
            "Do not promote a model only because it has the lowest MAE or RMSE. Use this order:",
            "",
            "1. Reject models with private-data leakage or invalid splits.",
            "2. Reject models with high impossible-glucose or rate-of-change violations.",
            "3. Review missed hypoglycemia and false hypo alarms.",
            "4. Compare horizon-specific degradation.",
            "5. Use MAE/RMSE as final tie-breakers, not as the only decision rule.",
            "",
            "## 6. Pitch-Friendly Explanation",
            "",
            "A normal AI model learns to be close on average. IINTS-AF also asks whether the prediction still behaves like glucose in a human body. The PINN loss adds a mathematical penalty when the model predicts values or rates that are physiologically implausible, and the comparison report checks those errors separately from normal MAE/RMSE.",
            "",
            "## 7. Boundary",
            "",
            "These metrics support research and education. They are not regulatory validation, clinical validation, or evidence that the model can be used for treatment decisions.",
        ]
    ) + "\n"


def render_hf_inference_example() -> str:
    return '''"""Minimal IINTS-AF glucose forecasting example.

This script is for research and simulation only. It is not for treatment,
insulin dosing, diagnosis, or real-time patient care.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from iints.research.config import PredictorConfig
from iints.research.glucose_model import standardize_glucose_forecast_frame
from iints.research.predictor import load_predictor_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one research-only glucose forecast.")
    parser.add_argument("--model", type=Path, default=Path("../predictor.pt"))
    parser.add_argument("--input", type=Path, default=Path("sample_glucose_trace.csv"))
    parser.add_argument("--config", type=Path, default=Path("../glucose_model_config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("forecast.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = load_predictor_service(args.model)
    if args.config.exists():
        payload = yaml.safe_load(args.config.read_text())
        predictor_cfg = PredictorConfig(**payload["predictor"])
    else:
        predictor_cfg = PredictorConfig(
            history_minutes=service.history_steps * 5,
            horizon_minutes=service.horizon_steps * 5,
            feature_columns=service.feature_columns,
        )

    raw = pd.read_csv(args.input)
    frame = standardize_glucose_forecast_frame(
        raw,
        source_label="example",
        time_step_minutes=predictor_cfg.time_step_minutes,
    )
    if "meal_announcement_grams" in predictor_cfg.feature_columns and "meal_announcement_grams" not in frame.columns:
        shift_steps = int(round(15 / predictor_cfg.time_step_minutes))
        frame["meal_announcement_grams"] = frame["carb_intake_grams"].shift(-shift_steps).fillna(0.0)
    for column in predictor_cfg.feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
    if len(frame) < predictor_cfg.history_steps:
        raise SystemExit(
            f"Need at least {predictor_cfg.history_steps} rows for this model; got {len(frame)}."
        )

    x = frame.tail(predictor_cfg.history_steps)[predictor_cfg.feature_columns].to_numpy(
        dtype=np.float32
    )[None, :, :]
    forecast = service.predict(x)[0].astype(float)
    horizons = [
        int((index + 1) * predictor_cfg.time_step_minutes)
        for index in range(len(forecast))
    ]
    output = {
        "research_only": True,
        "not_for_treatment": True,
        "horizon_minutes": horizons,
        "predicted_glucose_mgdl": forecast.tolist(),
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
'''


def render_hf_sample_trace_csv(rows: int = 80) -> str:
    lines = ["time_minutes,subject_id,glucose,carbs,insulin,heart_rate"]
    for index in range(rows):
        minutes = index * 5
        glucose = 125.0 + float(np.sin(index / 8.0) * 16.0)
        if 20 <= index <= 34:
            glucose += float((index - 20) * 2.2)
        if 35 <= index <= 50:
            glucose += float(max(0, 50 - index) * 1.4)
        carbs = 45.0 if index == 20 else 0.0
        insulin = 3.0 if index == 22 else 0.0
        heart_rate = 78.0 + float(np.sin(index / 11.0) * 4.0)
        lines.append(f"{minutes},example,{glucose:.2f},{carbs:.1f},{insulin:.2f},{heart_rate:.1f}")
    return "\n".join(lines) + "\n"


def render_hf_publishing_notes(*, repo_id: Optional[str]) -> str:
    target = repo_id or "YOUR_USERNAME/iints-glucose-forecast-v0"
    return "\n".join(
        [
            "# Publishing Notes",
            "",
            "This folder is Hugging Face-ready, but publishing should be deliberate.",
            "",
            "## Recommended private-first upload",
            "",
            "```bash",
            f"huggingface-cli upload {target} . . --private",
            "```",
            "",
            "Before making the repository public, verify:",
            "",
            "- `README.md` states research-only and not-for-treatment boundaries.",
            "- `dataset_manifest.public.json` contains aggregate metadata only.",
            "- No raw OhioT1DM/private patient rows are present.",
            "- External held-out evaluation is included or clearly marked as pending.",
        ]
    )


@dataclass(frozen=True)
class GlucoseModelSpec:
    label: str
    path: Optional[Path] = None
    kind: str = "checkpoint"


@dataclass(frozen=True)
class GlucoseModelComparisonBundle:
    output_dir: Path
    report_json: Path
    report_md: Path
    horizon_metrics_csv: Path
    physiological_violations_csv: Path
    hypo_detection_csv: Path
    model_card_metrics_json: Path
    row_count: int
    model_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "report_json": str(self.report_json),
            "report_md": str(self.report_md),
            "horizon_metrics_csv": str(self.horizon_metrics_csv),
            "physiological_violations_csv": str(self.physiological_violations_csv),
            "hypo_detection_csv": str(self.hypo_detection_csv),
            "model_card_metrics_json": str(self.model_card_metrics_json),
            "row_count": self.row_count,
            "model_count": self.model_count,
        }


def parse_model_specs(values: Sequence[str]) -> list[GlucoseModelSpec]:
    specs: list[GlucoseModelSpec] = []
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        if "=" in value:
            label, path_text = value.split("=", 1)
            label = label.strip()
            path = Path(path_text.strip()).expanduser()
        else:
            path = Path(value).expanduser()
            label = path.parent.name or path.stem
        if not label:
            raise ValueError(f"Invalid model spec: {raw!r}")
        specs.append(GlucoseModelSpec(label=label, path=path))
    return specs


def _load_predictor_config_payload(config_path: Optional[Path]) -> dict[str, Any]:
    if config_path is None:
        return glucose_model_config_payload(profile="quick")
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or "predictor" not in payload:
        raise ValueError("Comparison config must contain a 'predictor' block.")
    return payload


def _ensure_comparison_frame(df: pd.DataFrame, predictor_cfg: PredictorConfig) -> pd.DataFrame:
    required = set(predictor_cfg.feature_columns + [predictor_cfg.target_column])
    if required.issubset(df.columns):
        out = df.copy()
        if "subject_id" not in out.columns:
            out["subject_id"] = "subject_000"
        if "segment" not in out.columns:
            out["segment"] = out["subject_id"].astype(str) + ":segment_0"
        if "time_minutes" not in out.columns:
            out["time_minutes"] = np.arange(len(out), dtype=float) * float(predictor_cfg.time_step_minutes)
        return out
    if predictor_cfg.target_column in df.columns:
        out = df.copy()
        if "subject_id" not in out.columns:
            out["subject_id"] = "subject_000"
        if "segment" not in out.columns:
            out["segment"] = out["subject_id"].astype(str) + ":segment_0"
        if "time_minutes" not in out.columns:
            out["time_minutes"] = np.arange(len(out), dtype=float) * float(predictor_cfg.time_step_minutes)
        for column in predictor_cfg.feature_columns:
            if column not in out.columns:
                out[column] = _DEFAULT_FEATURE_VALUES.get(column, 0.0)
        return out
    return standardize_glucose_forecast_frame(
        df,
        source_label="comparison",
        time_step_minutes=predictor_cfg.time_step_minutes,
    )


def _apply_comparison_meal_announcement(
    df: pd.DataFrame,
    predictor_cfg: PredictorConfig,
    training_cfg: TrainingConfig,
) -> pd.DataFrame:
    minutes = training_cfg.meal_announcement_minutes
    feature = training_cfg.meal_announcement_feature
    if minutes is None or feature not in predictor_cfg.feature_columns:
        return df
    out = df.copy()
    source = training_cfg.meal_announcement_column
    if source not in out.columns:
        out[feature] = 0.0
        return out
    shift_steps = int(round(minutes / predictor_cfg.time_step_minutes))
    if shift_steps <= 0:
        return out
    group_cols = [column for column in ("subject_id", "segment") if column in out.columns]
    sort_cols = [column for column in (*group_cols, "time_minutes") if column in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    if group_cols:
        out[feature] = out.groupby(group_cols, observed=False)[source].shift(-shift_steps).fillna(0.0)
    else:
        out[feature] = out[source].shift(-shift_steps).fillna(0.0)
    return out


def _build_comparison_sequences(
    data_path: Path,
    predictor_cfg: PredictorConfig,
    training_cfg: Optional[TrainingConfig] = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = _ensure_comparison_frame(load_dataset(data_path), predictor_cfg)
    if training_cfg is not None:
        df = _apply_comparison_meal_announcement(df, predictor_cfg, training_cfg)
    segment_column = "segment" if "segment" in df.columns else None
    X, y = build_sequences(
        df,
        history_steps=predictor_cfg.history_steps,
        horizon_steps=predictor_cfg.horizon_steps,
        feature_columns=predictor_cfg.feature_columns,
        target_column=predictor_cfg.target_column,
        segment_column=segment_column,
    )
    return df, X, y


def _safe_metric(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def horizon_error_rows(
    *,
    label: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    time_step_minutes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(predicted.shape[1]):
        obs = observed[:, index]
        pred = predicted[:, index]
        report = forecast_error_report(obs, pred)
        rows.append(
            {
                "model": label,
                "horizon_step": index + 1,
                "horizon_minutes": int((index + 1) * time_step_minutes),
                "n": report["n"],
                "mae": report["mae"],
                "rmse": report["rmse"],
                "bias": report["bias"],
                "within_10_mgdl_pct": report["within_10_mgdl_pct"],
                "within_20_mgdl_pct": report["within_20_mgdl_pct"],
                "false_hypo_alarm_rate_pct": report["false_hypo_alarm_rate_pct"],
                "missed_hypo_rate_pct": report["missed_hypo_rate_pct"],
            }
        )
    return rows


def physiological_violation_report(
    X: np.ndarray,
    predicted: np.ndarray,
    *,
    feature_columns: Sequence[str],
    time_step_minutes: int,
    absolute_low_mgdl: float = 20.0,
    absolute_high_mgdl: float = 600.0,
    display_low_mgdl: float = 35.0,
    display_high_mgdl: float = 450.0,
    max_roc_mgdl_min: float = 10.0,
    suspicious_roc_mgdl_min: float = 2.0,
) -> dict[str, Any]:
    glucose_index = feature_columns.index("glucose_actual_mgdl") if "glucose_actual_mgdl" in feature_columns else 0
    iob_index = feature_columns.index("patient_iob_units") if "patient_iob_units" in feature_columns else None
    cob_index = feature_columns.index("patient_cob_grams") if "patient_cob_grams" in feature_columns else None
    last_glucose = X[:, -1, glucose_index].astype(float)
    previous = np.concatenate([last_glucose[:, None], predicted[:, :-1]], axis=1)
    roc = (predicted.astype(float) - previous) / max(float(time_step_minutes), 1e-6)

    impossible_low = predicted < absolute_low_mgdl
    impossible_high = predicted > absolute_high_mgdl
    outside_display = (predicted < display_low_mgdl) | (predicted > display_high_mgdl)
    roc_violation = np.abs(roc) > max_roc_mgdl_min

    iob = np.zeros(len(X), dtype=float) if iob_index is None else X[:, -1, iob_index].astype(float)
    cob = np.zeros(len(X), dtype=float) if cob_index is None else X[:, -1, cob_index].astype(float)
    first_roc = roc[:, 0]
    suspicious_rise = (first_roc > suspicious_roc_mgdl_min) & (iob > 1.0) & (cob < 5.0)
    suspicious_drop = (first_roc < -suspicious_roc_mgdl_min) & (cob > 10.0) & (iob < 0.5)
    any_sample_violation = (
        impossible_low.any(axis=1)
        | impossible_high.any(axis=1)
        | roc_violation.any(axis=1)
        | suspicious_rise
        | suspicious_drop
    )

    denominator = max(1, int(np.size(predicted)))
    sample_denominator = max(1, int(len(predicted)))
    return {
        "absolute_low_threshold_mgdl": float(absolute_low_mgdl),
        "absolute_high_threshold_mgdl": float(absolute_high_mgdl),
        "display_low_threshold_mgdl": float(display_low_mgdl),
        "display_high_threshold_mgdl": float(display_high_mgdl),
        "max_roc_mgdl_min": float(max_roc_mgdl_min),
        "suspicious_roc_mgdl_min": float(suspicious_roc_mgdl_min),
        "impossible_low_count": int(np.sum(impossible_low)),
        "impossible_high_count": int(np.sum(impossible_high)),
        "outside_display_count": int(np.sum(outside_display)),
        "roc_violation_count": int(np.sum(roc_violation)),
        "suspicious_rise_iob_no_cob_count": int(np.sum(suspicious_rise)),
        "suspicious_drop_cob_no_iob_count": int(np.sum(suspicious_drop)),
        "impossible_low_pct": float(np.sum(impossible_low) / denominator * 100.0),
        "impossible_high_pct": float(np.sum(impossible_high) / denominator * 100.0),
        "outside_display_pct": float(np.sum(outside_display) / denominator * 100.0),
        "roc_violation_pct": float(np.sum(roc_violation) / denominator * 100.0),
        "suspicious_rise_iob_no_cob_pct": float(np.sum(suspicious_rise) / sample_denominator * 100.0),
        "suspicious_drop_cob_no_iob_pct": float(np.sum(suspicious_drop) / sample_denominator * 100.0),
        "any_physiology_violation_pct": float(np.sum(any_sample_violation) / sample_denominator * 100.0),
        "max_abs_roc_mgdl_min": float(np.max(np.abs(roc))) if roc.size else 0.0,
    }


def _baseline_predictions(
    *,
    X: np.ndarray,
    horizon_steps: int,
    time_step_minutes: int,
    feature_columns: Sequence[str],
) -> dict[str, np.ndarray]:
    baselines: list[Any] = [
        LastValueBaseline(horizon_steps),
        LinearTrendBaseline(horizon_steps, time_step_minutes),
        PhysiologyAwareBaseline(
            horizon_steps,
            time_step_minutes=time_step_minutes,
            feature_columns=feature_columns,
        ),
    ]
    return {baseline.name(): baseline.predict(X) for baseline in baselines}


def _write_comparison_figures(output_dir: Path, horizon_df: pd.DataFrame, violation_df: pd.DataFrame) -> list[str]:
    figures_dir = output_dir / "figures"
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return written

    figures_dir.mkdir(parents=True, exist_ok=True)
    if not horizon_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=160)
        for model, group in horizon_df.groupby("model"):
            ax.plot(group["horizon_minutes"], group["mae"], marker="o", label=str(model))
        ax.set_title("Glucose forecast MAE by horizon")
        ax.set_xlabel("Forecast horizon (minutes)")
        ax.set_ylabel("MAE (mg/dL)")
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
        fig.tight_layout()
        path = figures_dir / "horizon_mae.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))
    if not violation_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=160)
        ax.bar(violation_df["model"], violation_df["any_physiology_violation_pct"], color="#B45309")
        ax.set_title("Physiological violation rate by model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Any violation (% windows)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        path = figures_dir / "physiology_violations.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))
    return written


def _render_comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# IINTS Glucose Model Comparison",
        "",
        "Research-only comparison of glucose forecasting models and transparent baselines.",
        "This report is not a medical device evaluation and must not be used for treatment decisions.",
        "",
        "## Summary",
        "",
        f"- Dataset: `{report.get('data_path')}`",
        f"- Rows: {report.get('row_count')}",
        f"- Sequence windows: {report.get('sequence_count')}",
        f"- Horizon: {report.get('horizon_minutes')} minutes",
        f"- Models compared: {report.get('model_count')}",
        "",
        "## Model Metrics",
        "",
        "| Model | MAE | RMSE | Missed hypo % | Any physiology violation % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("models", []):
        lines.append(
            "| {model} | {mae} | {rmse} | {missed} | {viol} |".format(
                model=row.get("model"),
                mae=_fmt_metric(row.get("mae")),
                rmse=_fmt_metric(row.get("rmse")),
                missed=_fmt_metric(row.get("missed_hypo_rate_pct")),
                viol=_fmt_metric(row.get("any_physiology_violation_pct")),
            )
        )
    best = report.get("best_by_mae")
    if isinstance(best, Mapping):
        lines.extend(
            [
                "",
                "## Best By MAE",
                "",
                f"- `{best.get('model')}` with MAE `{_fmt_metric(best.get('mae'))}` mg/dL.",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `horizon_metrics.csv`: error by horizon step",
            "- `physiological_violation_metrics.csv`: impossible bounds, rate-of-change and IOB/COB logic checks",
            "- `hypo_detection_metrics.csv`: hypoglycemia detection quality",
            "- `model_card_metrics.json`: compact metrics payload for Hugging Face model cards",
            "",
            "## Interpretation",
            "",
            "A PINN-trained model should not only reduce MAE/RMSE; it should also reduce physiologically impossible predictions. If a lower-error model has more rate-of-change or IOB/COB violations, do not promote it without review.",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_glucose_models(
    *,
    data_path: Path,
    output_dir: Path,
    model_specs: Sequence[GlucoseModelSpec] = (),
    config_path: Optional[Path] = None,
    include_baselines: bool = True,
    mc_samples: int = 0,
    max_roc_mgdl_min: float = 10.0,
) -> GlucoseModelComparisonBundle:
    payload = _load_predictor_config_payload(config_path)
    predictor_cfg = PredictorConfig(**payload["predictor"])
    training_cfg = TrainingConfig(**payload.get("training", {}))
    df, X, y = _build_comparison_sequences(data_path, predictor_cfg, training_cfg)
    predictions: dict[str, tuple[np.ndarray, Optional[np.ndarray], str]] = {}

    if include_baselines:
        for label, values in _baseline_predictions(
            X=X,
            horizon_steps=predictor_cfg.horizon_steps,
            time_step_minutes=predictor_cfg.time_step_minutes,
            feature_columns=predictor_cfg.feature_columns,
        ).items():
            predictions[label] = (values, None, "baseline")

    for spec in model_specs:
        if spec.path is None:
            continue
        service = load_predictor_service(spec.path)
        if service.feature_columns and service.feature_columns != predictor_cfg.feature_columns:
            raise ValueError(
                f"Model '{spec.label}' uses different feature columns than the comparison config. "
                "Run a separate comparison with that model's config."
            )
        if int(service.horizon_steps) != predictor_cfg.horizon_steps:
            raise ValueError(
                f"Model '{spec.label}' horizon_steps={service.horizon_steps}, "
                f"comparison horizon_steps={predictor_cfg.horizon_steps}."
            )
        if mc_samples > 1:
            pred, std = service.predict_with_uncertainty(X, n_samples=mc_samples)
        else:
            pred = service.predict(X)
            std = None
        predictions[spec.label] = (pred, std, "checkpoint")

    if not predictions:
        raise ValueError("No models to compare. Pass --model or keep --include-baselines enabled.")

    output_dir.mkdir(parents=True, exist_ok=True)
    horizon_rows: list[dict[str, Any]] = []
    violation_rows: list[dict[str, Any]] = []
    hypo_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for label, (pred, std, kind) in predictions.items():
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != y.shape:
            raise ValueError(f"Prediction shape mismatch for {label}: {pred.shape} != {y.shape}")
        flat_std = None if std is None else np.asarray(std, dtype=np.float32).reshape(-1)
        global_report = forecast_error_report(y.reshape(-1), pred.reshape(-1), flat_std)
        hypo = hypoglycemia_detection_report(y.reshape(-1), pred.reshape(-1))
        violations = physiological_violation_report(
            X,
            pred,
            feature_columns=predictor_cfg.feature_columns,
            time_step_minutes=predictor_cfg.time_step_minutes,
            max_roc_mgdl_min=max_roc_mgdl_min,
        )
        uncertainty = None
        if std is not None:
            uncertainty = uncertainty_reliability_report(y.reshape(-1), pred.reshape(-1), np.asarray(std).reshape(-1))

        horizon_rows.extend(
            horizon_error_rows(
                label=label,
                observed=y,
                predicted=pred,
                time_step_minutes=predictor_cfg.time_step_minutes,
            )
        )
        violation_row = {"model": label, "kind": kind, **violations}
        violation_rows.append(violation_row)
        hypo_rows.append({"model": label, "kind": kind, **hypo})
        model_row = {
            "model": label,
            "kind": kind,
            "n": global_report["n"],
            "mae": global_report["mae"],
            "rmse": global_report["rmse"],
            "bias": global_report["bias"],
            "within_20_mgdl_pct": global_report["within_20_mgdl_pct"],
            "missed_hypo_rate_pct": global_report["missed_hypo_rate_pct"],
            "false_hypo_alarm_rate_pct": global_report["false_hypo_alarm_rate_pct"],
            "any_physiology_violation_pct": violations["any_physiology_violation_pct"],
            "roc_violation_pct": violations["roc_violation_pct"],
            "outside_display_pct": violations["outside_display_pct"],
            "uncertainty": uncertainty,
        }
        model_rows.append(model_row)

    horizon_df = pd.DataFrame(horizon_rows)
    violation_df = pd.DataFrame(violation_rows)
    hypo_df = pd.DataFrame(hypo_rows)
    horizon_path = output_dir / "horizon_metrics.csv"
    violation_path = output_dir / "physiological_violation_metrics.csv"
    hypo_path = output_dir / "hypo_detection_metrics.csv"
    horizon_df.to_csv(horizon_path, index=False)
    violation_df.to_csv(violation_path, index=False)
    hypo_df.to_csv(hypo_path, index=False)
    figures = _write_comparison_figures(output_dir, horizon_df, violation_df)

    best = min(model_rows, key=lambda row: float(row["mae"])) if model_rows else None
    report = {
        "schema_version": "iints_glucose_model_comparison_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_path": str(data_path),
        "row_count": int(len(df)),
        "sequence_count": int(len(X)),
        "model_count": len(model_rows),
        "history_minutes": predictor_cfg.history_minutes,
        "horizon_minutes": predictor_cfg.horizon_minutes,
        "time_step_minutes": predictor_cfg.time_step_minutes,
        "feature_columns": predictor_cfg.feature_columns,
        "models": model_rows,
        "best_by_mae": best,
        "artifacts": {
            "horizon_metrics_csv": str(horizon_path),
            "physiological_violation_metrics_csv": str(violation_path),
            "hypo_detection_metrics_csv": str(hypo_path),
            "figures": figures,
        },
        "research_boundary": {
            "not_medical_device": True,
            "not_for_treatment": True,
            "interpretation": "Promote a model only if error, hypo detection, uncertainty and physiological violations are all acceptable.",
        },
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    report_md_path = output_dir / "comparison_report.md"
    report_md_path.write_text(_render_comparison_markdown(report))
    model_card_metrics = {
        "model_id": GLUCOSE_MODEL_ID,
        "comparison_created_utc": report["created_utc"],
        "best_by_mae": best,
        "models": [
            {
                key: value
                for key, value in row.items()
                if key not in {"uncertainty"}
            }
            for row in model_rows
        ],
        "privacy": {
            "raw_data_included": False,
            "note": "This metrics payload should not include raw patient rows.",
        },
    }
    model_card_path = output_dir / "model_card_metrics.json"
    model_card_path.write_text(json.dumps(model_card_metrics, indent=2))
    return GlucoseModelComparisonBundle(
        output_dir=output_dir,
        report_json=report_path,
        report_md=report_md_path,
        horizon_metrics_csv=horizon_path,
        physiological_violations_csv=violation_path,
        hypo_detection_csv=hypo_path,
        model_card_metrics_json=model_card_path,
        row_count=int(len(df)),
        model_count=len(model_rows),
    )
