from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.core.algorithms.imitation_controller import ExperimentalImitationController
from iints.core.algorithms.neural_controller import ExperimentalNeuralController
from iints.research.control import (
    build_control_dataset_from_runs,
    save_linear_controller,
    summarize_control_dataset,
    train_linear_imitation_controller,
)
from iints.research.local_ai_gate import review_controller_training_artifacts
from iints.research.control_eval import (
    DEFAULT_HELD_OUT_PRESETS,
    ControllerFactory,
    evaluate_controller_factories,
)
from iints.research.neural_control import (
    NeuralControllerConfig,
    save_neural_controller,
    train_neural_imitation_controller,
)
from iints.utils.run_io import compute_sha256


PREDICTOR_EXPORT_COLUMNS = [
    "subject_id",
    "segment",
    "time_minutes",
    "glucose_actual_mgdl",
    "glucose_to_algo_mgdl",
    "predicted_glucose_30min",
    "glucose_trend_mgdl_min",
    "carb_intake_grams",
    "carb_grams",
    "patient_iob_units",
    "patient_cob_grams",
    "effective_isf",
    "effective_icr",
    "effective_basal_rate_u_per_hr",
    "steps",
    "calories",
    "heart_rate",
    "sleep_minutes",
    "time_of_day_sin",
    "time_of_day_cos",
    "delivered_insulin_units",
    "input_validator_fail_soft",
    "sensor_status",
]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_run_steps_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "raw" / "steps.csv",
        run_dir / "results.csv",
        run_dir / "research" / "predictor_training.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No raw/steps.csv, results.csv, or research/predictor_training.csv found in {run_dir}")


def _resolve_predictor_source(run_dir: Path) -> Path:
    candidate = run_dir / "research" / "predictor_training.csv"
    return candidate if candidate.is_file() else _resolve_run_steps_path(run_dir)


def _prepare_predictor_frame(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "time_minutes" not in df.columns:
        df["time_minutes"] = np.arange(len(df)) * 5
    if "glucose_actual_mgdl" not in df.columns:
        raise ValueError(f"{path}: missing glucose_actual_mgdl")

    if "carb_grams" not in df.columns:
        source = df["carb_intake_grams"] if "carb_intake_grams" in df.columns else pd.Series(0.0, index=df.index)
        df["carb_grams"] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    if "segment" not in df.columns:
        df["segment"] = 0
    existing_subject = df["subject_id"].astype(str) if "subject_id" in df.columns else pd.Series("subject", index=df.index)
    df["subject_id"] = label + ":" + existing_subject

    day_fraction = pd.to_numeric(df["time_minutes"], errors="coerce").fillna(0.0) % 1440 / 1440.0
    df["time_of_day_sin"] = np.sin(day_fraction * 2.0 * np.pi)
    df["time_of_day_cos"] = np.cos(day_fraction * 2.0 * np.pi)
    for column in PREDICTOR_EXPORT_COLUMNS:
        if column not in df.columns:
            df[column] = "ok" if column == "sensor_status" else 0.0
    return df[PREDICTOR_EXPORT_COLUMNS].copy()


def build_predictor_dataset_from_runs(
    run_dirs: Iterable[Tuple[str, Path]],
    *,
    output_path: Path,
    manifest_path: Path | None = None,
) -> Dict[str, Any]:
    frames: List[pd.DataFrame] = []
    sources: List[Dict[str, Any]] = []
    for label, run_dir in run_dirs:
        source_path = _resolve_predictor_source(run_dir)
        frame = _prepare_predictor_frame(source_path, label)
        frames.append(frame)
        sources.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "source_path": str(source_path),
                "rows": int(len(frame)),
                "sha256": compute_sha256(source_path),
            }
        )
    if not frames:
        raise ValueError("At least one run directory is required.")

    dataset = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    manifest = {
        "rows": int(len(dataset)),
        "subjects": int(dataset["subject_id"].nunique()),
        "columns": PREDICTOR_EXPORT_COLUMNS,
        "sources": sources,
        "output_path": str(output_path),
        "output_sha256": compute_sha256(output_path),
    }
    if manifest_path is not None:
        _write_json(manifest_path, manifest)
    return manifest


def _run_predictor_training(
    *,
    repo_root: Path,
    data_path: Path,
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    training_script = repo_root / "research" / "train_predictor.py"
    rows = int(len(pd.read_csv(data_path)))
    if not training_script.is_file():
        return {"status": "skipped", "reason": f"Training script not found: {training_script}", "rows": rows}
    if rows < 60:
        return {"status": "skipped", "reason": "Need at least 60 predictor rows.", "rows": rows}
    command = [
        sys.executable,
        str(training_script),
        "--data",
        str(data_path),
        "--config",
        str(config_path),
        "--out",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "rows": rows,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_dir": str(output_dir),
    }


def _render_local_ai_report(payload: Dict[str, Any]) -> str:
    lines = [
        "# Local AI Research Lab Report",
        "",
        "This bundle turns one or more IINTS/Jetson runs into local AI training evidence.",
        "",
        "## Datasets",
        "",
        f"- Predictor rows: `{payload['predictor_dataset']['rows']}`",
        f"- Predictor subjects: `{payload['predictor_dataset']['subjects']}`",
        f"- Controller rows: `{payload['controller_dataset']['rows']}`",
        f"- Controller teacher sources: `{payload['controller_dataset'].get('teacher_source_columns', [])}`",
        "",
        "## Models",
        "",
        f"- Linear controller: `{payload['linear_controller']['model_path']}`",
        f"- Neural controller: `{payload['neural_controller'].get('model_path', payload['neural_controller']['status'])}`",
        f"- Glucose predictor: `{payload['predictor_training']['status']}`",
        "",
        "## Validation",
        "",
        f"- Closed-loop evaluation: `{payload['closed_loop_evaluation'].get('status', 'completed')}`",
        f"- Training safety gate: `{payload['training_safety_gate']['status']}`",
    ]
    if payload["closed_loop_evaluation"].get("artifacts"):
        lines.append(f"- Evaluation report: `{payload['closed_loop_evaluation']['artifacts']['report_md']}`")
    if payload["closed_loop_evaluation"].get("safety_gate"):
        lines.append(f"- Closed-loop safety gate: `{payload['closed_loop_evaluation']['safety_gate']['status']}`")
    if payload["training_safety_gate"].get("critical_failures"):
        lines.extend(["", "### Training Gate Critical Failures"])
        lines.extend(f"- {item}" for item in payload["training_safety_gate"]["critical_failures"])
    lines.extend(
        [
            "",
            "## Research Guardrails",
            "",
            "- These models are local research artifacts, not medical-device software.",
            "- The controller learns from a conservative synthetic teacher, then remains wrapped by the deterministic supervisor.",
            "- Predictor evidence from Jetson simulation is useful for pipeline development, but external real-data evaluation is required before making scientific claims.",
            "- Use multiple profiles, multiple seeds, and external datasets before comparing algorithms publicly.",
            "",
            "## Next Commands",
            "",
            "```bash",
            f"iints research evaluate-controller --model {payload['linear_controller']['model_path']} --model-kind linear --output-dir {payload['output_dir']}/evaluation_linear",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_local_ai_lab(
    run_dirs: Iterable[Tuple[str, Path]],
    *,
    output_dir: Path,
    repo_root: Path,
    train_predictor: bool = True,
    predictor_config_path: Path | None = None,
    train_neural: bool = True,
    evaluate: bool = True,
    evaluation_presets: Iterable[str] = DEFAULT_HELD_OUT_PRESETS,
    evaluation_seeds: Iterable[int] = (101, 202, 303),
    evaluation_duration_minutes: int = 1440,
) -> Dict[str, Any]:
    run_dir_list = [(label, Path(path)) for label, path in run_dirs]
    if not run_dir_list:
        raise ValueError("At least one run directory is required.")

    datasets_dir = output_dir / "datasets"
    models_dir = output_dir / "models"
    evaluation_dir = output_dir / "evaluation"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    predictor_manifest = build_predictor_dataset_from_runs(
        run_dir_list,
        output_path=datasets_dir / "predictor_training.csv",
        manifest_path=datasets_dir / "predictor_dataset_manifest.json",
    )
    controller_manifest = build_control_dataset_from_runs(
        run_dir_list,
        output_path=datasets_dir / "controller_teacher_dataset.csv",
        manifest_path=datasets_dir / "controller_dataset_manifest.json",
    )

    controller_df = pd.read_csv(datasets_dir / "controller_teacher_dataset.csv")
    linear_model = train_linear_imitation_controller(controller_df)
    training_gate = review_controller_training_artifacts(
        summarize_control_dataset(controller_df),
        train_metrics=linear_model["train_metrics"],
    ).to_dict()
    linear_model_path = models_dir / "linear_controller.json"
    save_linear_controller(linear_model, linear_model_path)

    neural_payload: Dict[str, Any]
    if train_neural:
        try:
            checkpoint = train_neural_imitation_controller(
                controller_df,
                config=NeuralControllerConfig(epochs=80),
            )
            neural_model_path = models_dir / "neural_controller.pt"
            save_neural_controller(checkpoint, neural_model_path)
            neural_payload = {
                "status": "completed",
                "model_path": str(neural_model_path),
                "train_metrics": checkpoint["train_metrics"],
                "validation_metrics": checkpoint["validation_metrics"],
            }
        except Exception as exc:
            neural_payload = {"status": "failed", "reason": str(exc)}
    else:
        neural_payload = {"status": "skipped", "reason": "Disabled by caller."}

    predictor_config = predictor_config_path or repo_root / "research" / "configs" / "predictor.yaml"
    if train_predictor:
        predictor_result = _run_predictor_training(
            repo_root=repo_root,
            data_path=datasets_dir / "predictor_training.csv",
            config_path=predictor_config,
            output_dir=models_dir / "predictor",
        )
    else:
        predictor_result = {"status": "skipped", "reason": "Disabled by caller."}

    if evaluate:
        factories: Dict[str, ControllerFactory] = {
            "clinical_baseline": ClinicalBaselineAlgorithm,
            "linear_imitation": lambda: ExperimentalImitationController(
                settings={"model_path": str(linear_model_path)}
            ),
        }
        if neural_payload["status"] == "completed":
            factories["neural_controller"] = lambda: ExperimentalNeuralController(
                settings={"model_path": str(neural_payload["model_path"])}
            )
        closed_loop = evaluate_controller_factories(
            factories,
            output_dir=evaluation_dir,
            presets=evaluation_presets,
            seeds=evaluation_seeds,
            duration_minutes=evaluation_duration_minutes,
        )
        closed_loop["status"] = "completed"
    else:
        closed_loop = {"status": "skipped", "reason": "Disabled by caller."}

    dataset_card = {
        "rows": {
            "predictor": predictor_manifest["rows"],
            "controller": controller_manifest["rows"],
        },
        "sources": predictor_manifest["sources"],
        "controller_summary": summarize_control_dataset(controller_df),
        "research_use_only": True,
        "notes": [
            "Predictor data teaches glucose forecasting.",
            "Controller data teaches a research policy from conservative teacher labels.",
            "Do not use these artifacts for medical treatment or pump dosing.",
        ],
    }
    _write_json(datasets_dir / "LOCAL_AI_DATASET_CARD.json", dataset_card)

    payload: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_dirs": [{"label": label, "path": str(path)} for label, path in run_dir_list],
        "predictor_dataset": predictor_manifest,
        "controller_dataset": controller_manifest,
        "dataset_card": str(datasets_dir / "LOCAL_AI_DATASET_CARD.json"),
        "linear_controller": {
            "model_path": str(linear_model_path),
            "train_metrics": linear_model["train_metrics"],
        },
        "training_safety_gate": training_gate,
        "neural_controller": neural_payload,
        "predictor_training": predictor_result,
        "closed_loop_evaluation": closed_loop,
        "artifacts": {
            "predictor_dataset_csv": str(datasets_dir / "predictor_training.csv"),
            "controller_dataset_csv": str(datasets_dir / "controller_teacher_dataset.csv"),
            "dataset_card_json": str(datasets_dir / "LOCAL_AI_DATASET_CARD.json"),
            "summary_json": str(output_dir / "LOCAL_AI_RESEARCH_SUMMARY.json"),
            "report_md": str(output_dir / "LOCAL_AI_RESEARCH_REPORT.md"),
        },
    }
    _write_json(output_dir / "LOCAL_AI_RESEARCH_SUMMARY.json", payload)
    (output_dir / "LOCAL_AI_RESEARCH_REPORT.md").write_text(_render_local_ai_report(payload), encoding="utf-8")
    return payload
