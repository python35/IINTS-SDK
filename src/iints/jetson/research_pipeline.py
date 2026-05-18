from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.core.algorithms.imitation_controller import ExperimentalImitationController
from iints.core.algorithms.neural_controller import ExperimentalNeuralController
from iints.research.control import save_linear_controller, train_linear_imitation_controller
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_predictor(
    *,
    repo_root: Path,
    data_path: Path,
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    training_script = repo_root / "research" / "train_predictor.py"
    if not training_script.is_file():
        return {
            "status": "skipped",
            "reason": f"Predictor training script not found: {training_script}",
        }
    min_rows = 60
    rows = int(len(pd.read_csv(data_path)))
    if rows < min_rows:
        return {
            "status": "skipped",
            "reason": f"Need at least {min_rows} predictor rows, found {rows}.",
            "rows": rows,
        }
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
    completed = subprocess.run(  # noqa: S603 - command is assembled from trusted local paths.
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


def _render_research_report(payload: Dict[str, Any]) -> str:
    predictor = payload["predictor_training"]
    neural = payload["neural_controller"]
    lines = [
        "# Jetson Research Finalization Report",
        "",
        "This report closes the post-run research loop for one endurance bundle.",
        "",
        "## Outputs",
        "",
        f"- Controller dataset rows: `{payload['controller_dataset_rows']}`",
        f"- Linear controller: `{payload['linear_controller']['model_path']}`",
        f"- Neural controller: `{neural.get('model_path', neural['status'])}`",
        f"- Predictor training: `{predictor['status']}`",
        f"- Closed-loop evaluation: `{payload['closed_loop_evaluation']['artifacts']['report_md']}`",
        "",
        "## Controller Training",
        "",
        f"- Linear MAE: `{payload['linear_controller']['train_metrics']['mae_units']}` U",
    ]
    if neural["status"] == "completed":
        validation_metrics = neural.get("validation_metrics") or {}
        lines.extend(
            [
                f"- Neural train MAE: `{neural['train_metrics']['mae_units']}` U",
                f"- Neural validation MAE: `{validation_metrics.get('mae_units', 'n/a')}` U",
            ]
        )
    else:
        lines.append(f"- Neural controller: `{neural['status']}` ({neural.get('reason', 'not available')})")
    lines.extend(
        [
            "",
            "## Predictor Training",
            "",
            f"- Status: `{predictor['status']}`",
        ]
    )
    if predictor.get("reason"):
        lines.append(f"- Reason: `{predictor['reason']}`")
    if predictor.get("output_dir"):
        lines.append(f"- Output directory: `{predictor['output_dir']}`")
    lines.extend(
        [
            "",
            "## Scientific Guardrails",
            "",
            "- The endurance run is a simulator-derived acquisition bundle; predictor claims still require external real-data evaluation.",
            "- Controller promotion should depend on the held-out closed-loop report, not only training-set fit.",
            "- Every learned policy remains wrapped by the deterministic supervisor.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_endurance_research(
    output_dir: Path,
    *,
    repo_root: Path,
    train_predictor: bool = True,
    predictor_config_path: Path | None = None,
    train_neural: bool = True,
    evaluation_presets: Iterable[str] = DEFAULT_HELD_OUT_PRESETS,
    evaluation_seeds: Iterable[int] = (101, 202, 303),
    evaluation_duration_minutes: int = 1440,
) -> Dict[str, Any]:
    research_dir = output_dir / "research"
    controller_data_path = research_dir / "controller_teacher_dataset.csv"
    predictor_data_path = research_dir / "predictor_training.csv"
    if not controller_data_path.is_file():
        raise FileNotFoundError(f"Controller teacher dataset not found: {controller_data_path}")
    if not predictor_data_path.is_file():
        raise FileNotFoundError(f"Predictor training dataset not found: {predictor_data_path}")

    models_dir = research_dir / "models"
    evaluation_dir = research_dir / "evaluation"
    controller_df = pd.read_csv(controller_data_path)
    linear_model = train_linear_imitation_controller(controller_df)
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

    factories: dict[str, ControllerFactory] = {
        "clinical_baseline": ClinicalBaselineAlgorithm,
        "linear_imitation": lambda: ExperimentalImitationController(
            settings={"model_path": str(linear_model_path)}
        ),
    }
    if neural_payload["status"] == "completed":
        factories["neural_controller"] = lambda: ExperimentalNeuralController(
            settings={"model_path": str(neural_payload["model_path"])}
        )
    evaluation = evaluate_controller_factories(
        factories,
        output_dir=evaluation_dir,
        presets=evaluation_presets,
        seeds=evaluation_seeds,
        duration_minutes=evaluation_duration_minutes,
    )

    if train_predictor:
        predictor_result = _train_predictor(
            repo_root=repo_root,
            data_path=predictor_data_path,
            config_path=predictor_config_path or repo_root / "research" / "configs" / "predictor.yaml",
            output_dir=models_dir / "predictor",
        )
    else:
        predictor_result = {"status": "skipped", "reason": "Disabled by caller."}

    payload = {
        "output_dir": str(output_dir),
        "controller_dataset_rows": int(len(controller_df)),
        "linear_controller": {
            "model_path": str(linear_model_path),
            "train_metrics": linear_model["train_metrics"],
        },
        "neural_controller": neural_payload,
        "predictor_training": predictor_result,
        "closed_loop_evaluation": evaluation,
    }
    summary_path = research_dir / "research_pipeline_summary.json"
    report_path = research_dir / "RESEARCH_PIPELINE_REPORT.md"
    payload["artifacts"] = {
        "summary_json": str(summary_path),
        "report_md": str(report_path),
    }
    _write_json(summary_path, payload)
    report_path.write_text(_render_research_report(payload), encoding="utf-8")
    return payload
