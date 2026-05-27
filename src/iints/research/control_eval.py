from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from iints.api.base_algorithm import InsulinAlgorithm
from iints.core.devices.models import create_sensor_model
from iints.core.patient.patient_factory import PatientFactory
from iints.core.simulator import Simulator
from iints.presets import get_preset
from iints.validation import build_stress_events, load_patient_config_by_name
from .local_ai_gate import review_closed_loop_evaluation


DEFAULT_HELD_OUT_PRESETS = (
    "hypo_prone_night",
    "hyper_challenge",
    "pizza_paradox",
    "midnight_crash",
)

ControllerFactory = Callable[[], InsulinAlgorithm]


def _run_metrics(
    df: pd.DataFrame,
    safety_report: Dict[str, Any],
    *,
    requested_duration_minutes: int,
    time_step_minutes: int,
) -> Dict[str, Any]:
    if df.empty:
        return {
            "steps": 0,
            "completed_duration_minutes": 0,
            "completion_pct": 0.0,
            "tir_70_180_pct": 0.0,
            "time_below_70_pct": 0.0,
            "time_below_54_pct": 0.0,
            "mean_glucose_mgdl": None,
            "critical_rows_below_54": 0,
            "supervisor_interventions": 0,
            "supervisor_intervention_rate_pct": 0.0,
            "terminated_early": bool(safety_report.get("terminated_early", False)),
        }
    glucose = pd.to_numeric(df["glucose_actual_mgdl"], errors="coerce")
    interventions = df.get("safety_triggered", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    completed_duration = int(df["time_minutes"].max()) + time_step_minutes
    return {
        "steps": int(len(df)),
        "completed_duration_minutes": completed_duration,
        "completion_pct": round((completed_duration / requested_duration_minutes) * 100.0, 4),
        "tir_70_180_pct": round(float(((glucose >= 70.0) & (glucose <= 180.0)).mean() * 100.0), 4),
        "time_below_70_pct": round(float((glucose < 70.0).mean() * 100.0), 4),
        "time_below_54_pct": round(float((glucose < 54.0).mean() * 100.0), 4),
        "mean_glucose_mgdl": round(float(glucose.mean()), 4),
        "critical_rows_below_54": int((glucose < 54.0).sum()),
        "supervisor_interventions": int(interventions.sum()),
        "supervisor_intervention_rate_pct": round(float(interventions.mean() * 100.0), 4),
        "terminated_early": bool(safety_report.get("terminated_early", False)),
    }


def _run_controller_once(
    factory: ControllerFactory,
    *,
    preset_name: str,
    seed: int,
    duration_minutes: int,
    time_step_minutes: int,
    sensor_profile: str,
) -> Dict[str, Any]:
    preset = get_preset(preset_name)
    patient_config = load_patient_config_by_name(str(preset["patient_config"]))
    patient_model = PatientFactory.create_patient(
        patient_type="custom",
        **patient_config.model_dump(),
    )
    simulator = Simulator(
        patient_model=patient_model,
        algorithm=factory(),
        time_step=time_step_minutes,
        seed=seed,
        sensor_model=create_sensor_model(profile=sensor_profile, seed=seed),
    )
    for event in build_stress_events(preset["scenario"].get("stress_events", [])):
        simulator.add_stress_event(event)
    df, safety_report = simulator.run_batch(duration_minutes)
    metrics = _run_metrics(
        df,
        safety_report,
        requested_duration_minutes=duration_minutes,
        time_step_minutes=time_step_minutes,
    )
    metrics.update(
        {
            "preset": preset_name,
            "patient_config": preset["patient_config"],
            "seed": seed,
        }
    )
    return metrics


def _summarize_runs(run_df: pd.DataFrame) -> Dict[str, Any]:
    metric_columns = [
        "tir_70_180_pct",
        "time_below_70_pct",
        "time_below_54_pct",
        "mean_glucose_mgdl",
        "supervisor_interventions",
        "supervisor_intervention_rate_pct",
        "completion_pct",
        "terminated_early",
    ]
    summary: dict[str, dict[str, Any]] = {}
    for algorithm_name, frame in run_df.groupby("algorithm", sort=True):
        row: dict[str, Any] = {"runs": int(len(frame))}
        for metric in metric_columns:
            series = pd.to_numeric(frame[metric], errors="coerce")
            if metric == "terminated_early":
                row["terminated_early_runs"] = int(series.sum())
            else:
                row[f"mean_{metric}"] = round(float(series.mean()), 6)
        summary[str(algorithm_name)] = row
    baseline = summary.get("clinical_baseline")
    if baseline is not None:
        for algorithm_name, row in summary.items():
            if algorithm_name == "clinical_baseline":
                continue
            row["delta_vs_clinical_baseline"] = {
                "tir_70_180_pct": round(
                    row["mean_tir_70_180_pct"] - baseline["mean_tir_70_180_pct"],
                    6,
                ),
                "time_below_70_pct": round(
                    row["mean_time_below_70_pct"] - baseline["mean_time_below_70_pct"],
                    6,
                ),
                "time_below_54_pct": round(
                    row["mean_time_below_54_pct"] - baseline["mean_time_below_54_pct"],
                    6,
                ),
                "supervisor_intervention_rate_pct": round(
                    row["mean_supervisor_intervention_rate_pct"]
                    - baseline["mean_supervisor_intervention_rate_pct"],
                    6,
                ),
            }
    return summary


def _render_markdown_report(summary: Dict[str, Any], run_df: pd.DataFrame, safety_gate: Dict[str, Any]) -> str:
    lines = [
        "# Closed-Loop Controller Evaluation",
        "",
        "Research-only held-out evaluation over unseen preset/seed combinations.",
        "",
        "## Aggregate Results",
        "",
        "| Algorithm | Runs | TIR 70-180 (%) | <70 (%) | <54 (%) | Supervisor rate (%) | Early terminations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for algorithm_name, row in summary.items():
        lines.append(
            "| "
            f"{algorithm_name} | {row['runs']} | {row['mean_tir_70_180_pct']:.2f} | "
            f"{row['mean_time_below_70_pct']:.2f} | {row['mean_time_below_54_pct']:.2f} | "
            f"{row['mean_supervisor_intervention_rate_pct']:.2f} | {row['terminated_early_runs']} |"
        )
    lines.extend(
        [
            "",
            "## Run Matrix",
            "",
            "| Algorithm | Preset | Seed | TIR (%) | <70 (%) | <54 (%) | Supervisor interventions | Completion (%) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in run_df.sort_values(["algorithm", "preset", "seed"]).iterrows():
        lines.append(
            "| "
            f"{row['algorithm']} | {row['preset']} | {int(row['seed'])} | "
            f"{row['tir_70_180_pct']:.2f} | {row['time_below_70_pct']:.2f} | {row['time_below_54_pct']:.2f} | "
            f"{int(row['supervisor_interventions'])} | {row['completion_pct']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Safety Gate",
            "",
            f"- Status: `{safety_gate['status']}`",
            f"- Passed: `{safety_gate['passed']}`",
            f"- Score: `{safety_gate['score']}`",
        ]
    )
    if safety_gate.get("critical_failures"):
        lines.extend(["", "### Critical Failures"])
        lines.extend(f"- {item}" for item in safety_gate["critical_failures"])
    if safety_gate.get("warnings"):
        lines.extend(["", "### Warnings"])
        lines.extend(f"- {item}" for item in safety_gate["warnings"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Higher TIR is better only when hypo burden and supervisor burden do not worsen.",
            "- `time_below_54_pct` is the first safety metric to inspect before claiming controller improvement.",
            "- This report is pre-clinical simulator evidence, not a clinical dosing claim.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_controller_factories(
    factories: Dict[str, ControllerFactory],
    *,
    output_dir: Path,
    presets: Iterable[str] = DEFAULT_HELD_OUT_PRESETS,
    seeds: Iterable[int] = (101, 202, 303),
    duration_minutes: int = 1440,
    time_step_minutes: int = 5,
    sensor_profile: str = "clinical_cgm",
) -> Dict[str, Any]:
    if "clinical_baseline" not in factories:
        raise ValueError("factories must include a clinical_baseline entry.")
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be greater than zero.")

    rows: list[dict[str, Any]] = []
    preset_list = list(presets)
    seed_list = [int(seed) for seed in seeds]
    for algorithm_name, factory in factories.items():
        for preset_name in preset_list:
            for seed in seed_list:
                metrics = _run_controller_once(
                    factory,
                    preset_name=preset_name,
                    seed=seed,
                    duration_minutes=duration_minutes,
                    time_step_minutes=time_step_minutes,
                    sensor_profile=sensor_profile,
                )
                metrics["algorithm"] = algorithm_name
                rows.append(metrics)

    run_df = pd.DataFrame(rows)
    summary = _summarize_runs(run_df)
    safety_gate = review_closed_loop_evaluation(summary).to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "closed_loop_runs.csv"
    summary_path = output_dir / "closed_loop_summary.json"
    report_path = output_dir / "CONTROL_EVALUATION_REPORT.md"
    run_df.to_csv(runs_path, index=False)
    payload = {
        "presets": preset_list,
        "seeds": seed_list,
        "duration_minutes": duration_minutes,
        "time_step_minutes": time_step_minutes,
        "sensor_profile": sensor_profile,
        "algorithms": summary,
        "safety_gate": safety_gate,
        "artifacts": {
            "runs_csv": str(runs_path),
            "summary_json": str(summary_path),
            "report_md": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(_render_markdown_report(summary, run_df, safety_gate), encoding="utf-8")
    return payload
