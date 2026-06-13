from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

from iints.research.glucose_model import glucose_model_config_payload, write_huggingface_export_bundle


JETSON_HF_LEADERBOARD_FIELDS = [
    "timestamp_utc",
    "trial_id",
    "status",
    "repo_id",
    "warm_start_path",
    "candidate_path",
    "score_metric",
    "score",
    "base_score",
    "candidate_mae",
    "candidate_rmse",
    "candidate_missed_hypo_rate_pct",
    "candidate_any_physiology_violation_pct",
    "base_mae",
    "base_rmse",
    "base_missed_hypo_rate_pct",
    "base_any_physiology_violation_pct",
    "accepted",
    "learning_rate",
    "pinn_lambda",
    "weight_decay",
    "epochs",
    "batch_size",
    "duration_sec",
    "trial_dir",
    "comparison_dir",
    "error",
]


@dataclass(frozen=True)
class JetsonHFTrainingResult:
    repo_id: Optional[str]
    work_dir: Path
    base_dir: Path
    champion_dir: Path
    leaderboard: Path
    trial_count: int
    accepted_count: int
    best_score: Optional[float]
    upload_mode: str
    uploaded: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def finite_float(value: Any, default: float = math.inf) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def model_score(row: Mapping[str, Any], *, physiology_weight: float = 0.10, hypo_weight: float = 0.20) -> float:
    """Composite lower-is-better score for glucose predictor promotion."""
    mae = finite_float(row.get("mae"))
    if not math.isfinite(mae):
        return math.inf
    physiology = finite_float(row.get("any_physiology_violation_pct"), 0.0)
    missed_hypo = finite_float(row.get("missed_hypo_rate_pct"), 0.0)
    return mae + physiology_weight * physiology + hypo_weight * missed_hypo


def append_leaderboard_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JETSON_HF_LEADERBOARD_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in JETSON_HF_LEADERBOARD_FIELDS})


def run_logged(
    cmd: Sequence[str],
    *,
    log_path: Path,
    env: Optional[Mapping[str, str]] = None,
    timeout_minutes: Optional[float] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        subprocess.run(
            list(cmd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=None if timeout_minutes is None else max(60.0, float(timeout_minutes) * 60.0),
            env=dict(env) if env is not None else None,
        )


def _hf_env(hf_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    _apply_jetson_env_defaults(env)
    env["HF_HOME"] = str(hf_home)
    return env


def _apply_jetson_env_defaults(env: Optional[dict[str, str]] = None) -> None:
    target = os.environ if env is None else env
    target.setdefault("OMP_NUM_THREADS", "1")
    target.setdefault("MKL_NUM_THREADS", "1")
    target.setdefault("OPENBLAS_NUM_THREADS", "1")
    target.setdefault("NUMEXPR_NUM_THREADS", "1")
    target.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    target.setdefault("MPLBACKEND", "Agg")


def _download_hf_model(
    *,
    repo_id: str,
    revision: Optional[str],
    output_dir: Path,
    hf_home: Path,
    force: bool,
) -> None:
    if output_dir.exists() and not force and (output_dir / "predictor.pt").exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        repo_id,
        "--type",
        "model",
        "--local-dir",
        str(output_dir),
        "--include",
        "predictor.pt",
        "--include",
        "glucose_model_config.yaml",
        "--include",
        "config.json",
        "--include",
        "training_report.json",
        "--include",
        "dataset_manifest.public.json",
    ]
    if revision:
        cmd.extend(["--revision", revision])
    run_logged(cmd, log_path=output_dir / "hf_download.log", env=_hf_env(hf_home), timeout_minutes=30)


def _load_yaml_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _checkpoint_config(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    config = payload.get("config")
    return dict(config) if isinstance(config, Mapping) else {}


def _base_config_payload(base_dir: Path, profile: str, checkpoint_config: Mapping[str, Any]) -> dict[str, Any]:
    payload = (
        _load_yaml_if_exists(base_dir / "glucose_model_config.yaml")
        or _load_yaml_if_exists(base_dir / "glucose_model_config.resolved.yaml")
        or glucose_model_config_payload(profile=profile)
    )
    predictor = payload.setdefault("predictor", {})
    training = payload.setdefault("training", {})
    if "feature_columns" in checkpoint_config:
        predictor["feature_columns"] = list(checkpoint_config["feature_columns"])
    if "target_column" in checkpoint_config:
        predictor["target_column"] = str(checkpoint_config["target_column"])
    if "history_steps" in checkpoint_config and "time_step_minutes" in checkpoint_config:
        predictor["history_minutes"] = int(checkpoint_config["history_steps"]) * int(checkpoint_config["time_step_minutes"])
    if "horizon_steps" in checkpoint_config and "time_step_minutes" in checkpoint_config:
        predictor["horizon_minutes"] = int(checkpoint_config["horizon_steps"]) * int(checkpoint_config["time_step_minutes"])
    if "time_step_minutes" in checkpoint_config:
        predictor["time_step_minutes"] = int(checkpoint_config["time_step_minutes"])
    if "hidden_size" in checkpoint_config:
        training["hidden_size"] = int(checkpoint_config["hidden_size"])
    if "num_layers" in checkpoint_config:
        training["num_layers"] = int(checkpoint_config["num_layers"])
    if "dropout" in checkpoint_config:
        training["dropout"] = float(checkpoint_config["dropout"])
    return payload


def _trial_config(
    base_payload: Mapping[str, Any],
    *,
    rng: random.Random,
    trial_index: int,
    seed: int,
    epochs: int,
    batch_size: int,
    min_lr: float,
    max_lr: float,
    min_pinn_lambda: float,
    max_pinn_lambda: float,
    weight_decay_choices: Sequence[float],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    training = payload.setdefault("training", {})
    training["epochs"] = int(epochs)
    training["batch_size"] = int(batch_size)
    training["learning_rate"] = round(10 ** rng.uniform(math.log10(min_lr), math.log10(max_lr)), 7)
    training["pinn_lambda"] = round(rng.uniform(min_pinn_lambda, max_pinn_lambda), 4)
    training["weight_decay"] = float(rng.choice(list(weight_decay_choices)))
    training["seed"] = int(seed + trial_index * 1009)
    training["early_stopping_patience"] = int(training.get("early_stopping_patience") or 4)
    training["early_stopping_min_delta"] = float(training.get("early_stopping_min_delta") or 0.0005)
    payload.setdefault("iints_glucose_model", {})["jetson_hf_trial"] = True
    return payload


def _copy_champion_seed(base_dir: Path, champion_dir: Path) -> None:
    if (champion_dir / "predictor.pt").exists():
        return
    champion_dir.mkdir(parents=True, exist_ok=True)
    for name in ("predictor.pt", "glucose_model_config.yaml", "glucose_model_config.resolved.yaml", "training_report.json"):
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, champion_dir / name)
    metadata = {
        "seeded_utc": utc_now(),
        "source": str(base_dir),
        "research_only": True,
        "not_for_treatment": True,
    }
    (champion_dir / "champion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _promote_candidate(
    *,
    candidate_dir: Path,
    config_path: Path,
    comparison_dir: Path,
    champion_dir: Path,
    decision: Mapping[str, Any],
) -> None:
    tmp_dir = champion_dir.with_name(champion_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for name in ("predictor.pt", "training_report.json", "glucose_model_config.resolved.yaml"):
        src = candidate_dir / name
        if src.exists():
            shutil.copy2(src, tmp_dir / name)
    shutil.copy2(config_path, tmp_dir / "glucose_model_config.yaml")
    if comparison_dir.exists():
        shutil.copytree(comparison_dir, tmp_dir / "comparison", dirs_exist_ok=True)
    (tmp_dir / "champion_metadata.json").write_text(json.dumps(dict(decision), indent=2), encoding="utf-8")
    if champion_dir.exists():
        shutil.rmtree(champion_dir)
    tmp_dir.rename(champion_dir)


def _comparison_row(report: Mapping[str, Any], label: str) -> Optional[dict[str, Any]]:
    models = report.get("models")
    if not isinstance(models, list):
        return None
    for row in models:
        if isinstance(row, Mapping) and row.get("model") == label:
            return dict(row)
    return None


def _run_compare(
    *,
    dataset: Path,
    config_path: Path,
    output_dir: Path,
    base_checkpoint: Path,
    candidate_checkpoint: Path,
    timeout_minutes: float,
) -> dict[str, Any]:
    cmd = [
        "iints",
        "research",
        "glucose-model",
        "compare",
        "--data",
        str(dataset),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--model",
        f"current_champion={base_checkpoint}",
        "--model",
        f"jetson_candidate={candidate_checkpoint}",
        "--no-baselines",
    ]
    run_logged(cmd, log_path=output_dir / "compare_stdout_stderr.log", timeout_minutes=timeout_minutes)
    report_path = output_dir / "comparison_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def _train_one_trial(
    *,
    dataset: Path,
    trial_dir: Path,
    config_path: Path,
    warm_start: Path,
    timeout_minutes: float,
) -> None:
    cmd = [
        "iints",
        "research",
        "glucose-model",
        "train",
        "--data",
        str(dataset),
        "--config",
        str(config_path),
        "--output-dir",
        str(trial_dir),
        "--warm-start",
        str(warm_start),
        "--no-export-hf",
    ]
    run_logged(cmd, log_path=trial_dir / "train_stdout_stderr.log", timeout_minutes=timeout_minutes)


def _upload_hf_bundle(
    *,
    repo_id: str,
    bundle_dir: Path,
    hf_home: Path,
    upload_mode: str,
    private: bool,
) -> bool:
    if upload_mode == "none":
        return False
    cmd = ["hf", "upload", repo_id, str(bundle_dir), ".", "--type", "model"]
    if private:
        cmd.append("--private")
    if upload_mode == "pr":
        cmd.append("--create-pr")
    cmd.extend(["--commit-message", "Update IINTS-AF glucose forecast candidate from Jetson training"])
    run_logged(cmd, log_path=bundle_dir / "hf_upload.log", env=_hf_env(hf_home), timeout_minutes=60)
    return True


def _iter_weight_decay(values: Optional[Iterable[float]]) -> list[float]:
    parsed = [float(value) for value in values] if values is not None else [0.0, 1e-6, 1e-5, 1e-4]
    return parsed or [0.0]


def run_jetson_hf_training(
    *,
    repo_id: Optional[str],
    dataset: Path,
    work_dir: Path,
    local_base_dir: Optional[Path] = None,
    revision: Optional[str] = None,
    profile: str = "quick",
    max_trials: int = 1,
    timeout_minutes: float = 45.0,
    cooldown_seconds: float = 10.0,
    epochs: int = 8,
    batch_size: int = 64,
    seed: int = 42,
    min_lr: float = 1e-5,
    max_lr: float = 5e-4,
    min_pinn_lambda: float = 0.05,
    max_pinn_lambda: float = 0.8,
    weight_decay_choices: Optional[Iterable[float]] = None,
    min_score_improvement: float = 0.0,
    physiology_weight: float = 0.10,
    hypo_weight: float = 0.20,
    dataset_manifest: Optional[Path] = None,
    upload_mode: str = "none",
    private_upload: bool = True,
    force_download: bool = False,
    hf_home: Optional[Path] = None,
) -> JetsonHFTrainingResult:
    _apply_jetson_env_defaults()
    if repo_id is None and local_base_dir is None:
        raise ValueError("Provide --repo-id or --local-base-dir so the Jetson has a warm-start model.")
    if upload_mode not in {"none", "pr", "direct"}:
        raise ValueError("upload_mode must be one of: none, pr, direct")
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    if shutil.which("iints") is None:
        raise RuntimeError("The 'iints' CLI is not on PATH. Activate the SDK venv first.")
    if repo_id and shutil.which("hf") is None:
        raise RuntimeError("The 'hf' CLI is not on PATH. Install/login with Hugging Face CLI first.")

    work_dir.mkdir(parents=True, exist_ok=True)
    hf_home = hf_home or work_dir / ".hf_home"
    base_dir = local_base_dir or work_dir / "hf_base"
    trials_dir = work_dir / "trials"
    champion_dir = work_dir / "champion"
    leaderboard = work_dir / "jetson_hf_leaderboard.csv"
    trials_dir.mkdir(parents=True, exist_ok=True)

    if repo_id and local_base_dir is None:
        _download_hf_model(
            repo_id=repo_id,
            revision=revision,
            output_dir=base_dir,
            hf_home=hf_home,
            force=force_download,
        )
    if not (base_dir / "predictor.pt").exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {base_dir / 'predictor.pt'}")

    checkpoint_config = _checkpoint_config(base_dir / "predictor.pt")
    base_payload = _base_config_payload(base_dir, profile, checkpoint_config)
    _copy_champion_seed(base_dir, champion_dir)

    rng = random.Random(seed)
    accepted_count = 0
    uploaded = False
    best_score: Optional[float] = None
    trial_index = 0

    try:
        while max_trials <= 0 or trial_index < max_trials:
            trial_index += 1
            trial_id = f"trial_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{trial_index:05d}"
            trial_dir = trials_dir / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            warm_start = champion_dir / "predictor.pt"
            config = _trial_config(
                base_payload,
                rng=rng,
                trial_index=trial_index,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                min_lr=min_lr,
                max_lr=max_lr,
                min_pinn_lambda=min_pinn_lambda,
                max_pinn_lambda=max_pinn_lambda,
                weight_decay_choices=_iter_weight_decay(weight_decay_choices),
            )
            config_path = trial_dir / "trial_config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            started = time.time()
            status = "success"
            error = ""
            accepted = False
            candidate_score = math.inf
            base_score = math.inf
            base_row: dict[str, Any] = {}
            candidate_row: dict[str, Any] = {}
            comparison_dir = trial_dir / "comparison"
            try:
                _train_one_trial(
                    dataset=dataset,
                    trial_dir=trial_dir,
                    config_path=config_path,
                    warm_start=warm_start,
                    timeout_minutes=timeout_minutes,
                )
                comparison = _run_compare(
                    dataset=dataset,
                    config_path=config_path,
                    output_dir=comparison_dir,
                    base_checkpoint=warm_start,
                    candidate_checkpoint=trial_dir / "predictor.pt",
                    timeout_minutes=timeout_minutes,
                )
                base_row = _comparison_row(comparison, "current_champion") or {}
                candidate_row = _comparison_row(comparison, "jetson_candidate") or {}
                base_score = model_score(base_row, physiology_weight=physiology_weight, hypo_weight=hypo_weight)
                candidate_score = model_score(
                    candidate_row,
                    physiology_weight=physiology_weight,
                    hypo_weight=hypo_weight,
                )
                best_score = base_score if best_score is None else min(best_score, base_score)
                accepted = candidate_score + min_score_improvement < base_score
                if accepted:
                    accepted_count += 1
                    best_score = candidate_score
                    decision = {
                        "promoted_utc": utc_now(),
                        "repo_id": repo_id,
                        "trial_id": trial_id,
                        "score_metric": "mae + physiology/hypo penalties",
                        "candidate_score": candidate_score,
                        "previous_champion_score": base_score,
                        "candidate_metrics": candidate_row,
                        "previous_champion_metrics": base_row,
                        "research_only": True,
                        "not_for_treatment": True,
                    }
                    _promote_candidate(
                        candidate_dir=trial_dir,
                        config_path=config_path,
                        comparison_dir=comparison_dir,
                        champion_dir=champion_dir,
                        decision=decision,
                    )
                    hf_outputs = write_huggingface_export_bundle(
                        model_dir=champion_dir,
                        output_dir=champion_dir / "huggingface",
                        repo_id=repo_id,
                        dataset_manifest=dataset_manifest,
                        comparison_dir=champion_dir / "comparison",
                    )
                    if repo_id and upload_mode != "none":
                        uploaded = _upload_hf_bundle(
                            repo_id=repo_id,
                            bundle_dir=Path(hf_outputs["output_dir"]),
                            hf_home=hf_home,
                            upload_mode=upload_mode,
                            private=private_upload,
                        ) or uploaded
            except subprocess.TimeoutExpired:
                status = "failed"
                error = f"timeout_after_{timeout_minutes:g}_minutes"
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error = f"exit_code_{exc.returncode}"
            except Exception as exc:
                status = "failed"
                error = str(exc)

            training = config.get("training", {})
            append_leaderboard_row(
                leaderboard,
                {
                    "timestamp_utc": utc_now(),
                    "trial_id": trial_id,
                    "status": status,
                    "repo_id": repo_id or "",
                    "warm_start_path": str(warm_start),
                    "candidate_path": str(trial_dir / "predictor.pt"),
                    "score_metric": "mae + physiology/hypo penalties",
                    "score": "" if not math.isfinite(candidate_score) else candidate_score,
                    "base_score": "" if not math.isfinite(base_score) else base_score,
                    "candidate_mae": candidate_row.get("mae", ""),
                    "candidate_rmse": candidate_row.get("rmse", ""),
                    "candidate_missed_hypo_rate_pct": candidate_row.get("missed_hypo_rate_pct", ""),
                    "candidate_any_physiology_violation_pct": candidate_row.get(
                        "any_physiology_violation_pct", ""
                    ),
                    "base_mae": base_row.get("mae", ""),
                    "base_rmse": base_row.get("rmse", ""),
                    "base_missed_hypo_rate_pct": base_row.get("missed_hypo_rate_pct", ""),
                    "base_any_physiology_violation_pct": base_row.get("any_physiology_violation_pct", ""),
                    "accepted": accepted,
                    "learning_rate": training.get("learning_rate", ""),
                    "pinn_lambda": training.get("pinn_lambda", ""),
                    "weight_decay": training.get("weight_decay", ""),
                    "epochs": training.get("epochs", ""),
                    "batch_size": training.get("batch_size", ""),
                    "duration_sec": round(time.time() - started, 2),
                    "trial_dir": str(trial_dir),
                    "comparison_dir": str(comparison_dir),
                    "error": error,
                },
            )
            if max_trials <= 0 or trial_index < max_trials:
                time.sleep(max(0.0, float(cooldown_seconds)))
    except KeyboardInterrupt:
        pass

    return JetsonHFTrainingResult(
        repo_id=repo_id,
        work_dir=work_dir,
        base_dir=base_dir,
        champion_dir=champion_dir,
        leaderboard=leaderboard,
        trial_count=trial_index,
        accepted_count=accepted_count,
        best_score=best_score,
        upload_mode=upload_mode,
        uploaded=uploaded,
    )
