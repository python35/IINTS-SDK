#!/usr/bin/env python3
"""Jetson-safe AutoML loop for IINTS glucose forecast experiments.

This script is intentionally conservative: every model is trained in a child
process, failed trials are logged, and only a fully written model/report can be
promoted to champion. It is research tooling, not a treatment system.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

# Jetson Nano safety constraints. Keep these before any torch subprocess starts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from iints.research.glucose_model import glucose_model_config_payload
except Exception:  # pragma: no cover - gives a friendly runtime error in main().
    glucose_model_config_payload = None  # type: ignore[assignment]

DEFAULT_DATASET = Path("models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.parquet")
DEFAULT_TRIALS_DIR = Path("models/jetson_trials")
DEFAULT_CHAMPION_DIR = Path("models/jetson_champion")
DEFAULT_LEADERBOARD = Path("jetson_leaderboard.csv")

LEADERBOARD_FIELDS = [
    "timestamp_utc",
    "trial_id",
    "status",
    "score_metric",
    "score",
    "test_rmse",
    "test_mae",
    "val_loss_final",
    "train_loss_final",
    "learning_rate",
    "hidden_size",
    "num_layers",
    "dropout",
    "pinn_lambda",
    "loss",
    "epochs",
    "batch_size",
    "duration_sec",
    "is_champion",
    "trial_dir",
    "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def finite_float(value: Any, default: float = math.inf) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def score_from_report(report: Mapping[str, Any]) -> tuple[str, float]:
    """Return the metric used for champion selection. Lower is better."""
    for key in ("test_rmse", "val_loss_final", "test_mae", "train_loss_final"):
        value = finite_float(report.get(key))
        if math.isfinite(value):
            return key, value
    return "unavailable", math.inf


def load_best_score(leaderboard: Path) -> float:
    if not leaderboard.exists():
        return math.inf
    best = math.inf
    with leaderboard.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("status", "")).lower() != "success":
                continue
            score = finite_float(row.get("score"))
            if score < best:
                best = score
    return best


def append_leaderboard_row(leaderboard: Path, row: Mapping[str, Any]) -> None:
    leaderboard.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not leaderboard.exists() or leaderboard.stat().st_size == 0
    with leaderboard.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LEADERBOARD_FIELDS})


def preflight(dataset: Path, trials_dir: Path, champion_dir: Path) -> None:
    if glucose_model_config_payload is None:
        raise SystemExit(
            "Could not import the installed IINTS SDK. Activate your venv and run from the SDK repo: "
            "source .venv/bin/activate && python scratch/jetson_automl_trainer.py"
        )
    if shutil.which("iints") is None:
        raise SystemExit("The 'iints' CLI is not on PATH. Activate the SDK venv or run: pip install -e .")
    if not dataset.exists():
        raise SystemExit(
            f"Dataset not found: {dataset}\n"
            "Build it first with `iints research glucose-model build-dataset ...`, "
            "or pass --dataset path/to/glucose_training_dataset.parquet."
        )
    trials_dir.mkdir(parents=True, exist_ok=True)
    champion_dir.mkdir(parents=True, exist_ok=True)


def build_trial_config(args: argparse.Namespace, rng: random.Random, trial_dir: Path, trial_index: int) -> dict[str, Any]:
    assert glucose_model_config_payload is not None
    cfg = glucose_model_config_payload(profile="quick")
    training = cfg.setdefault("training", {})
    training["epochs"] = int(args.epochs)
    training["batch_size"] = int(args.batch_size)
    training["learning_rate"] = round(10 ** rng.uniform(math.log10(args.min_lr), math.log10(args.max_lr)), 6)
    training["hidden_size"] = int(rng.choice(args.hidden_sizes))
    training["num_layers"] = int(rng.choice(args.num_layers_choices))
    training["dropout"] = round(rng.uniform(args.min_dropout, args.max_dropout), 3)
    training["pinn_lambda"] = round(rng.uniform(args.min_pinn_lambda, args.max_pinn_lambda), 3)
    training["seed"] = int(args.seed + trial_index * 1009)
    training["early_stopping_patience"] = min(int(training.get("early_stopping_patience", 5)), int(args.early_stopping_patience))
    training["early_stopping_min_delta"] = float(args.early_stopping_min_delta)
    cfg.setdefault("iints_glucose_model", {})["automl_trial_dir"] = str(trial_dir)
    return cfg


def run_trial(dataset: Path, config_path: Path, trial_dir: Path, timeout_minutes: float) -> tuple[bool, str]:
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
        "--no-export-hf",
    ]
    log_path = trial_dir / "train_stdout_stderr.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        try:
            subprocess.run(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=max(60.0, timeout_minutes * 60.0),
            )
            return True, ""
        except subprocess.TimeoutExpired:
            return False, f"timeout_after_{timeout_minutes:g}_minutes"
        except subprocess.CalledProcessError as exc:
            return False, f"training_exit_code_{exc.returncode}"


def promote_champion(trial_dir: Path, config_path: Path, champion_dir: Path, report: Mapping[str, Any], score_metric: str, score: float) -> None:
    required = [trial_dir / "predictor.pt", trial_dir / "training_report.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot promote champion; missing: " + ", ".join(missing))

    tmp_dir = champion_dir.with_name(champion_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trial_dir / "predictor.pt", tmp_dir / "predictor.pt")
    shutil.copy2(trial_dir / "training_report.json", tmp_dir / "training_report.json")
    shutil.copy2(config_path, tmp_dir / "config.yaml")
    metadata = {
        "promoted_utc": utc_now(),
        "source_trial_dir": str(trial_dir),
        "score_metric": score_metric,
        "score": score,
        "test_rmse": report.get("test_rmse"),
        "test_mae": report.get("test_mae"),
        "val_loss_final": report.get("val_loss_final"),
        "research_only": True,
        "not_for_treatment": True,
    }
    (tmp_dir / "champion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if champion_dir.exists():
        shutil.rmtree(champion_dir)
    tmp_dir.rename(champion_dir)


def cleanup_trial(trial_dir: Path, *, keep_success: bool, keep_failed: bool) -> None:
    if keep_success or keep_failed:
        return
    try:
        shutil.rmtree(trial_dir)
    except OSError:
        pass


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson-safe AutoML loop for IINTS glucose forecasting.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trials-dir", type=Path, default=DEFAULT_TRIALS_DIR)
    parser.add_argument("--champion-dir", type=Path, default=DEFAULT_CHAMPION_DIR)
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--max-trials", type=int, default=0, help="0 means run forever until Ctrl+C.")
    parser.add_argument("--timeout-minutes", type=float, default=45.0)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=5e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 96, 128, 160])
    parser.add_argument("--num-layers-choices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--min-dropout", type=float, default=0.05)
    parser.add_argument("--max-dropout", type=float, default=0.30)
    parser.add_argument("--min-pinn-lambda", type=float, default=0.10)
    parser.add_argument("--max-pinn-lambda", type=float, default=1.20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0005)
    parser.add_argument("--keep-success-trials", action="store_true", help="Keep full folders for successful non-champion trials.")
    parser.add_argument("--keep-failed-trials", action="store_true", help="Keep failed trial folders for debugging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preflight(args.dataset, args.trials_dir, args.champion_dir)
    rng = random.Random(args.seed)
    best_score = load_best_score(args.leaderboard)

    print("==============================================")
    print("   IINTS-AF Jetson AutoML Factory v2")
    print("   Research-only glucose forecast search")
    print("   Press Ctrl+C to stop safely")
    print("==============================================")
    print(f"Dataset: {args.dataset}")
    print(f"Champion dir: {args.champion_dir}")
    print(f"Current score to beat: {best_score if math.isfinite(best_score) else 'none yet'}")

    trial_index = 0
    try:
        while args.max_trials <= 0 or trial_index < args.max_trials:
            trial_index += 1
            trial_id = f"trial_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{trial_index:05d}"
            trial_dir = args.trials_dir / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            config = build_trial_config(args, rng, trial_dir, trial_index)
            config_path = trial_dir / "trial_config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            training = config["training"]

            print(f"\n[trial] {trial_id}")
            print(
                "  lr={learning_rate} hidden={hidden_size} layers={num_layers} "
                "dropout={dropout} pinn_lambda={pinn_lambda}".format(**training)
            )
            started = time.time()
            success, error = run_trial(args.dataset, config_path, trial_dir, args.timeout_minutes)
            duration = time.time() - started

            report: dict[str, Any] = {}
            score_metric = "unavailable"
            score = math.inf
            is_champion = False
            status = "success" if success else "failed"

            report_path = trial_dir / "training_report.json"
            if success and report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                score_metric, score = score_from_report(report)
                is_champion = score < best_score
                if is_champion:
                    promote_champion(trial_dir, config_path, args.champion_dir, report, score_metric, score)
                    best_score = score
                    print(f"  NEW CHAMPION: {score_metric}={score:.4f}")
                else:
                    print(f"  score: {score_metric}={score:.4f}; champion remains {best_score:.4f}")
            elif success:
                status = "failed"
                error = "missing_training_report_json"

            append_leaderboard_row(
                args.leaderboard,
                {
                    "timestamp_utc": utc_now(),
                    "trial_id": trial_id,
                    "status": status,
                    "score_metric": score_metric,
                    "score": score if math.isfinite(score) else "",
                    "test_rmse": report.get("test_rmse", ""),
                    "test_mae": report.get("test_mae", ""),
                    "val_loss_final": report.get("val_loss_final", ""),
                    "train_loss_final": report.get("train_loss_final", ""),
                    "learning_rate": training.get("learning_rate"),
                    "hidden_size": training.get("hidden_size"),
                    "num_layers": training.get("num_layers"),
                    "dropout": training.get("dropout"),
                    "pinn_lambda": training.get("pinn_lambda"),
                    "loss": training.get("loss"),
                    "epochs": training.get("epochs"),
                    "batch_size": training.get("batch_size"),
                    "duration_sec": round(duration, 2),
                    "is_champion": is_champion,
                    "trial_dir": str(trial_dir),
                    "error": error,
                },
            )

            cleanup_trial(trial_dir, keep_success=args.keep_success_trials or is_champion, keep_failed=args.keep_failed_trials and status == "failed")
            release_memory()
            print(f"  status={status}; cooldown={args.cooldown_seconds:g}s")
            time.sleep(max(0.0, float(args.cooldown_seconds)))
    except KeyboardInterrupt:
        print("\nStopped by user. Leaderboard and champion files are preserved.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
