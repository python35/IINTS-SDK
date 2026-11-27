"""Module D — Experiment Reproducibility, Provenance & Artifact Storage.

Provides:
- Deterministic SHA-256 seed derivation (``derive_seed``) so adding algorithms or
  sensor arms never alters patient physiology or unrelated random streams,
- Full software/git/config provenance manifests (``ProvenanceManifest``),
- Clarke A+B & dropout/recovery metric calculators,
- Parquet (with transparent CSV fallback) + SQLite metadata registry (``ExperimentStore``),
- ``run_single_experiment`` to execute any declarative ``ExperimentSpec`` end-to-end.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
import pydantic
import scipy
import yaml

from iints._version import __version__ as SDK_VERSION
from iints.analysis.error_grid import clarke_zones
from iints.sdk2.contracts import (
    RESEARCH_ONLY_DISCLAIMER,
    ExperimentSpec,
    load_experiment_spec,
)
from iints.sdk2.forecasters import evaluate_forecasters_on_trace
from iints.sdk2.sensors import apply_sensor_fault_model, simulate_patient_trajectory

__all__ = [
    "derive_seed",
    "sha256_file",
    "sha256_json",
    "capture_provenance",
    "summarize_forecast_metrics",
    "write_dataframe_artifact",
    "read_dataframe_artifact",
    "ExperimentStore",
    "run_single_experiment",
]


def derive_seed(master_seed: int, *tokens: Union[str, int]) -> int:
    """Derive an independent 31-bit deterministic seed from ``(master_seed, *tokens)``."""
    payload = "::".join([str(int(master_seed)), *[str(t) for t in tokens]]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> Dict[str, Any]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=3)
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=3)
            .decode("utf-8")
            .strip()
        )
        return {"commit": commit, "dirty": bool(status)}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def capture_provenance(config_payload: Dict[str, Any], artifact_paths: Optional[Sequence[Path]] = None) -> Dict[str, Any]:
    """Capture software, dependency, git, and cryptographic artifact hashes."""
    artifacts = {}
    for p in artifact_paths or ():
        if p.exists() and p.is_file():
            artifacts[p.name] = sha256_file(p)
    return {
        "disclaimer": RESEARCH_ONLY_DISCLAIMER,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sdk_version": SDK_VERSION,
        "git": _git_info(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "pydantic": pydantic.__version__,
            "pyyaml": getattr(yaml, "__version__", "unknown"),
        },
        "config_sha256": sha256_json(config_payload),
        "artifact_sha256": artifacts,
    }


def write_dataframe_artifact(df: pd.DataFrame, stem_path: Path) -> Path:
    """Write Parquet when pyarrow/fastparquet is installed, else deterministic CSV."""
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = stem_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = stem_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def read_dataframe_artifact(stem_or_path: Path) -> pd.DataFrame:
    if stem_or_path.exists():
        if stem_or_path.suffix == ".parquet":
            return pd.read_parquet(stem_or_path)
        return pd.read_csv(stem_or_path)
    p_parquet = stem_or_path.with_suffix(".parquet")
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    p_csv = stem_or_path.with_suffix(".csv")
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"No Parquet or CSV artifact found for {stem_or_path}")


def summarize_forecast_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy, Clarke A+B, dropout MAE, and post-dropout recovery MAE."""
    if forecasts.empty:
        return pd.DataFrame()

    rows = []
    for (algo_id, horizon), grp in forecasts.groupby(["algorithm_id", "horizon_min"]):
        y_true = grp["y_true"].to_numpy(dtype=float)
        y_pred = grp["y_pred"].to_numpy(dtype=float)
        abs_err = grp["abs_error"].to_numpy(dtype=float)
        sq_err = grp["sq_error"].to_numpy(dtype=float)

        zones = clarke_zones(y_true, y_pred)
        ab_pct = 100.0 * float(np.mean(np.isin(zones, ["A", "B"])))

        drop_grp = grp[grp["origin_in_dropout"]]
        rec_grp = grp[grp["origin_in_recovery"]]

        # Hypo-region false reassurance: predicted >= 85 mg/dL while true < 70 mg/dL
        hypo_grp = grp[grp["y_true"] < 70.0]
        missed_hypo_pct = (
            100.0 * float(np.mean(hypo_grp["y_pred"].to_numpy(dtype=float) >= 85.0))
            if len(hypo_grp) > 0
            else float("nan")
        )

        rows.append(
            {
                "algorithm_id": str(algo_id),
                "horizon_min": int(cast(Any, horizon)),
                "n_origins": int(len(grp)),
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(sq_err))),
                "mard_pct": 100.0 * float(np.mean(abs_err / np.maximum(y_true, 20.0))),
                "bias": float(np.mean(grp["signed_error"].to_numpy(dtype=float))),
                "p95_abs_error": float(np.quantile(abs_err, 0.95)),
                "clarke_ab_pct": ab_pct,
                "coverage_90_pct": 100.0 * float(grp["inside_90"].mean()),
                "dropout_mae": float(drop_grp["abs_error"].mean()) if len(drop_grp) > 0 else float("nan"),
                "dropout_recovery_mae": float(rec_grp["abs_error"].mean()) if len(rec_grp) > 0 else float("nan"),
                "missed_hypo_pct": missed_hypo_pct,
            }
        )
    return pd.DataFrame(rows)


class ExperimentStore:
    """SQLite metadata index backing reproducible SDK 2.0 & Research Lab runs."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL,
                    sdk_version TEXT NOT NULL,
                    git_commit TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    master_seed INTEGER NOT NULL,
                    artifact_dir TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS algorithm_metrics (
                    experiment_id TEXT NOT NULL,
                    condition_id TEXT NOT NULL,
                    algorithm_id TEXT NOT NULL,
                    horizon_min INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    mae REAL,
                    rmse REAL,
                    dropout_mae REAL,
                    dropout_recovery_mae REAL,
                    clarke_ab_pct REAL,
                    coverage_90_pct REAL
                )
                """
            )

    def record_experiment(
        self,
        experiment_id: str,
        provenance: Dict[str, Any],
        master_seed: int,
        artifact_dir: Path,
        summary_payload: Dict[str, Any],
        metrics_df: Optional[pd.DataFrame] = None,
        condition_id: str = "single",
        split: str = "dev",
    ) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (experiment_id, created_utc, sdk_version, git_commit, config_sha256, master_seed, artifact_dir, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    provenance["created_utc"],
                    provenance["sdk_version"],
                    provenance["git"]["commit"],
                    provenance["config_sha256"],
                    int(master_seed),
                    str(artifact_dir.resolve()),
                    json.dumps(summary_payload, default=str),
                ),
            )
            if metrics_df is not None and not metrics_df.empty:
                conn.execute(
                    "DELETE FROM algorithm_metrics WHERE experiment_id = ? AND condition_id = ? AND split = ?",
                    (experiment_id, condition_id, split),
                )
                for _, row in metrics_df.iterrows():
                    conn.execute(
                        """
                        INSERT INTO algorithm_metrics
                        (experiment_id, condition_id, algorithm_id, horizon_min, split,
                         mae, rmse, dropout_mae, dropout_recovery_mae, clarke_ab_pct, coverage_90_pct)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            experiment_id,
                            condition_id,
                            str(row["algorithm_id"]),
                            int(row["horizon_min"]),
                            split,
                            float(row["mae"]),
                            float(row["rmse"]),
                            None if np.isnan(row["dropout_mae"]) else float(row["dropout_mae"]),
                            None if np.isnan(row["dropout_recovery_mae"]) else float(row["dropout_recovery_mae"]),
                            float(row["clarke_ab_pct"]),
                            float(row["coverage_90_pct"]),
                        ),
                    )

    def list_experiments(self) -> list[Dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM experiments ORDER BY created_utc DESC").fetchall()
            return [dict(r) for r in rows]


def run_single_experiment(
    spec_or_path: Union[ExperimentSpec, str, Path, Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute one declarative ``ExperimentSpec`` and persist full provenance & artifacts."""
    spec = spec_or_path if isinstance(spec_or_path, ExperimentSpec) else load_experiment_spec(spec_or_path)
    pat_seed = derive_seed(spec.seed, "patient", spec.patient.profile, spec.scenario.id)
    sen_seed = derive_seed(spec.seed, "sensor", spec.patient.profile, spec.scenario.id, spec.sensor.condition_id)

    traj = simulate_patient_trajectory(spec.patient, spec.scenario, seed=pat_seed)
    trace = apply_sensor_fault_model(traj, spec.sensor, seed=sen_seed)
    forecasts = evaluate_forecasters_on_trace(
        trace,
        spec.resolved_algorithms(),
        spec.evaluation,
        step_minutes=spec.scenario.step_minutes,
    )
    summary_df = summarize_forecast_metrics(forecasts)

    artifact_paths = []
    out_root = Path(output_dir) if output_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
        cfg_path = out_root / "experiment_config.yaml"
        from iints.sdk2.contracts import dump_experiment_yaml

        cfg_path.write_text(dump_experiment_yaml(spec), encoding="utf-8")
        trace_path = write_dataframe_artifact(trace, out_root / "timeseries_trace")
        fc_path = write_dataframe_artifact(forecasts, out_root / "forecasts")
        sum_path = out_root / "metrics_summary.csv"
        summary_df.to_csv(sum_path, index=False)
        artifact_paths = [cfg_path, trace_path, fc_path, sum_path]

    config_dict = spec.model_dump(mode="json")
    provenance = capture_provenance(config_dict, artifact_paths)
    if out_root is not None:
        (out_root / "provenance_manifest.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        store = ExperimentStore(out_root.parent / "experiments.sqlite")
        store.record_experiment(
            spec.id,
            provenance,
            spec.seed,
            out_root,
            {"metrics": summary_df.to_dict(orient="records")},
            metrics_df=summary_df,
            condition_id=spec.sensor.condition_id,
            split=spec.patient.split,
        )

    return {
        "experiment_id": spec.id,
        "spec": spec,
        "trace": trace,
        "forecasts": forecasts,
        "metrics": summary_df,
        "provenance": provenance,
    }
