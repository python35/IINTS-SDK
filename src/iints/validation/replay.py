from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import iints
from iints.api.base_algorithm import InsulinAlgorithm
from iints.core.safety import SafetyConfig


NON_DETERMINISTIC_COLUMNS = {
    "algorithm_latency_ms",
    "supervisor_latency_ms",
    "step_latency_ms",
}

NON_DETERMINISTIC_REPORT_KEYS = {
    "performance_report",
}


@dataclass(frozen=True)
class ReplayRunDigest:
    results_hash: str
    safety_hash: str
    rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results_hash": self.results_hash,
            "safety_hash": self.safety_hash,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class ReplayCheckResult:
    passed: bool
    seed: int
    digests: List[ReplayRunDigest]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "seed": self.seed,
            "digests": [digest.to_dict() for digest in self.digests],
        }


def _stable_hash_dataframe(
    df: pd.DataFrame,
    *,
    ignore_columns: Optional[Iterable[str]] = None,
    float_decimals: int = 8,
) -> str:
    ignore = set(ignore_columns or ())
    keep_columns = [col for col in df.columns if col not in ignore]
    stable_df = df[keep_columns].copy()
    stable_df = stable_df.sort_values(by=["time_minutes"] if "time_minutes" in stable_df.columns else keep_columns).reset_index(drop=True)

    # Stabilize float formatting before hashing.
    for col in stable_df.select_dtypes(include=[np.number]).columns:
        stable_df[col] = np.round(stable_df[col].astype(float), decimals=float_decimals)

    payload = stable_df.to_json(orient="records", date_format="iso", double_precision=float_decimals)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_hash_report(
    safety_report: Dict[str, Any],
    *,
    ignore_keys: Optional[Iterable[str]] = None,
) -> str:
    ignore = set(ignore_keys or ())
    payload = {key: value for key, value in safety_report.items() if key not in ignore}
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_deterministic_replay_check(
    *,
    algorithm: InsulinAlgorithm,
    scenario: Optional[Dict[str, Any]],
    patient_config: Any,
    duration_minutes: int,
    time_step: int,
    seed: int,
    repeats: int = 2,
    safety_config: Optional[SafetyConfig] = None,
    predictor: Optional[object] = None,
) -> ReplayCheckResult:
    digests: List[ReplayRunDigest] = []
    for run_idx in range(repeats):
        with tempfile.TemporaryDirectory(prefix=f"iints_replay_{run_idx}_") as tmp_dir:
            outputs = iints.run_simulation(
                algorithm=algorithm,
                scenario=scenario,
                patient_config=patient_config,
                duration_minutes=duration_minutes,
                time_step=time_step,
                seed=seed,
                output_dir=Path(tmp_dir),
                compare_baselines=False,
                export_audit=False,
                generate_report=False,
                safety_config=safety_config,
                predictor=predictor,
            )
            results_df = outputs["results"]
            safety_report = outputs.get("safety_report", {})
            digests.append(
                ReplayRunDigest(
                    results_hash=_stable_hash_dataframe(
                        results_df,
                        ignore_columns=NON_DETERMINISTIC_COLUMNS,
                    ),
                    safety_hash=_stable_hash_report(
                        safety_report,
                        ignore_keys=NON_DETERMINISTIC_REPORT_KEYS,
                    ),
                    rows=int(len(results_df)),
                )
            )

    passed = True
    if digests:
        baseline = digests[0]
        for digest in digests[1:]:
            if (
                digest.results_hash != baseline.results_hash
                or digest.safety_hash != baseline.safety_hash
                or digest.rows != baseline.rows
            ):
                passed = False
                break

    return ReplayCheckResult(passed=passed, seed=seed, digests=digests)
