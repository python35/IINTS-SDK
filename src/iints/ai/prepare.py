from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import importlib
import json
import math

import pandas as pd

from iints.utils.run_io import compute_sha256


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    if hasattr(value, "item"):
        return _normalize_value(value.item())
    return value


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(value) for key, value in record.items()}


def _normalize_series_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    return {str(key): _normalize_value(value) for key, value in record.items()}


def _glucose_column(df: pd.DataFrame) -> str:
    for candidate in ("glucose_actual_mgdl", "glucose_to_algo_mgdl", "cgm"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Results CSV does not contain a supported glucose column.")


def _bool_sum(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def _safe_sum(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).sum())


def _time_in_band_pct(series: pd.Series, low: float, high: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    mask = (clean >= low) & (clean <= high)
    return float(mask.mean() * 100.0)


def _sample_trace(df: pd.DataFrame, *, max_rows: int = 48) -> list[dict[str, Any]]:
    interesting_columns = [
        "time_minutes",
        "glucose_actual_mgdl",
        "glucose_to_algo_mgdl",
        "glucose_trend_mgdl_min",
        "predicted_glucose_30min",
        "algo_recommended_insulin_units",
        "delivered_insulin_units",
        "safety_triggered",
        "safety_reason",
    ]
    present = [column for column in interesting_columns if column in df.columns]
    if not present:
        return []
    if len(df) <= max_rows:
        sampled = df[present]
    else:
        step = max(1, len(df) // max_rows)
        sampled = df.iloc[::step][present].head(max_rows)
    return [_normalize_series_record(record) for record in sampled.to_dict(orient="records")]


def _position_for_label(df: pd.DataFrame, label: Any) -> int:
    location = df.index.get_loc(label)
    if isinstance(location, int):
        return location
    raise ValueError(f"Could not resolve row position for label: {label!r}")


def _select_step_payload(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    if df.empty:
        raise ValueError("Results CSV is empty.")

    glucose_column = _glucose_column(df)
    risk_position = len(df.index) - 1
    selection_reason = "latest_step"

    if "safety_triggered" in df.columns and df["safety_triggered"].fillna(False).astype(bool).any():
        safety_rows = df[df["safety_triggered"].fillna(False).astype(bool)]
        risk_position = _position_for_label(df, safety_rows.index[0])
        selection_reason = "first_safety_trigger"
    else:
        glucose_values = pd.to_numeric(df[glucose_column], errors="coerce")
        if (glucose_values < 70).any():
            risk_label = glucose_values.idxmin()
            risk_position = _position_for_label(df, risk_label)
            selection_reason = "lowest_glucose"
        elif "predicted_glucose_30min" in df.columns:
            predicted = pd.to_numeric(df["predicted_glucose_30min"], errors="coerce")
            if (predicted > 180).any():
                risk_label = predicted.idxmax()
                risk_position = _position_for_label(df, risk_label)
                selection_reason = "highest_predicted_glucose_30min"
    latest_position = len(df.index) - 1

    def _row_at(position: int) -> Optional[dict[str, Any]]:
        if position < 0 or position >= len(df.index):
            return None
        return _normalize_series_record(df.iloc[position].to_dict())

    risk_payload = {
        "selection_reason": selection_reason,
        "selected_step": _row_at(risk_position),
        "previous_step": _row_at(risk_position - 1),
        "next_step": _row_at(risk_position + 1),
    }
    latest_payload = {
        "selection_reason": "latest_step",
        "selected_step": _row_at(latest_position),
        "previous_step": _row_at(latest_position - 1),
        "next_step": None,
    }
    return risk_payload, latest_payload


def _build_summary(df: pd.DataFrame, run_metadata: dict[str, Any], audit_summary: dict[str, Any]) -> dict[str, Any]:
    glucose_column = _glucose_column(df)
    glucose = pd.to_numeric(df[glucose_column], errors="coerce").dropna()
    if glucose.empty:
        raise ValueError("Results CSV glucose series is empty.")

    duration_minutes = run_metadata.get("config", {}).get("duration_minutes")
    if duration_minutes is None and "time_minutes" in df.columns:
        duration_minutes = _normalize_value(pd.to_numeric(df["time_minutes"], errors="coerce").max())

    return {
        "steps": int(len(df)),
        "duration_minutes": _normalize_value(duration_minutes),
        "mean_glucose_mgdl": _normalize_value(float(glucose.mean())),
        "min_glucose_mgdl": _normalize_value(float(glucose.min())),
        "max_glucose_mgdl": _normalize_value(float(glucose.max())),
        "time_in_range_70_180_pct": _normalize_value(_time_in_band_pct(glucose, 70.0, 180.0)),
        "time_below_70_pct": _normalize_value(float((glucose < 70.0).mean() * 100.0)),
        "time_above_180_pct": _normalize_value(float((glucose > 180.0).mean() * 100.0)),
        "delivered_insulin_total_units": _normalize_value(_safe_sum(df, "delivered_insulin_units")),
        "recommended_insulin_total_units": _normalize_value(_safe_sum(df, "algo_recommended_insulin_units")),
        "safety_trigger_count": _bool_sum(df, "safety_triggered"),
        "audit_override_count": int(audit_summary.get("total_overrides", 0)),
    }


def _build_payloads(
    *,
    run_dir: Path,
    results_df: pd.DataFrame,
    run_metadata: dict[str, Any],
    run_manifest: dict[str, Any],
    audit_summary: dict[str, Any],
    baseline_comparison: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    summary = _build_summary(results_df, run_metadata, audit_summary)
    risk_payload, latest_payload = _select_step_payload(results_df)
    trace_sample = _sample_trace(results_df)

    common = {
        "generated_at_utc": _now_utc(),
        "run_dir": str(run_dir),
        "run_id": run_metadata.get("run_id"),
        "sdk_version": run_metadata.get("sdk_version"),
        "algorithm": run_metadata.get("config", {}).get("algorithm", {}),
        "scenario": run_metadata.get("config", {}).get("scenario"),
        "summary": summary,
    }

    payloads: dict[str, dict[str, Any]] = {
        "report_payload.json": {
            **common,
            "artifacts": {
                "run_metadata": str(run_dir / "run_metadata.json"),
                "run_manifest": str(run_dir / "run_manifest.json"),
                "results_csv": str(run_dir / "results.csv"),
                "audit_summary": str(run_dir / "audit" / "audit_summary.json"),
                "baseline_comparison": str(run_dir / "baseline" / "baseline_comparison.json"),
            },
            "audit_summary": audit_summary,
            "baseline_comparison": baseline_comparison,
            "trace_sample": trace_sample,
            "run_manifest": run_manifest,
        },
        "anomalies_payload.json": {
            **common,
            "audit_summary": audit_summary,
            "safety_events": [
                record
                for record in trace_sample
                if bool(record.get("safety_triggered"))
            ],
        },
        "trends_payload.json": {
            **common,
            "trace_sample": trace_sample,
            "baseline_comparison": baseline_comparison,
        },
        "step_riskiest.json": {
            **common,
            **risk_payload,
        },
        "step_latest.json": {
            **common,
            **latest_payload,
        },
    }
    return payloads


def _load_mdmp_signer_tools() -> tuple[type[Any], Any]:
    try:
        module = importlib.import_module("mdmp_core")
    except Exception as exc:
        raise ImportError(
            "Local AI certification requires the bundled MDMP crypto support.\n"
            "Install with: pip install 'iints-sdk-python35[mdmp]'"
        ) from exc

    signer_cls = getattr(module, "MDMPSigner", None)
    keygen_fn = getattr(module, "generate_keypair", None)
    if signer_cls is None or keygen_fn is None:
        raise ImportError("mdmp_core is installed but does not expose MDMPSigner/generate_keypair.")
    return signer_cls, keygen_fn


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_ai_ready_artifacts(
    run_dir: str | Path,
    *,
    create_dev_mdmp_cert: bool = True,
    grade: str = "research_grade",
    expires_days: int = 30,
    key_dir: str | Path | None = None,
) -> dict[str, str]:
    bundle_dir = Path(run_dir).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {bundle_dir}")

    results_csv = bundle_dir / "results.csv"
    run_metadata_path = bundle_dir / "run_metadata.json"
    run_manifest_path = bundle_dir / "run_manifest.json"

    for required in (results_csv, run_metadata_path, run_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required run artifact missing: {required}")

    results_df = pd.read_csv(results_csv)
    run_metadata = _read_json(run_metadata_path)
    run_manifest = _read_json(run_manifest_path)

    audit_summary_path = bundle_dir / "audit" / "audit_summary.json"
    baseline_path = bundle_dir / "baseline" / "baseline_comparison.json"
    audit_summary = _read_json(audit_summary_path) if audit_summary_path.is_file() else {}
    baseline_comparison = _read_json(baseline_path) if baseline_path.is_file() else None

    ai_dir = bundle_dir / "ai"
    payloads = _build_payloads(
        run_dir=bundle_dir,
        results_df=results_df,
        run_metadata=run_metadata,
        run_manifest=run_manifest,
        audit_summary=audit_summary,
        baseline_comparison=baseline_comparison,
    )

    written: dict[str, str] = {}
    for filename, payload in payloads.items():
        target = ai_dir / filename
        _write_json(target, payload)
        written[filename.removesuffix(".json")] = str(target)

    if create_dev_mdmp_cert:
        signer_cls, keygen_fn = _load_mdmp_signer_tools()
        resolved_key_dir = Path(key_dir).expanduser().resolve() if key_dir else ai_dir / "keys"
        private_key_path = resolved_key_dir / "mdmp_private_v1.pem"
        public_key_path = resolved_key_dir / "mdmp_pub_v1.pem"
        if not private_key_path.is_file() or not public_key_path.is_file():
            keygen_fn(output_dir=resolved_key_dir)

        cert_payload = {
            "mdmp_object": "iints_ai_local_cert",
            "spec_version": "1.0",
            "grade": grade,
            "generated_at_utc": _now_utc(),
            "run_id": run_metadata.get("run_id"),
            "run_dir": str(bundle_dir),
            "sdk_version": run_metadata.get("sdk_version"),
            "purpose": "local_research_ai",
            "results_csv_sha256": f"sha256:{compute_sha256(results_csv)}",
            "run_manifest_sha256": f"sha256:{compute_sha256(run_manifest_path)}",
            "notes": "Local development certificate generated by IINTS AI prepare.",
        }
        signer = signer_cls(
            private_key_path=private_key_path,
            signed_by="IINTS-Local-AI",
            key_id="iints_local_ai_v1",
        )
        signed_cert = signer.sign_card(cert_payload, expires_days=expires_days)
        cert_path = ai_dir / "report.signed.mdmp"
        _write_json(cert_path, signed_cert)
        written["mdmp_cert"] = str(cert_path)
        written["mdmp_public_key"] = str(public_key_path)
        written["mdmp_private_key"] = str(private_key_path)

    return written
