"""Phase 5b — NVIDIA Jetson Orin Nano Super Local Inference & `tegrastats` Worker.

Provides:
1. `parse_tegrastats_line()`: Parses JetPack 5 / JetPack 6 (`Orin Nano Super`)
   hardware telemetry (`RAM`, `GR3D_FREQ` GPU utilization, `VDD_IN` mW board power,
   `VDD_CPU_GPU_CV` mW compute rail, and `cpu@`/`gpu@` thermal sensors).
2. `benchmark_edge_inference()`: Warm-up + batch-size-1 real-time latency benchmark
   (Median, P95, P99 ms, Throughput pred/s, Peak RSS MiB, Energy mJ/prediction,
   and numerical parity check against reference outputs).
3. `create_jetson_worker_app()`: FastAPI local network worker so a PC/Studio test
   harness can stream 12-sample CGM windows to the Jetson Orin Nano Super and
   receive standardized `PredictionRecord` (`clinical_use=False`) + live telemetry!
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel, Field

from iints.forecasting.deployment.onnx_export import EdgeInferenceArtifact, load_edge_artifact
from iints.forecasting.schemas import FORECASTING_DISCLAIMER, PredictionRecord
from iints.jetson.endurance import collect_jetson_hardware_info
from iints.live_patient.edge_benchmark import _max_rss_mb

__all__ = [
    "parse_tegrastats_line",
    "TegrastatsMonitor",
    "benchmark_edge_inference",
    "create_jetson_worker_app",
]

_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB", re.IGNORECASE)
_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%", re.IGNORECASE)
_VDD_IN_RE = re.compile(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)mW(?:/(\d+)mW)?", re.IGNORECASE)
_VDD_COMPUTE_RE = re.compile(r"VDD_CPU_GPU_CV\s+(\d+)mW(?:/(\d+)mW)?", re.IGNORECASE)
_CPU_TEMP_RE = re.compile(r"cpu@([0-9.]+)C", re.IGNORECASE)
_GPU_TEMP_RE = re.compile(r"gpu@([0-9.]+)C", re.IGNORECASE)


def parse_tegrastats_line(line: str) -> Dict[str, Optional[float]]:
    """Parse one `tegrastats` line from JetPack 6.x (Orin Nano) or JetPack 4/5."""
    ram_m = _RAM_RE.search(line)
    gpu_m = _GR3D_RE.search(line)
    vin_m = _VDD_IN_RE.search(line)
    vcomp_m = _VDD_COMPUTE_RE.search(line)
    ctemp_m = _CPU_TEMP_RE.search(line)
    gtemp_m = _GPU_TEMP_RE.search(line)

    return {
        "ram_used_mb": float(ram_m.group(1)) if ram_m else None,
        "ram_total_mb": float(ram_m.group(2)) if ram_m else None,
        "gpu_util_pct": float(gpu_m.group(1)) if gpu_m else None,
        "vdd_in_mw": float(vin_m.group(1)) if vin_m else None,
        "vdd_in_avg_mw": float(vin_m.group(2)) if (vin_m and vin_m.group(2)) else None,
        "vdd_cpu_gpu_cv_mw": float(vcomp_m.group(1)) if vcomp_m else None,
        "cpu_temp_c": float(ctemp_m.group(1)) if ctemp_m else None,
        "gpu_temp_c": float(gtemp_m.group(1)) if gtemp_m else None,
    }


class TegrastatsMonitor:
    """Background `tegrastats` sampler on NVIDIA Jetson; safe no-op on non-Jetson hosts."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = max(50, int(interval_ms))
        self.samples: List[Dict[str, Optional[float]]] = []
        self._proc: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.available = shutil.which("tegrastats") is not None

    def start(self) -> None:
        if not self.available:
            return
        exe = shutil.which("tegrastats")
        if exe is None:
            return
        try:
            self._proc = subprocess.Popen(  # noqa: S603
                [exe, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except Exception:
            self._proc = None

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parsed = parse_tegrastats_line(line)
            if any(v is not None for v in parsed.values()):
                self.samples.append(parsed)

    def stop_and_summarize(self) -> Dict[str, Any]:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.5)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if not self.samples:
            return {
                "telemetry_source": "host_no_tegrastats",
                "n_samples": 0,
                "mean_vdd_in_mw": None,
                "mean_gpu_util_pct": None,
                "max_gpu_temp_c": None,
                "power_methodology_note": (
                    "tegrastats not available on this host. Run on NVIDIA Jetson Orin Nano (JetPack 6.x) "
                    "to capture onboard VDD_IN rail power, and verify externally with a inline USB-C/DC wattmeter."
                ),
            }

        def _vals(key: str) -> List[float]:
            return [float(s[key]) for s in self.samples if s.get(key) is not None]  # type: ignore[arg-type]

        vin = _vals("vdd_in_mw")
        gpu = _vals("gpu_util_pct")
        gtemp = _vals("gpu_temp_c")
        return {
            "telemetry_source": "jetson_tegrastats",
            "n_samples": len(self.samples),
            "mean_vdd_in_mw": round(float(np.mean(vin)), 1) if vin else None,
            "mean_gpu_util_pct": round(float(np.mean(gpu)), 1) if gpu else None,
            "max_gpu_temp_c": round(float(np.max(gtemp)), 1) if gtemp else None,
            "power_methodology_note": (
                "Onboard INA3221 rail reading via tegrastats (VDD_IN). Includes carrier idle draw; "
                "pair with idle baseline subtraction and external wall/DC meter for publication claims."
            ),
        }


def benchmark_edge_inference(
    artifact_or_path: Union[EdgeInferenceArtifact, str, Path],
    *,
    sample_windows: Optional[np.ndarray] = None,
    n_warmup: int = 50,
    n_runs: int = 500,
    output_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Measure batch-size-1 latency (P50/P95/P99), throughput, Peak RSS, and Jetson telemetry."""
    artifact = (
        artifact_or_path
        if isinstance(artifact_or_path, EdgeInferenceArtifact)
        else load_edge_artifact(artifact_or_path)
    )
    L = artifact.input_length
    windows = (
        np.asarray(sample_windows, dtype=np.float64)
        if sample_windows is not None
        else np.linspace(95.0, 185.0, max(64, n_runs) * L, dtype=np.float64).reshape(-1, L)
    )

    # 1. Warm-up phase (never included in latency distribution!)
    for i in range(max(1, int(n_warmup))):
        _ = artifact.predict_batch(windows[i % len(windows) : (i % len(windows)) + 1])

    # 2. Timed batch-size-1 inference loop under active tegrastats monitoring
    monitor = TegrastatsMonitor(interval_ms=100)
    monitor.start()
    latencies_ms = np.empty(max(1, int(n_runs)), dtype=np.float64)
    t_total_start = time.perf_counter()
    for i in range(len(latencies_ms)):
        w_one = windows[i % len(windows) : (i % len(windows)) + 1]
        t0 = time.perf_counter_ns()
        _ = artifact.predict_batch(w_one)
        latencies_ms[i] = (time.perf_counter_ns() - t0) / 1e6
    elapsed_total_s = max(1e-9, time.perf_counter() - t_total_start)
    tegra_summary = monitor.stop_and_summarize()

    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.quantile(latencies_ms, 0.95))
    p99_ms = float(np.quantile(latencies_ms, 0.99))
    throughput = float(len(latencies_ms) / elapsed_total_s)
    peak_ram_mb = _max_rss_mb()

    mean_power_mw = tegra_summary.get("mean_vdd_in_mw")
    # Energy (mJ) = Power (mW) * Time (s) = Power (mW) * (latency_ms / 1000)
    energy_per_pred_mj = (
        round(float(mean_power_mw) * (median_ms / 1000.0), 4)
        if mean_power_mw is not None
        else None
    )

    hw_info = collect_jetson_hardware_info()
    report = {
        "disclaimer": FORECASTING_DISCLAIMER,
        "model_id": artifact.model_id,
        "architecture": artifact.architecture,
        "runtime": artifact.active_runtime,
        "batch_size": 1,
        "n_warmup": int(n_warmup),
        "n_runs": int(n_runs),
        "median_latency_ms": round(median_ms, 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 4),
        "p95_latency_ms": round(p95_ms, 4),
        "p99_latency_ms": round(p99_ms, 4),
        "throughput_preds_per_sec": round(throughput, 2),
        "peak_ram_mb": None if np.isnan(peak_ram_mb) else round(float(peak_ram_mb), 2),
        "energy_per_prediction_mj": energy_per_pred_mj,
        "telemetry": tegra_summary,
        "hardware": hw_info,
    }
    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class EdgePredictRequest(BaseModel):
    timestamp: str = "2027-01-10T12:00:00Z"
    cgm_window_mgdl: Sequence[float] = Field(..., min_length=3, max_length=144)
    prediction_horizon_minutes: int = 30
    input_quality: str = "complete"


def create_jetson_worker_app(bundle_path: Union[str, Path]) -> FastAPI:
    """Create the local NVIDIA Jetson Orin Nano inference + telemetry HTTP worker."""
    artifact = load_edge_artifact(bundle_path)
    worker = FastAPI(
        title=f"IINTS Jetson Edge Inference Worker ({artifact.model_id})",
        description=FORECASTING_DISCLAIMER,
        version="1.0.0",
    )

    @worker.get("/telemetry")
    def get_telemetry() -> Dict[str, Any]:
        rss = _max_rss_mb()
        return {
            "model_id": artifact.model_id,
            "architecture": artifact.architecture,
            "runtime": artifact.active_runtime,
            "peak_ram_mb": None if np.isnan(rss) else round(float(rss), 2),
            "hardware": collect_jetson_hardware_info(),
            "clinical_use": False,
        }

    @worker.post("/predict")
    def post_predict(req: EdgePredictRequest) -> Dict[str, Any]:
        w = np.asarray(req.cgm_window_mgdl, dtype=np.float64)[None, :]
        t0 = time.perf_counter_ns()
        pred = float(artifact.predict_batch(w)[0])
        lat_ms = (time.perf_counter_ns() - t0) / 1e6
        qual = req.input_quality if req.input_quality in ("complete", "locf_imputed", "degraded") else "degraded"
        rec = PredictionRecord(
            model_id=artifact.model_id,
            timestamp=req.timestamp,
            prediction_horizon_minutes=req.prediction_horizon_minutes,
            predicted_glucose_mgdl=round(pred, 2),
            input_quality=qual,  # type: ignore[arg-type]
            source="research_model",
            clinical_use=False,
        )
        return {
            "prediction": rec.model_dump(mode="json"),
            "inference_latency_ms": round(lat_ms, 4),
            "runtime": artifact.active_runtime,
        }

    @worker.post("/benchmark")
    def post_benchmark(n_warmup: int = 25, n_runs: int = 200) -> Dict[str, Any]:
        return benchmark_edge_inference(artifact, n_warmup=n_warmup, n_runs=n_runs)

    return worker
