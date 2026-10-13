from __future__ import annotations

import json
import platform
import socket
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from iints.live_patient.api import create_patient_app
from iints.live_patient.daemon import _start_api_server
from iints.live_patient.runtime import LivePatientDaemon, PatientRuntimeConfig


def _platform_label(requested: str) -> str:
    if requested != "auto":
        return requested
    machine = platform.machine().lower()
    system = platform.system()
    try:
        model_path = Path("/proc/device-tree/model")
        if model_path.is_file():
            return model_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        pass
    return f"{system} ({machine})"


def _max_rss_mb() -> float:
    """Peak resident set size in MiB, on every supported platform.

    ``resource`` is Unix-only, so importing it unconditionally made this
    benchmark unavailable on Windows. The Windows branch reads
    PeakWorkingSetSize through psapi; both report the peak, not the current
    footprint.
    """
    try:
        import resource
    except ImportError:
        pass
    else:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return float(usage) / (1024 * 1024)
        return float(usage) / 1024.0

    try:
        import psutil  # optional; present in some research environments
        return float(psutil.Process().memory_info().peak_wset) / (1024 * 1024)
    except (ImportError, AttributeError, OSError):
        pass

    return _windows_peak_rss_mb()


def _windows_peak_rss_mb() -> float:
    """Query Windows with pointer-sized handles, without requiring psutil."""

    try:
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        handle = get_current_process()
        # psapi on older Windows, kernel32's K32 alias on newer builds.
        for library_name, function_name in (
            ("kernel32", "K32GetProcessMemoryInfo"),
            ("psapi", "GetProcessMemoryInfo"),
        ):
            try:
                library = ctypes.WinDLL(library_name, use_last_error=True)
                probe = getattr(library, function_name)
                probe.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
                probe.restype = wintypes.BOOL
                if probe(handle, ctypes.byref(counters), counters.cb):
                    break
            except (OSError, AttributeError):
                continue
        else:
            raise OSError("GetProcessMemoryInfo unavailable")
        return float(counters.PeakWorkingSetSize) / (1024 * 1024)
    except Exception:
        # Reporting a wrong number would be worse than reporting none.
        return float("nan")


def _measure_http_ms(url: str, requests: int = 5) -> dict[str, float]:
    timings: list[float] = []
    for _ in range(max(1, requests)):
        started = time.perf_counter()
        with urllib.request.urlopen(url, timeout=5.0) as response:  # nosec B310 - local loopback probe
            response.read()
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": sum(timings) / len(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def _measure_asgi_ms(app: Any, path: str, requests: int = 5) -> dict[str, float]:
    from fastapi.testclient import TestClient

    timings: list[float] = []
    with TestClient(app) as client:
        for _ in range(max(1, requests)):
            started = time.perf_counter()
            response = client.get(path)
            response.raise_for_status()
            _ = response.content
            timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_ms": sum(timings) / len(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def _can_bind(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def run_edge_benchmark(
    *,
    algo_path: Path,
    patient_config: str = "default_patient",
    patient_model_type: str = "auto",
    scenario_profile: str = "normal_day",
    steps: int = 72,
    platform_name: str = "auto",
    api_host: str = "127.0.0.1",
    api_port: int = 8766,
    seed: int | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="iints-edge-benchmark-") as tmpdir:
        workspace = Path(tmpdir) / "patient_runtime"
        config = PatientRuntimeConfig(
            workspace=str(workspace),
            algo_path=str(algo_path.expanduser().resolve()),
            patient_config=patient_config,
            patient_model_type=patient_model_type,
            scenario_profile=scenario_profile,
            mode="demo-time",
            speed=9999.0,
            api_host=api_host,
            api_port=api_port,
            seed=seed,
        )

        daemon = LivePatientDaemon(config)
        daemon.install_signal_handlers()
        daemon.bootstrap(reset=True)

        started = time.perf_counter()
        for _ in range(max(1, steps)):
            daemon.advance_once()
        elapsed = time.perf_counter() - started
        steps_per_second = max(1, steps) / max(elapsed, 1e-9)

        app = create_patient_app(workspace)
        if _can_bind(config.api_host, config.api_port):
            server, thread = _start_api_server(config)
            try:
                time.sleep(0.5)
                dashboard_metrics = _measure_http_ms(f"{config.dashboard_url}", requests=5)
                status_metrics = _measure_http_ms(f"{config.api_url}/status", requests=5)
                probe_mode = "loopback-http"
                latest_status = daemon.store.read_status()
            finally:
                server.should_exit = True
                thread.join(timeout=2.0)
        else:
            dashboard_metrics = _measure_asgi_ms(app, "/dashboard", requests=5)
            status_metrics = _measure_asgi_ms(app, "/status", requests=5)
            probe_mode = "in-process-asgi"
            latest_status = daemon.store.read_status()

        daemon.stop_requested = True
        daemon.shutdown()

        results = {
            "platform": _platform_label(platform_name),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "steps": int(steps),
            "scenario_profile": scenario_profile,
            "seed": config.seed,
            "time_step_minutes": config.time_step_minutes,
            "runtime": {
                "elapsed_seconds": elapsed,
                "steps_per_second": steps_per_second,
                "mean_step_latency_ms": (elapsed / max(1, steps)) * 1000.0,
                "peak_process_memory_mb": _max_rss_mb(),
            },
            "dashboard": {
                "probe_mode": probe_mode,
                "dashboard_response_ms": dashboard_metrics,
                "status_response_ms": status_metrics,
            },
            "latest_status": latest_status,
        }
        return json.loads(json.dumps(results))
