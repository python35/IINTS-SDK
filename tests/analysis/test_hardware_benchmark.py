from __future__ import annotations

import builtins
import subprocess

import pytest

pytest.importorskip("psutil")

from iints.analysis.hardware_benchmark import HardwareBenchmark


def test_detect_jetson_does_not_swallow_keyboard_interrupt(monkeypatch) -> None:
    benchmark = HardwareBenchmark()

    def _raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(builtins, "open", _raise_interrupt)

    with pytest.raises(KeyboardInterrupt):
        benchmark._detect_jetson()


def test_get_tegrastats_metrics_handles_timeout(monkeypatch) -> None:
    benchmark = HardwareBenchmark()
    benchmark.is_jetson = True

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="tegrastats", timeout=2)

    monkeypatch.setattr(subprocess, "run", _timeout)

    assert benchmark._get_tegrastats_metrics() is None
