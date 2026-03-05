#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iints-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "iints-cache"))

import torch

from iints.research.predictor import LSTMPredictor


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="iints_onnx_smoke_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        model_path = tmp_root / "predictor.pt"
        onnx_path = tmp_root / "predictor.onnx"
        report_path = tmp_root / "parity_report.json"

        model = LSTMPredictor(
            input_size=3,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            horizon_steps=2,
        )
        payload = {
            "state_dict": model.state_dict(),
            "config": {
                "input_size": 3,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "horizon_steps": 2,
                "feature_columns": ["glucose_actual_mgdl", "patient_iob_units", "glucose_trend_mgdl_min"],
                "history_steps": 12,
            },
        }
        torch.save(payload, model_path)

        export_cmd = [
            sys.executable,
            "research/export_predictor.py",
            "--model",
            str(model_path),
            "--out",
            str(onnx_path),
        ]
        subprocess.run(export_cmd, check=True)

        parity_cmd = [
            sys.executable,
            "-c",
            "from iints.cli.cli import app; app()",
            "research",
            "parity-check",
            "--model",
            str(model_path),
            "--onnx",
            str(onnx_path),
            "--samples",
            "16",
            "--tolerance",
            "0.01",
            "--output-json",
            str(report_path),
        ]
        subprocess.run(parity_cmd, check=True)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("passed", False)):
            print("ONNX parity smoke check failed.")
            return 1

    print("ONNX parity smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
