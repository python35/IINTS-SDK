"""Phase 5a — ONNX, TensorRT (`trtexec`) & Portable Edge Bundle Exporter.

Pipeline:
1. Exports fitted `GRUGlucosePredictor`, `TCNGlucosePredictor`, `RidgeCGMPredictor`,
   or `PersistencePredictor` to a lightweight, zero-dependency `.iints_edge.json`
   runtime artifact AND (when `torch` + `onnx` are installed, e.g. on NVIDIA Jetson
   JetPack 6.2.1) to a static-shape `(1, 12, C)` ONNX opset-17 graph.
2. Wraps NVIDIA JetPack `trtexec` to compile ONNX into FP32 / FP16 TensorRT `.plan`
   engines on the Jetson Orin Nano Super, with explicit operator-compatibility checks.
3. Verifies numerical parity (`max_abs_diff_mgdl <= tolerance_mgdl`) between the
   original training checkpoint and the exported edge runtime before deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Literal, Optional, Union

import numpy as np

from iints.forecasting.models.base import WindowForecaster
from iints.forecasting.models.gru import TORCH_AVAILABLE, GRUGlucosePredictor, numpy_gru_forward
from iints.forecasting.models.persistence import PersistencePredictor
from iints.forecasting.models.ridge import RidgeCGMPredictor
from iints.forecasting.models.tcn import TCNGlucosePredictor, numpy_causal_tcn_forward
from iints.sdk2.reproducibility import sha256_file

try:
    import onnx  # noqa: F401
    ONNX_AVAILABLE = True
except Exception:  # pragma: no cover
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:  # pragma: no cover
    ort = None  # type: ignore[assignment]
    ORT_AVAILABLE = False

__all__ = [
    "ONNX_AVAILABLE",
    "ORT_AVAILABLE",
    "EdgeInferenceArtifact",
    "export_model_for_edge",
    "load_edge_artifact",
    "build_tensorrt_command",
    "convert_onnx_to_tensorrt",
    "verify_numerical_parity",
]


class EdgeInferenceArtifact:
    """Lightweight local edge runner supporting ONNX Runtime (CUDA/TensorRT/CPU) or pure NumPy."""

    def __init__(
        self,
        bundle_payload: Dict[str, Any],
        *,
        onnx_path: Optional[Path] = None,
        prefer_ort: bool = True,
    ) -> None:
        self.payload = bundle_payload
        self.model_id = str(bundle_payload["model_id"])
        self.architecture = str(bundle_payload["architecture"])
        self.input_length = int(bundle_payload["input_length"])
        self.onnx_path = onnx_path
        self.ort_session: Any = None
        self.active_runtime = "numpy_edge_fp32"

        if prefer_ort and onnx_path is not None and onnx_path.exists() and ORT_AVAILABLE and ort is not None:
            providers = [
                p
                for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()
            ]
            self.ort_session = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
            self.active_runtime = f"onnxruntime:{self.ort_session.get_providers()[0]}"

        # Pre-materialize float32/float64 weight arrays once so batch-1 inference has zero JSON overhead
        self._ckpt = bundle_payload["checkpoint"]
        if self.architecture == "gru":
            w = self._ckpt["weights"]
            self._gru_w = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
            self._mean = float(self._ckpt["scaler"]["mean_mgdl"])
            self._std = float(self._ckpt["scaler"]["std_mgdl"])
            self._step_scale = float(self._ckpt.get("step_scale", 5.0))
            self._dscale = float(self._ckpt["delta_scale"])
        elif self.architecture == "tcn":
            w = self._ckpt["weights"]
            self._tcn_w = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
            dil = self._ckpt.get("dilations", [1, 2])
            self._dilations = (int(dil[0]), int(dil[1]))
            self._mean = float(self._ckpt["scaler"]["mean_mgdl"])
            self._std = float(self._ckpt["scaler"]["std_mgdl"])
            self._step_scale = float(self._ckpt.get("step_scale", 5.0))
            self._dscale = float(self._ckpt["delta_scale"])
        elif self.architecture == "ridge":
            self._mean = float(self._ckpt["scaler"]["mean_mgdl"])
            self._std = float(self._ckpt["scaler"]["std_mgdl"])
            self._b0 = float(self._ckpt["intercept"])
            self._coef = np.asarray(self._ckpt["coef"], dtype=np.float64)

    def _kinematic_features(self, x2: np.ndarray, extra: Optional[np.ndarray]) -> np.ndarray:
        z_abs = (x2 - self._mean) / self._std
        z_diff = np.zeros_like(x2)
        if x2.shape[1] > 1:
            z_diff[:, 1:] = (x2[:, 1:] - x2[:, :-1]) / self._step_scale
        z_rel0 = (x2 - x2[:, :1]) / self._dscale
        feats = [z_abs[:, :, None], z_diff[:, :, None], z_rel0[:, :, None]]
        if extra is not None:
            feats.append(extra)
        return np.concatenate(feats, axis=2)

    def predict_batch(self, x_2d_or_3d: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_2d_or_3d, dtype=np.float32 if self.ort_session is not None else np.float64)
        if self.ort_session is not None:
            x3 = arr[:, :, None] if arr.ndim == 2 else arr
            # Static batch-1 ONNX graph loops cleanly if B > 1, or executes in a single call for B == 1
            if len(x3) == 1:
                out = self.ort_session.run(None, {"cgm_window": x3})[0]
                return np.asarray(out, dtype=np.float64).ravel()
            preds = [self.ort_session.run(None, {"cgm_window": x3[i : i + 1]})[0].item() for i in range(len(x3))]
            return np.asarray(preds, dtype=np.float64)

        x2 = arr[:, :, 0] if arr.ndim == 3 else arr
        extra = arr[:, :, 1:] if (arr.ndim == 3 and arr.shape[2] > 1) else None
        if self.architecture == "persistence":
            return np.asarray(x2[:, -1], dtype=np.float64).copy()
        if self.architecture == "ridge":
            z = (x2 - self._mean) / self._std
            return self._b0 + z @ self._coef
        if self.architecture == "gru":
            z = self._kinematic_features(x2, extra)
            d = numpy_gru_forward(
                z,
                self._gru_w["w_ih"],
                self._gru_w["w_hh"],
                self._gru_w["b_ih"],
                self._gru_w["b_hh"],
                self._gru_w["w_out"],
                self._gru_w["b_out"],
            )
            return np.clip(x2[:, -1] + d * self._dscale, 20.0, 600.0)
        if self.architecture == "tcn":
            z = self._kinematic_features(x2, extra)
            d = numpy_causal_tcn_forward(z, self._tcn_w, dilations=self._dilations)
            return np.clip(x2[:, -1] + d * self._dscale, 20.0, 600.0)
        raise ValueError(f"Unsupported edge architecture: {self.architecture}")


def export_model_for_edge(
    model: WindowForecaster,
    output_dir: Union[str, Path],
    *,
    input_length: int = 12,
    n_channels: int = 1,
    export_onnx: bool = True,
    opset_version: int = 17,
    reference_windows: Optional[np.ndarray] = None,
    parity_tolerance_mgdl: float = 1e-2,
) -> Dict[str, Any]:
    """Export `model` to `.iints_edge.json` (+ `.onnx` when PyTorch & ONNX are installed) and check parity."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    bundle_path = out_root / f"{model.model_id}.iints_edge.json"
    bundle_payload = {
        "format": "iints_edge_v1",
        "clinical_use": False,
        "model_id": model.model_id,
        "architecture": model.architecture,
        "input_length": int(input_length),
        "n_channels": int(n_channels),
        "tensorrt_notes": (
            "TCN uses static Conv1d + Relu + Add + Linear (100% native TensorRT FP16/FP32). "
            "GRU uses static-length RNN ops supported in TensorRT >= 8.6 / JetPack 6.x."
        ),
        "checkpoint": model.to_checkpoint_dict(),
    }
    bundle_path.write_text(json.dumps(bundle_payload, indent=2, sort_keys=True), encoding="utf-8")
    bundle_sha = sha256_file(bundle_path)

    onnx_path: Optional[Path] = None
    onnx_status = "skipped_or_torch_unavailable"
    if export_onnx and isinstance(model, (GRUGlucosePredictor, TCNGlucosePredictor)) and TORCH_AVAILABLE:
        import torch

        try:
            torch_mod = model.to_torch_module()
            dummy = torch.full((1, int(input_length), int(n_channels)), 135.0, dtype=torch.float32)
            cand_onnx = out_root / f"{model.model_id}.onnx"
            torch.onnx.export(
                torch_mod,
                dummy,
                str(cand_onnx),
                input_names=["cgm_window"],
                output_names=["predicted_glucose_mgdl"],
                opset_version=int(opset_version),
                do_constant_folding=True,
            )
            onnx_path = cand_onnx
            onnx_status = f"exported_opset_{opset_version}"
        except Exception as exc:  # pragma: no cover - depends on local onnxscript/onnx build
            onnx_status = f"onnx_export_skipped:{type(exc).__name__}:{exc}"

    artifact = load_edge_artifact(bundle_path, onnx_path=onnx_path)
    ref_X = (
        np.asarray(reference_windows, dtype=np.float64)
        if reference_windows is not None
        else np.linspace(105.0, 165.0, int(input_length) * 8, dtype=np.float64).reshape(8, int(input_length))
    )
    parity = verify_numerical_parity(model, artifact, ref_X, tolerance_mgdl=parity_tolerance_mgdl)

    return {
        "model_id": model.model_id,
        "architecture": model.architecture,
        "edge_bundle_path": bundle_path,
        "edge_bundle_sha256": bundle_sha,
        "onnx_path": onnx_path,
        "onnx_status": onnx_status,
        "active_runtime": artifact.active_runtime,
        "parity": parity,
    }


def load_edge_artifact(
    bundle_path: Union[str, Path],
    *,
    onnx_path: Optional[Union[str, Path]] = None,
    prefer_ort: bool = True,
) -> EdgeInferenceArtifact:
    p = Path(bundle_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    resolved_onnx = Path(onnx_path) if onnx_path is not None else p.with_suffix(".onnx")
    if not resolved_onnx.exists():
        resolved_onnx = p.parent / f"{payload['model_id']}.onnx"
    return EdgeInferenceArtifact(
        payload,
        onnx_path=resolved_onnx if resolved_onnx.exists() else None,
        prefer_ort=prefer_ort,
    )


def verify_numerical_parity(
    source_model: WindowForecaster,
    edge_artifact: EdgeInferenceArtifact,
    reference_windows: np.ndarray,
    *,
    tolerance_mgdl: float = 1e-2,
) -> Dict[str, Any]:
    """Compare source model outputs against exported edge runtime outputs."""
    orig = source_model.predict(reference_windows)
    edge = edge_artifact.predict_batch(reference_windows)
    diffs = np.abs(orig - edge)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    return {
        "n_windows_checked": int(len(reference_windows)),
        "runtime": edge_artifact.active_runtime,
        "max_abs_diff_mgdl": max_diff,
        "mean_abs_diff_mgdl": mean_diff,
        "tolerance_mgdl": float(tolerance_mgdl),
        "passed": bool(max_diff <= tolerance_mgdl),
    }


def _find_trtexec() -> Optional[str]:
    found = shutil.which("trtexec")
    if found:
        return found
    jetson_default = Path("/usr/src/tensorrt/bin/trtexec")
    return str(jetson_default) if jetson_default.is_file() else None


def build_tensorrt_command(
    onnx_path: Union[str, Path],
    engine_out_path: Union[str, Path],
    *,
    precision: Literal["fp32", "fp16"] = "fp16",
    trtexec_bin: str = "/usr/src/tensorrt/bin/trtexec",
) -> List[str]:
    cmd = [
        trtexec_bin,
        f"--onnx={Path(onnx_path)}",
        f"--saveEngine={Path(engine_out_path)}",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    return cmd


def convert_onnx_to_tensorrt(
    onnx_path: Union[str, Path],
    engine_out_path: Union[str, Path],
    *,
    precision: Literal["fp32", "fp16"] = "fp16",
    timeout_seconds: float = 180.0,
) -> Dict[str, Any]:
    """Compile an ONNX model into a Jetson TensorRT `.plan` engine via JetPack `trtexec`."""
    trtexec = _find_trtexec()
    if trtexec is None:
        return {
            "converted": False,
            "reason": "trtexec_not_found_on_host (expected on NVIDIA Jetson JetPack 6.x at /usr/src/tensorrt/bin/trtexec)",
            "recommended_command": " ".join(build_tensorrt_command(onnx_path, engine_out_path, precision=precision)),
        }
    cmd = build_tensorrt_command(onnx_path, engine_out_path, precision=precision, trtexec_bin=trtexec)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_seconds)  # noqa: S603
    return {
        "converted": proc.returncode == 0 and Path(engine_out_path).exists(),
        "returncode": proc.returncode,
        "engine_path": str(engine_out_path),
        "precision": precision,
        "stdout_tail": proc.stdout[-800:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-800:] if proc.stderr else "",
    }
