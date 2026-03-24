from __future__ import annotations

from pathlib import Path


def log_mdmp_artifact_to_mlflow(path: str | Path, *, artifact_path: str = "mdmp") -> str:
    target = str(Path(path))
    try:
        import mlflow  # type: ignore
    except Exception:
        return "mlflow_not_installed"

    mlflow.log_artifact(target, artifact_path=artifact_path)
    return "ok"
