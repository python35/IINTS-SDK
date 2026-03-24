from __future__ import annotations

from pathlib import Path


def log_mdmp_artifact_to_wandb(path: str | Path, *, name: str = "mdmp-artifact") -> str:
    target = str(Path(path))
    try:
        import wandb  # type: ignore
    except Exception:
        return "wandb_not_installed"

    run = wandb.run
    if run is None:
        return "wandb_run_not_active"
    artifact = wandb.Artifact(name=name, type="mdmp")
    artifact.add_file(target)
    run.log_artifact(artifact)
    return "ok"
