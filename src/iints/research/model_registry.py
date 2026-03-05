from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


ModelStage = Literal["candidate", "validated", "production", "archived"]


@dataclass(frozen=True)
class PromotionResult:
    run_id: str
    stage: ModelStage
    updated: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "updated": self.updated,
            "reason": self.reason,
        }


def load_registry(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Registry file must contain a JSON list")
    rows: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def write_registry(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def append_registry_entry(path: Path, entry: Dict[str, Any]) -> None:
    rows = load_registry(path)
    rows.append(entry)
    write_registry(path, rows)


def list_registry(path: Path) -> List[Dict[str, Any]]:
    return load_registry(path)


def promote_registry_run(
    path: Path,
    *,
    run_id: str,
    stage: ModelStage,
    force: bool = False,
) -> PromotionResult:
    rows = load_registry(path)
    for row in rows:
        if str(row.get("run_id")) != run_id:
            continue
        current_stage = str(row.get("stage", "candidate"))
        if stage == "production" and current_stage != "validated" and not force:
            return PromotionResult(
                run_id=run_id,
                stage=stage,
                updated=False,
                reason="run must be validated before production (use --force to override)",
            )
        row["stage"] = stage
        write_registry(path, rows)
        return PromotionResult(run_id=run_id, stage=stage, updated=True, reason="ok")

    return PromotionResult(run_id=run_id, stage=stage, updated=False, reason="run_id not found")

