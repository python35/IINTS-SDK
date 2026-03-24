from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def build_dvc_stage(
    *,
    stage_name: str,
    contract_path: str,
    dataset_path: str,
    report_path: str,
) -> Dict[str, Any]:
    cmd = f"mdmp validate {contract_path} {dataset_path} --output-json {report_path}"
    return {
        "stages": {
            stage_name: {
                "cmd": cmd,
                "deps": [contract_path, dataset_path],
                "outs": [report_path],
            }
        }
    }


def write_dvc_stage(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
