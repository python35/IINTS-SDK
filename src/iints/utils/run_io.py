from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import platform
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from importlib.metadata import version as pkg_version
except Exception:  # pragma: no cover - stdlib fallback
    pkg_version = None  # type: ignore[assignment]


def resolve_seed(seed: Optional[int]) -> int:
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    return int(seed)


def generate_run_id(seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:6]
    return f"{timestamp}-{seed}-{token}"


def resolve_output_dir(output_dir: Optional[Union[str, Path]], run_id: str) -> Path:
    if output_dir is None:
        output_path = Path.cwd() / "results" / run_id
    else:
        output_path = Path(output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        else:
            output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _serialize_payload(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    safe_payload = {key: _serialize_payload(value) for key, value in payload.items()}
    path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True))


def get_sdk_version(package_name: str = "iints-sdk-python35") -> str:
    if pkg_version is None:
        return "unknown"
    try:
        return pkg_version(package_name)
    except Exception:
        return "unknown"


def build_run_metadata(run_id: str, seed: int, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "output_dir": str(output_dir),
        "sdk_version": get_sdk_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
    }
