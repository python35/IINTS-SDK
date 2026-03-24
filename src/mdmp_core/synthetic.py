from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_synthetic_metadata(
    *,
    generator: str,
    source_fingerprint: str,
    method: str,
    privacy_epsilon: float | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "spec_version": "1.0",
        "mdmp_object": "synthetic_metadata",
        "synthetic": True,
        "generator": generator,
        "generated_utc": _now_iso(),
        "source_fingerprint": source_fingerprint,
        "method": method,
        "privacy_epsilon": privacy_epsilon,
        "notes": notes,
    }
    return payload
