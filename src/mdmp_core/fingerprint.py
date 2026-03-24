from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
from pathlib import Path
from typing import Any, Dict, Optional

from mdmp_core.exceptions import MDMPFingerprintError


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def file_sha256(path: str | Path) -> str:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise MDMPFingerprintError(f"Unable to read dataset for fingerprint: {path}") from exc
    return sha256(payload).hexdigest()


def compute_fingerprint(path: str | Path, *, expires_days: int = 365) -> Dict[str, Any]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise MDMPFingerprintError(f"Dataset path does not exist: {dataset_path}")
    now = _utc_now()
    expires = now + timedelta(days=max(0, int(expires_days)))
    digest = file_sha256(dataset_path)
    return {
        "fingerprint": f"sha256:{digest}",
        "created": _to_iso(now),
        "expires": _to_iso(expires),
        "source_path": str(dataset_path),
        "status": "valid",
        "stale_reason": None,
    }


def check_fingerprint(
    stored_record: Dict[str, Any],
    *,
    data_path: str | Path,
    checked_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = checked_at or _utc_now()
    current_sha = f"sha256:{file_sha256(data_path)}"
    expected_sha = str(stored_record.get("fingerprint", "")).strip()
    expires_raw = str(stored_record.get("expires", "")).strip()

    mismatch = not hmac.compare_digest(current_sha, expected_sha)
    expired = False
    if expires_raw:
        try:
            expired = now > _parse_iso(expires_raw)
        except Exception:
            expired = True

    reason: Optional[str]
    if mismatch:
        reason = "data_changed"
    elif expired:
        reason = "expired"
    else:
        reason = None

    result = dict(stored_record)
    result.update(
        {
            "status": "stale" if reason else "valid",
            "stale_reason": reason,
            "checked_at": _to_iso(now),
            "current_fingerprint": current_sha,
            "source_path": str(data_path),
        }
    )
    return result
