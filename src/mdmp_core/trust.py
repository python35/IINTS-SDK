from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


TRUST_STORE_VERSION = "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_store() -> dict[str, Any]:
    return {
        "version": TRUST_STORE_VERSION,
        "active_key_id": None,
        "trusted_keys": {},
        "revoked_keys": {},
        "revoked_delegates": {},
        "updated_utc": _now_iso(),
    }


def _normalize_trusted_key_entry(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        text = value.strip()
        if "BEGIN PUBLIC KEY" not in text:
            return None
        return {
            "pem": text + ("\n" if not text.endswith("\n") else ""),
            "signed_by": None,
        }

    if isinstance(value, dict):
        raw_pem = value.get("pem", value.get("public_key"))
        if not isinstance(raw_pem, str):
            return None
        text = raw_pem.strip()
        if "BEGIN PUBLIC KEY" not in text:
            return None
        signed_by = value.get("signed_by")
        if signed_by is not None:
            signed_by = str(signed_by).strip() or None
        return {
            "pem": text + ("\n" if not text.endswith("\n") else ""),
            "signed_by": signed_by,
        }

    return None


def _normalize_store(payload: dict[str, Any]) -> dict[str, Any]:
    store = _default_store()
    if not isinstance(payload, dict):
        return store

    store["version"] = str(payload.get("version", TRUST_STORE_VERSION))
    store["active_key_id"] = payload.get("active_key_id")
    trusted_keys: dict[str, Any] = {}
    for key_id, raw_value in dict(payload.get("trusted_keys", {}) or {}).items():
        normalized_entry = _normalize_trusted_key_entry(raw_value)
        if normalized_entry is not None:
            trusted_keys[str(key_id)] = normalized_entry
    store["trusted_keys"] = trusted_keys
    store["revoked_keys"] = dict(payload.get("revoked_keys", {}) or {})
    store["revoked_delegates"] = dict(payload.get("revoked_delegates", {}) or {})
    store["updated_utc"] = str(payload.get("updated_utc", _now_iso()))
    return store


def load_trust_store(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return _default_store()
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _default_store()
    return _normalize_store(payload if isinstance(payload, dict) else {})


def save_trust_store(path: str | Path, store: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_store(store)
    normalized["updated_utc"] = _now_iso()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def _load_public_pem_text(public_key_path: str | Path) -> str:
    text = Path(public_key_path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("public key file is empty")
    if "BEGIN PUBLIC KEY" not in text:
        raise ValueError("public key file is not PEM")
    return text + ("\n" if not text.endswith("\n") else "")


def trust_init(
    trust_store_path: str | Path,
    *,
    key_id: str,
    public_key_path: str | Path,
    set_active: bool = True,
    signed_by: str | None = None,
) -> dict[str, Any]:
    store = _default_store()
    pem = _load_public_pem_text(public_key_path)
    store["trusted_keys"][key_id] = {"pem": pem, "signed_by": signed_by}
    if set_active:
        store["active_key_id"] = key_id
    return save_trust_store(trust_store_path, store)


def trust_add_key(
    trust_store_path: str | Path,
    *,
    key_id: str,
    public_key_path: str | Path,
    set_active: bool = False,
    signed_by: str | None = None,
) -> dict[str, Any]:
    store = load_trust_store(trust_store_path)
    pem = _load_public_pem_text(public_key_path)
    store["trusted_keys"][key_id] = {"pem": pem, "signed_by": signed_by}
    if set_active:
        store["active_key_id"] = key_id
    return save_trust_store(trust_store_path, store)


def trust_revoke_key(
    trust_store_path: str | Path,
    *,
    key_id: str,
    reason: str = "revoked",
) -> dict[str, Any]:
    store = load_trust_store(trust_store_path)
    store["revoked_keys"][key_id] = {"revoked_at": _now_iso(), "reason": reason}
    return save_trust_store(trust_store_path, store)


def trust_revoke_delegate(
    trust_store_path: str | Path,
    *,
    delegate_id: str,
    reason: str = "revoked",
) -> dict[str, Any]:
    store = load_trust_store(trust_store_path)
    store["revoked_delegates"][delegate_id] = {"revoked_at": _now_iso(), "reason": reason}
    return save_trust_store(trust_store_path, store)


def trust_unrevoke_key(trust_store_path: str | Path, *, key_id: str) -> dict[str, Any]:
    store = load_trust_store(trust_store_path)
    store["revoked_keys"].pop(key_id, None)
    return save_trust_store(trust_store_path, store)


def trust_unrevoke_delegate(trust_store_path: str | Path, *, delegate_id: str) -> dict[str, Any]:
    store = load_trust_store(trust_store_path)
    store["revoked_delegates"].pop(delegate_id, None)
    return save_trust_store(trust_store_path, store)


def get_trusted_key_pem(store: dict[str, Any], key_id: str | None) -> str | None:
    if not key_id:
        return None
    trusted_keys = store.get("trusted_keys", {})
    if not isinstance(trusted_keys, dict):
        return None
    entry = _normalize_trusted_key_entry(trusted_keys.get(key_id))
    if entry is None:
        return None
    return entry["pem"]


def get_trusted_key_label(store: dict[str, Any], key_id: str | None) -> str | None:
    if not key_id:
        return None
    trusted_keys = store.get("trusted_keys", {})
    if not isinstance(trusted_keys, dict):
        return None
    entry = _normalize_trusted_key_entry(trusted_keys.get(key_id))
    if entry is None:
        return None
    signed_by = entry.get("signed_by")
    if not isinstance(signed_by, str):
        return None
    return signed_by.strip() or None


def resolve_key_id(store: dict[str, Any], key_id: str | None) -> str | None:
    if key_id:
        return key_id
    active = store.get("active_key_id")
    if isinstance(active, str) and active.strip():
        return active.strip()
    return None


def is_key_revoked(store: dict[str, Any], key_id: str | None) -> bool:
    if not key_id:
        return False
    revoked = store.get("revoked_keys", {})
    return isinstance(revoked, dict) and key_id in revoked


def is_delegate_revoked(store: dict[str, Any], delegate_id: str | None) -> bool:
    if not delegate_id:
        return False
    revoked = store.get("revoked_delegates", {})
    return isinstance(revoked, dict) and delegate_id in revoked
