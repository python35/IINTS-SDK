from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional
import base64
import binascii
import hmac
import importlib.resources
import json
import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from mdmp_core.exceptions import MDMPSignatureError
from mdmp_core.fingerprint import file_sha256
from mdmp_core.trust import (
    get_trusted_key_label,
    get_trusted_key_pem,
    is_key_revoked,
    load_trust_store,
    resolve_key_id,
)


SIGNATURE_FIELDS = {
    "signature",
    "signature_algorithm",
    "signed_by",
    "signed_at",
    "key_id",
}


def _now_utc() -> datetime:
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


def canonical_payload(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def payload_digest(payload: Dict[str, Any]) -> bytes:
    return sha256(canonical_payload(payload)).digest()


def normalize_unsigned_payload(signed_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in signed_payload.items() if k not in SIGNATURE_FIELDS}


def normalize_passphrase(passphrase: str | bytes | None) -> bytes | None:
    if passphrase is None:
        return None
    if isinstance(passphrase, bytes):
        data = passphrase
    else:
        data = str(passphrase).encode("utf-8")
    if not data:
        raise MDMPSignatureError("Private key passphrase cannot be empty")
    return data


def generate_keypair(
    *,
    output_dir: str | Path,
    private_name: str = "mdmp_private_v1.pem",
    public_name: str = "mdmp_pub_v1.pem",
    passphrase: str | bytes | None = None,
) -> Dict[str, Any]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    private_path = output / private_name
    public_path = output / public_name

    password = normalize_passphrase(passphrase)
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=(
            serialization.BestAvailableEncryption(password)
            if password is not None
            else serialization.NoEncryption()
        ),
    )
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_path.write_bytes(private_bytes)
    os.chmod(private_path, 0o600)
    public_path.write_bytes(public_bytes)
    return {
        "private_key": str(private_path),
        "public_key": str(public_path),
        "private_key_encrypted": password is not None,
    }


def load_private_key(
    path: str | Path,
    *,
    passphrase: str | bytes | None = None,
) -> Ed25519PrivateKey:
    password = normalize_passphrase(passphrase)
    try:
        raw = Path(path).read_bytes()
        key = serialization.load_pem_private_key(raw, password=password)
    except TypeError as exc:
        if password is None:
            raise MDMPSignatureError(f"Private key is encrypted and requires a passphrase: {path}") from exc
        raise MDMPSignatureError(f"Private key requires a valid passphrase: {path}") from exc
    except ValueError as exc:
        if password is not None:
            raise MDMPSignatureError(f"Invalid private key passphrase: {path}") from exc
        raise MDMPSignatureError(f"Failed to parse private key: {path}") from exc
    except Exception as exc:
        raise MDMPSignatureError(f"Failed to load private key: {path}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise MDMPSignatureError("Private key is not Ed25519")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        raw = Path(path).read_bytes()
        key = serialization.load_pem_public_key(raw)
    except Exception as exc:
        raise MDMPSignatureError(f"Failed to load public key: {path}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise MDMPSignatureError("Public key is not Ed25519")
    return key


def load_public_key_from_pem_text(raw: str) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(raw.encode("utf-8"))
    if not isinstance(key, Ed25519PublicKey):
        raise MDMPSignatureError("Public key is not Ed25519")
    return key


def load_bundled_public_key() -> Ed25519PublicKey:
    key_path = importlib.resources.files("mdmp_core").joinpath("keys/mdmp_pub_v1.pem")
    raw = key_path.read_bytes()
    key = serialization.load_pem_public_key(raw)
    if not isinstance(key, Ed25519PublicKey):
        raise MDMPSignatureError("Bundled key is not Ed25519")
    return key


def _expected_fingerprint(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("dataset_fingerprint", "dataset_fingerprint_sha256", "fingerprint"):
        if payload.get(key):
            value = str(payload[key]).strip()
            if key == "dataset_fingerprint_sha256" and not value.startswith("sha256:"):
                return f"sha256:{value}"
            if value and not value.startswith("sha256:"):
                return f"sha256:{value}"
            return value
    return None


@dataclass
class MDMPSigner:
    private_key_path: str | Path
    signed_by: str = "MDMP-Authority-v1"
    key_id: str = "mdmp_pub_v1"
    private_key_passphrase: str | bytes | None = None

    def sign_card(
        self,
        card: Dict[str, Any],
        *,
        expires_days: int | None = None,
    ) -> Dict[str, Any]:
        payload = dict(card)
        if expires_days is not None and "expires" not in payload:
            payload["expires"] = _to_iso(_now_utc() + timedelta(days=max(0, int(expires_days))))

        digest = payload_digest(payload)
        signer_obj = load_private_key(self.private_key_path, passphrase=self.private_key_passphrase)
        try:
            signature = signer_obj.sign(digest)
        finally:
            del signer_obj
        return {
            **payload,
            "signature": base64.b64encode(signature).decode("utf-8"),
            "signature_algorithm": "ed25519+sha256",
            "signed_by": self.signed_by,
            "key_id": self.key_id,
            "signed_at": _to_iso(_now_utc()),
        }


@dataclass
class MDMPVerifier:
    public_key_path: str | Path | None = None
    trust_store_path: str | Path | None = None

    def __post_init__(self) -> None:
        self.pubkey: Ed25519PublicKey | None
        self.verification_mode: str
        if self.public_key_path:
            self.pubkey = load_public_key(self.public_key_path)
            self.verification_mode = "explicit_key"
        elif self.trust_store_path:
            self.pubkey = None
            self.verification_mode = "trust_store"
        else:
            self.pubkey = load_bundled_public_key()
            self.verification_mode = "bundled_root"

    def _resolve_public_key(
        self,
        signed_card: Dict[str, Any],
    ) -> tuple[Ed25519PublicKey | None, str | None, str | None, str | None]:
        key_id_raw = signed_card.get("key_id")
        key_id = str(key_id_raw).strip() if key_id_raw is not None else None

        if self.trust_store_path is None:
            return self.pubkey, key_id, None, None

        store = load_trust_store(self.trust_store_path)
        resolved_key_id = resolve_key_id(store, key_id)
        if not resolved_key_id:
            return None, None, "missing_key_id_and_no_active_key", None
        if is_key_revoked(store, resolved_key_id):
            return None, resolved_key_id, "key_revoked", None

        pem = get_trusted_key_pem(store, resolved_key_id)
        if not pem:
            return None, resolved_key_id, "unknown_key_id", None
        return (
            load_public_key_from_pem_text(pem),
            resolved_key_id,
            None,
            get_trusted_key_label(store, resolved_key_id),
        )

    def verify(
        self,
        signed_card: Dict[str, Any],
        *,
        dataset_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        signature_b64 = signed_card.get("signature")
        if not signature_b64:
            return {
                "valid": False,
                "tampered": True,
                "error": "missing_signature",
            }
        signature_algorithm = str(signed_card.get("signature_algorithm", "")).strip()
        if signature_algorithm and signature_algorithm != "ed25519+sha256":
            return {
                "valid": False,
                "tampered": True,
                "error": "unsupported_signature_algorithm",
                "signature_algorithm": signature_algorithm,
            }

        payload = normalize_unsigned_payload(signed_card)
        digest = payload_digest(payload)
        pubkey, resolved_key_id, key_error, trusted_signed_by = self._resolve_public_key(signed_card)
        if key_error:
            return {
                "valid": False,
                "tampered": False,
                "signature_valid": False,
                "issued_by": signed_card.get("signed_by"),
                "key_id": resolved_key_id,
                "signed_at": signed_card.get("signed_at"),
                "grade": payload.get("grade"),
                "expired": False,
                "error": key_error,
            }

        try:
            signature = base64.b64decode(str(signature_b64), validate=True)
            if pubkey is None:
                raise ValueError("missing_public_key")
            pubkey.verify(signature, digest)
            signature_valid = True
        except (binascii.Error, ValueError, TypeError):
            signature_valid = False
        except Exception:
            signature_valid = False

        expected_fp = _expected_fingerprint(payload)
        fingerprint_matches = None
        actual_fp = None
        if dataset_path is not None and expected_fp is not None:
            actual_fp = f"sha256:{file_sha256(dataset_path)}"
            fingerprint_matches = hmac.compare_digest(actual_fp, expected_fp)

        expired = False
        expires = payload.get("expires")
        if isinstance(expires, str) and expires.strip():
            try:
                expired = _now_utc() > _parse_iso(expires)
            except Exception:
                expired = True

        identity_error = None
        if signature_valid and self.verification_mode == "bundled_root":
            expected_key_id = "mdmp_pub_v1"
            expected_signed_by = "MDMP-Authority-v1"
            actual_key_id = str(signed_card.get("key_id", "")).strip()
            actual_signed_by = str(signed_card.get("signed_by", "")).strip()
            key_ok = hmac.compare_digest(actual_key_id, expected_key_id)
            signer_ok = hmac.compare_digest(actual_signed_by, expected_signed_by)
            if not key_ok:
                identity_error = "key_id_mismatch_for_bundled_root"
            elif not signer_ok:
                identity_error = "signed_by_mismatch_for_bundled_root"
        elif signature_valid and self.verification_mode == "trust_store" and trusted_signed_by:
            actual_signed_by = str(signed_card.get("signed_by", "")).strip()
            if not hmac.compare_digest(actual_signed_by, trusted_signed_by):
                identity_error = "signed_by_mismatch_for_trusted_key"

        valid = bool(
            signature_valid
            and (fingerprint_matches in {None, True})
            and not expired
            and identity_error is None
        )
        result = {
            "valid": valid,
            "tampered": not signature_valid,
            "signature_valid": signature_valid,
            "issued_by": signed_card.get("signed_by"),
            "key_id": resolved_key_id or signed_card.get("key_id"),
            "signed_at": signed_card.get("signed_at"),
            "grade": payload.get("grade"),
            "expires": expires,
            "expired": expired,
            "expected_fingerprint": expected_fp,
            "actual_fingerprint": actual_fp,
            "fingerprint_matches": fingerprint_matches,
            "verification_mode": self.verification_mode,
            "identity_error": identity_error,
            "expected_signed_by": (
                trusted_signed_by if self.verification_mode == "trust_store" else None
            ),
        }
        if not valid:
            result["error"] = identity_error or "signature_verification_failed_or_card_invalid"
        return result
