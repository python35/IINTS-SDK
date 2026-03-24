from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import base64
import binascii
import hmac
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from mdmp_core.crypto import (
    MDMPSigner,
    MDMPVerifier,
    load_private_key,
    normalize_unsigned_payload,
    payload_digest,
)
from mdmp_core.exceptions import MDMPSignatureError
from mdmp_core.fingerprint import file_sha256
from mdmp_core.trust import is_delegate_revoked, load_trust_store


NON_DELEGABLE_GRADES = {"clinical_grade", "ai_ready"}
DELEGATE_SIGNATURE_FIELDS = {"signature", "signature_algorithm", "signed_at"}


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


def _expected_fingerprint(payload: dict[str, Any]) -> str | None:
    for key in ("dataset_fingerprint", "dataset_fingerprint_sha256", "fingerprint"):
        if key in payload and payload[key] is not None:
            value = str(payload[key]).strip()
            if not value:
                continue
            if value.startswith("sha256:"):
                return value
            return f"sha256:{value}"
    return None


def _load_delegate_public_key(delegate_cert_payload: dict[str, Any]) -> Ed25519PublicKey:
    raw_b64 = delegate_cert_payload.get("delegate_pubkey")
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        raise ValueError("delegate certificate missing delegate_pubkey")
    try:
        raw = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError("delegate_pubkey is not valid base64") from exc
    key = serialization.load_pem_public_key(raw)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("delegate public key is not Ed25519")
    return key


@dataclass
class DelegateConstraints:
    max_expires_days: int = 365
    require_consent_field: bool = False
    allowed_flavors: list[str] = field(default_factory=list)
    allowed_grades: list[str] = field(default_factory=lambda: ["draft", "research_grade"])

    def validate_card(self, card_payload: dict[str, Any]) -> list[str]:
        violations: list[str] = []
        grade = str(card_payload.get("grade", "")).strip()
        if grade not in self.allowed_grades:
            violations.append(
                f"grade '{grade}' not allowed for delegate "
                f"(allowed: {', '.join(self.allowed_grades) or 'none'})"
            )

        if self.require_consent_field and "consent" not in card_payload:
            violations.append("required field 'consent' is missing")

        if self.allowed_flavors:
            flavor = str(card_payload.get("flavor", "")).strip()
            if flavor not in self.allowed_flavors:
                violations.append(
                    f"flavor '{flavor}' not allowed "
                    f"(allowed: {', '.join(self.allowed_flavors)})"
                )

        expires = card_payload.get("expires")
        issued_at = card_payload.get("issued_at")
        if expires and issued_at and self.max_expires_days >= 0:
            try:
                expires_dt = _parse_iso(str(expires))
                issued_dt = _parse_iso(str(issued_at))
                delta_days = (expires_dt - issued_dt).total_seconds() / 86400.0
                if delta_days > float(self.max_expires_days):
                    violations.append(
                        f"expires exceeds delegate max_expires_days ({self.max_expires_days})"
                    )
            except Exception:
                violations.append("invalid issued_at/expires timestamp format")
        return violations

    @classmethod
    def from_payload(cls, payload: dict[str, Any], allowed_grades: list[str]) -> "DelegateConstraints":
        raw = payload.get("constraints", {})
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            max_expires_days=int(raw.get("max_expires_days", 365)),
            require_consent_field=bool(raw.get("require_consent_field", False)),
            allowed_flavors=list(raw.get("allowed_flavors", []) or []),
            allowed_grades=list(allowed_grades),
        )


class DelegateIssuer:
    def __init__(self, root_signer: MDMPSigner):
        self.root_signer = root_signer

    def issue(
        self,
        *,
        delegate_id: str,
        delegate_name: str,
        delegate_pubkey_path: str | Path,
        allowed_grades: list[str],
        valid_days: int = 365,
        constraints: DelegateConstraints | None = None,
    ) -> dict[str, Any]:
        grades = [g.strip() for g in allowed_grades if g.strip()]
        if not grades:
            raise ValueError("allowed_grades cannot be empty")
        forbidden = NON_DELEGABLE_GRADES.intersection(grades)
        if forbidden:
            blocked = ", ".join(sorted(forbidden))
            raise ValueError(f"non-delegable grade requested: {blocked}")

        pubkey_bytes = Path(delegate_pubkey_path).read_bytes()
        constraints = constraints or DelegateConstraints(allowed_grades=grades)
        constraints.allowed_grades = grades

        issued_at = _now_utc()
        cert_payload = {
            "mdmp_object": "delegate_certificate",
            "version": "1",
            "delegate_id": delegate_id,
            "delegate_name": delegate_name,
            "delegate_pubkey": base64.b64encode(pubkey_bytes).decode("utf-8"),
            "allowed_grades": grades,
            "issued_at": _to_iso(issued_at),
            "expires": _to_iso(issued_at + timedelta(days=max(0, int(valid_days)))),
            "constraints": {
                "max_expires_days": int(constraints.max_expires_days),
                "require_consent_field": bool(constraints.require_consent_field),
                "allowed_flavors": list(constraints.allowed_flavors),
                "allowed_grades": grades,
            },
        }
        return self.root_signer.sign_card(cert_payload)


class DelegateSigner:
    def __init__(
        self,
        delegate_private_key_path: str | Path,
        delegate_cert: dict[str, Any],
        *,
        root_public_key_path: str | Path | None = None,
        trust_store_path: str | Path | None = None,
        delegate_private_key_passphrase: str | bytes | None = None,
    ):
        self.delegate_private_key_path = Path(delegate_private_key_path)
        self.delegate_private_key_passphrase = delegate_private_key_passphrase
        if not isinstance(delegate_cert, dict):
            raise ValueError("delegate certificate must be a JSON object")

        cert_verification = MDMPVerifier(
            root_public_key_path,
            trust_store_path=trust_store_path,
        ).verify(delegate_cert)
        if not cert_verification.get("valid"):
            error = cert_verification.get("error", "delegate_certificate_invalid")
            raise MDMPSignatureError(f"delegate certificate verification failed: {error}")

        self.delegate_cert = delegate_cert
        cert_payload = normalize_unsigned_payload(delegate_cert)
        self.delegate_id = str(cert_payload.get("delegate_id", "")).strip()
        self.allowed_grades = list(cert_payload.get("allowed_grades", []) or [])
        if not self.delegate_id:
            raise MDMPSignatureError("delegate certificate missing delegate_id")
        if not self.allowed_grades:
            raise MDMPSignatureError("delegate certificate missing allowed_grades")

        expected_pub = _load_delegate_public_key(cert_payload).public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        loaded = None
        try:
            loaded = load_private_key(
                self.delegate_private_key_path,
                passphrase=self.delegate_private_key_passphrase,
            )
            derived_pub = loaded.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        finally:
            if loaded is not None:
                del loaded
        if not hmac.compare_digest(derived_pub, expected_pub):
            raise MDMPSignatureError("delegate private key does not match delegate certificate public key")

        self.constraints = DelegateConstraints.from_payload(cert_payload, self.allowed_grades)

    def sign_card(
        self,
        card: dict[str, Any],
        *,
        default_expires_days: int | None = None,
    ) -> dict[str, Any]:
        unsigned = dict(card)
        now = _now_utc()
        unsigned.setdefault("issued_at", _to_iso(now))
        unsigned.setdefault("issued_by_delegate", self.delegate_id)
        unsigned.setdefault("delegate_cert_id", self.delegate_id)
        if default_expires_days is not None and "expires" not in unsigned:
            unsigned["expires"] = _to_iso(now + timedelta(days=max(0, int(default_expires_days))))

        violations = self.constraints.validate_card(unsigned)
        if violations:
            details = "; ".join(violations)
            raise PermissionError(f"delegate constraints violation: {details}")

        digest = payload_digest(unsigned)
        signer_obj = load_private_key(
            self.delegate_private_key_path,
            passphrase=self.delegate_private_key_passphrase,
        )
        try:
            signature = signer_obj.sign(digest)
        finally:
            del signer_obj
        return {
            **unsigned,
            "signature": base64.b64encode(signature).decode("utf-8"),
            "signature_algorithm": "ed25519+sha256",
            "signed_at": _to_iso(_now_utc()),
        }


class DelegateVerifier:
    def __init__(
        self,
        root_public_key_path: str | Path | None = None,
        trust_store_path: str | Path | None = None,
    ):
        self.root_verifier = MDMPVerifier(
            root_public_key_path,
            trust_store_path=trust_store_path,
        )
        self.trust_store_path = trust_store_path

    def verify(
        self,
        signed_card: dict[str, Any],
        delegate_cert: dict[str, Any],
        *,
        dataset_path: str | Path | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "valid": False,
            "grade": signed_card.get("grade"),
            "delegate": signed_card.get("issued_by_delegate"),
            "chain": [],
        }

        cert_check = self.root_verifier.verify(delegate_cert)
        result["chain"].append(
            {
                "step": "delegate_cert",
                "valid": bool(cert_check.get("valid")),
                "detail": "root signature on delegate certificate",
            }
        )
        if not cert_check.get("valid"):
            result["error"] = "delegate_certificate_invalid"
            return result

        cert_payload = normalize_unsigned_payload(delegate_cert)
        cert_expired = False
        cert_expires = cert_payload.get("expires")
        if isinstance(cert_expires, str) and cert_expires.strip():
            try:
                cert_expired = _now_utc() > _parse_iso(cert_expires)
            except Exception:
                cert_expired = True
        result["chain"].append(
            {
                "step": "delegate_expiry",
                "valid": not cert_expired,
                "detail": f"delegate cert expires at {cert_expires}",
            }
        )
        if cert_expired:
            result["error"] = "delegate_certificate_expired"
            return result

        delegate_id = str(cert_payload.get("delegate_id", "")).strip()
        delegate_revoked = False
        if self.trust_store_path is not None:
            store = load_trust_store(self.trust_store_path)
            delegate_revoked = is_delegate_revoked(store, delegate_id)
        result["chain"].append(
            {
                "step": "delegate_revocation",
                "valid": not delegate_revoked,
                "detail": f"delegate '{delegate_id}' not revoked",
            }
        )
        if delegate_revoked:
            result["error"] = "delegate_revoked"
            return result

        allowed_grades = list(cert_payload.get("allowed_grades", []) or [])
        grade = str(signed_card.get("grade", "")).strip()
        grade_ok = grade in allowed_grades
        result["chain"].append(
            {
                "step": "grade_permission",
                "valid": grade_ok,
                "detail": f"grade '{grade}' allowed for delegate '{delegate_id}'",
            }
        )
        if not grade_ok:
            result["error"] = "delegate_grade_not_allowed"
            return result

        unsigned_card = {
            k: v for k, v in signed_card.items() if k not in DELEGATE_SIGNATURE_FIELDS
        }
        sig_b64 = signed_card.get("signature")
        if not isinstance(sig_b64, str) or not sig_b64.strip():
            result["chain"].append(
                {
                    "step": "card_signature",
                    "valid": False,
                    "detail": "missing delegate card signature",
                }
            )
            result["error"] = "delegate_card_missing_signature"
            return result
        sig_algo = str(signed_card.get("signature_algorithm", "")).strip()
        if sig_algo and sig_algo != "ed25519+sha256":
            result["chain"].append(
                {
                    "step": "signature_algorithm",
                    "valid": False,
                    "detail": f"unsupported delegate signature algorithm: {sig_algo}",
                }
            )
            result["error"] = "delegate_card_unsupported_signature_algorithm"
            return result

        try:
            delegate_pub = _load_delegate_public_key(cert_payload)
            signature = base64.b64decode(sig_b64, validate=True)
            delegate_pub.verify(signature, payload_digest(unsigned_card))
            sig_valid = True
        except (binascii.Error, ValueError, TypeError):
            sig_valid = False
        except Exception:
            sig_valid = False

        result["chain"].append(
            {
                "step": "card_signature",
                "valid": sig_valid,
                "detail": "delegate signature on card payload",
            }
        )
        if not sig_valid:
            result["error"] = "delegate_card_signature_invalid"
            return result

        issuer_match = hmac.compare_digest(
            str(unsigned_card.get("issued_by_delegate", "")).strip(),
            delegate_id,
        )
        result["chain"].append(
            {
                "step": "delegate_identity",
                "valid": issuer_match,
                "detail": f"card delegate matches cert delegate_id ({delegate_id})",
            }
        )
        if not issuer_match:
            result["error"] = "delegate_identity_mismatch"
            return result

        constraints = DelegateConstraints.from_payload(cert_payload, allowed_grades)
        violations = constraints.validate_card(unsigned_card)
        constraints_ok = len(violations) == 0
        result["chain"].append(
            {
                "step": "constraints",
                "valid": constraints_ok,
                "detail": "all constraints satisfied" if constraints_ok else "; ".join(violations),
            }
        )
        if not constraints_ok:
            result["error"] = "delegate_constraints_violation"
            result["constraint_violations"] = violations
            return result

        card_expired = False
        expires = unsigned_card.get("expires")
        if isinstance(expires, str) and expires.strip():
            try:
                card_expired = _now_utc() > _parse_iso(expires)
            except Exception:
                card_expired = True
        result["chain"].append(
            {
                "step": "card_expiry",
                "valid": not card_expired,
                "detail": f"card expires at {expires}",
            }
        )
        if card_expired:
            result["error"] = "delegate_card_expired"
            return result

        expected_fp = _expected_fingerprint(unsigned_card)
        actual_fp = None
        fingerprint_matches = None
        if dataset_path is not None and expected_fp is not None:
            actual_fp = f"sha256:{file_sha256(dataset_path)}"
            fingerprint_matches = hmac.compare_digest(actual_fp, expected_fp)
            result["chain"].append(
                {
                    "step": "fingerprint_match",
                    "valid": bool(fingerprint_matches),
                    "detail": f"expected={expected_fp} actual={actual_fp}",
                }
            )
            if not fingerprint_matches:
                result["error"] = "dataset_fingerprint_mismatch"
                return result

        result.update(
            {
                "valid": True,
                "issued_by": cert_payload.get("delegate_name"),
                "delegate_id": delegate_id,
                "cert_expires": cert_expires,
                "expected_fingerprint": expected_fp,
                "actual_fingerprint": actual_fp,
                "fingerprint_matches": fingerprint_matches,
            }
        )
        return result
