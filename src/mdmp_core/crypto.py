from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Union
import base64
import binascii
import hmac
import importlib.resources
import json
import os
import secrets

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

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
    "signature_format_version",
    "signature_algorithm",
    "signed_by",
    "signed_at",
    "key_id",
    "pq_commitment",
    "encryption_header",
}

SUPPORTED_SIGNATURE_ALGORITHMS = {
    "ed25519+sha256",
}
CURRENT_SIGNATURE_FORMAT_VERSION = 2


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


# ==============================================================================
# Authenticated Symmetric Encryption (ChaCha20-Poly1305 / RFC 8439) & HKDF
# ==============================================================================

def derive_key_hkdf(
    secret: bytes | str,
    *,
    salt: bytes | None = None,
    info: bytes = b"mdmp-chacha20poly1305-aead-v1",
    length: int = 32,
) -> bytes:
    """
    Derive a 256-bit subkey from high-entropy key material with HKDF-SHA256.

    HKDF is not a password-hardening function. Human passphrases used by the
    encryption helpers are processed with a randomly salted scrypt KDF instead.
    """
    secret_bytes = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    salt_bytes = salt if salt is not None else b"iints-mdmp-biomedical-salt-v1"
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt_bytes,
        info=info,
    )
    return hkdf.derive(secret_bytes)


def derive_key_scrypt(
    passphrase: bytes | str,
    *,
    salt: bytes,
    length: int = 32,
    n: int = 2**14,
    r: int = 8,
    p: int = 1,
) -> bytes:
    """Derive a file-encryption key from a passphrase using salted scrypt."""

    secret = passphrase.encode("utf-8") if isinstance(passphrase, str) else bytes(passphrase)
    if not secret:
        raise MDMPSignatureError("Encryption passphrase cannot be empty")
    if len(salt) < 16:
        raise MDMPSignatureError("scrypt salt must contain at least 16 bytes")
    kdf = Scrypt(salt=salt, length=length, n=n, r=r, p=p)
    return kdf.derive(secret)


def _encryption_key_and_metadata(
    key: bytes | str,
    *,
    salt: bytes | None,
) -> tuple[bytes, Dict[str, Any]]:
    if isinstance(key, bytes) and len(key) == 32:
        return key, {"name": "raw-256"}
    passphrase_bytes = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    if len(passphrase_bytes) < 12:
        raise MDMPSignatureError(
            "New encrypted files require a passphrase of at least 12 UTF-8 bytes "
            "or an exact 32-byte raw key"
        )
    generated_salt = salt or secrets.token_bytes(16)
    return (
        derive_key_scrypt(passphrase_bytes, salt=generated_salt),
        {
            "name": "scrypt",
            "salt": base64.b64encode(generated_salt).decode("ascii"),
            "n": 2**14,
            "r": 8,
            "p": 1,
            "length": 32,
        },
    )


def encrypt_patient_payload(
    payload: Union[Dict[str, Any], bytes, str],
    key: Union[bytes, str],
    *,
    associated_data: Union[bytes, str, None] = None,
    nonce: Optional[bytes] = None,
    salt: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Encrypt sensitive patient data (CGM metrics, trajectories, or clinical profile)
    using ChaCha20-Poly1305 AEAD (RFC 8439).

    Returns a dictionary containing base64-encoded ciphertext, nonce, and cipher metadata.
    """
    raw_key, kdf_metadata = _encryption_key_and_metadata(key, salt=salt)
    chacha = ChaCha20Poly1305(raw_key)

    if isinstance(payload, dict):
        plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    elif isinstance(payload, str):
        plaintext = payload.encode("utf-8")
    else:
        plaintext = bytes(payload)

    aad_bytes: Optional[bytes] = None
    if associated_data is not None:
        aad_bytes = associated_data.encode("utf-8") if isinstance(associated_data, str) else bytes(associated_data)

    gen_nonce = nonce if nonce is not None else secrets.token_bytes(12)
    if len(gen_nonce) != 12:
        raise MDMPSignatureError("ChaCha20-Poly1305 nonce must contain exactly 12 bytes")
    ciphertext = chacha.encrypt(gen_nonce, plaintext, aad_bytes)

    return {
        "format": "mdmp-aead-envelope",
        "format_version": 2,
        "cipher": "chacha20-poly1305",
        "kdf": kdf_metadata,
        "nonce": base64.b64encode(gen_nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "aad_sha256": sha256(aad_bytes).hexdigest() if aad_bytes else "",
        "created_at": _to_iso(_now_utc()),
    }


def decrypt_patient_payload(
    encrypted_payload: Dict[str, Any],
    key: Union[bytes, str],
    *,
    associated_data: Union[bytes, str, None] = None,
    parse_json: bool = True,
) -> Any:
    """
    Decrypt and verify a ChaCha20-Poly1305 encrypted patient payload.
    Raises MDMPSignatureError if ciphertext or authentication tag is tampered with.
    """
    format_name = encrypted_payload.get("format")
    if format_name not in {None, "mdmp-aead-envelope"}:
        raise MDMPSignatureError(f"Unsupported encrypted payload format: {format_name}")
    try:
        format_version = int(encrypted_payload.get("format_version", 1))
    except (TypeError, ValueError) as exc:
        raise MDMPSignatureError("Malformed encrypted payload format version") from exc
    if format_version not in {1, 2}:
        raise MDMPSignatureError(f"Unsupported encrypted payload format version: {format_version}")

    try:
        cipher = str(encrypted_payload["cipher"])
        if cipher != "chacha20-poly1305":
            raise ValueError(f"unsupported cipher: {cipher}")
        nonce_bytes = base64.b64decode(str(encrypted_payload["nonce"]), validate=True)
        ciphertext_bytes = base64.b64decode(str(encrypted_payload["ciphertext"]), validate=True)
    except Exception as exc:
        raise MDMPSignatureError(f"Malformed encrypted payload structure: {exc}") from exc
    if len(nonce_bytes) != 12:
        raise MDMPSignatureError("Malformed encrypted payload structure: nonce must contain 12 bytes")
    if len(ciphertext_bytes) < 16:
        raise MDMPSignatureError("Malformed encrypted payload structure: ciphertext is shorter than its tag")

    kdf_metadata = encrypted_payload.get("kdf")
    if format_version >= 2 and not isinstance(kdf_metadata, dict):
        raise MDMPSignatureError("Version 2 encrypted payload is missing KDF metadata")
    if isinstance(kdf_metadata, dict):
        kdf_name = str(kdf_metadata.get("name", ""))
        if kdf_name == "raw-256":
            if not isinstance(key, bytes) or len(key) != 32:
                raise MDMPSignatureError("This envelope requires a 32-byte raw key")
            raw_key = key
        elif kdf_name == "scrypt":
            try:
                salt_bytes = base64.b64decode(str(kdf_metadata["salt"]), validate=True)
                n = int(kdf_metadata.get("n", 2**14))
                r = int(kdf_metadata.get("r", 8))
                p = int(kdf_metadata.get("p", 1))
                length = int(kdf_metadata.get("length", 32))
            except (KeyError, TypeError, ValueError, binascii.Error) as exc:
                raise MDMPSignatureError(f"Malformed scrypt metadata: {exc}") from exc
            if (n, r, p, length) != (2**14, 8, 1, 32):
                raise MDMPSignatureError("Unsupported scrypt parameter set")
            raw_key = derive_key_scrypt(key, salt=salt_bytes, n=n, r=r, p=p, length=length)
        else:
            raise MDMPSignatureError(f"Unsupported encryption KDF: {kdf_name or 'missing'}")
    else:
        # Read envelopes created by the pre-v2 implementation. New writes never
        # use this deterministic HKDF salt.
        raw_key = key if isinstance(key, bytes) and len(key) == 32 else derive_key_hkdf(key)
    chacha = ChaCha20Poly1305(raw_key)

    aad_bytes: Optional[bytes] = None
    if associated_data is not None:
        aad_bytes = associated_data.encode("utf-8") if isinstance(associated_data, str) else bytes(associated_data)
    expected_aad_hash = str(encrypted_payload.get("aad_sha256", ""))
    actual_aad_hash = sha256(aad_bytes).hexdigest() if aad_bytes else ""
    if not hmac.compare_digest(expected_aad_hash, actual_aad_hash):
        raise MDMPSignatureError("Decryption failed: associated data does not match the encrypted context")

    try:
        plaintext_bytes = chacha.decrypt(nonce_bytes, ciphertext_bytes, aad_bytes)
    except InvalidTag as exc:
        raise MDMPSignatureError("Decryption failed: Ciphertext or Authentication Tag tampered with") from exc

    if parse_json:
        try:
            return json.loads(plaintext_bytes.decode("utf-8"))
        except Exception:
            return plaintext_bytes.decode("utf-8", errors="replace")
    return plaintext_bytes


def encrypt_cgm_dataset_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Union[bytes, str],
    *,
    associated_data: Union[bytes, str, None] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Encrypt a complete CGM dataset file (CSV/Parquet) to a protected `.enc` file.
    """
    in_bytes = Path(input_path).read_bytes()
    encrypted_dict = encrypt_patient_payload(in_bytes, key, associated_data=associated_data)
    out_p = Path(output_path)
    if out_p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite encrypted output: {out_p}")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_p.with_suffix(out_p.suffix + ".tmp")
    temporary.write_text(json.dumps(encrypted_dict, indent=2), encoding="utf-8")
    temporary.replace(out_p)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_sha256": sha256(in_bytes).hexdigest(),
        "encrypted_at": encrypted_dict["created_at"],
    }


def decrypt_cgm_dataset_file(
    encrypted_path: Union[str, Path],
    output_path: Union[str, Path],
    key: Union[bytes, str],
    *,
    associated_data: Union[bytes, str, None] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Decrypt a protected `.enc` dataset file back to its original biomedical format.
    """
    payload_dict = json.loads(Path(encrypted_path).read_text(encoding="utf-8"))
    decrypted_bytes = decrypt_patient_payload(payload_dict, key, associated_data=associated_data, parse_json=False)
    out_p = Path(output_path)
    if out_p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite decrypted output: {out_p}")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_p.with_suffix(out_p.suffix + ".tmp")
    temporary.write_bytes(decrypted_bytes)
    temporary.replace(out_p)
    return {
        "encrypted_path": str(encrypted_path),
        "output_path": str(output_path),
        "decrypted_sha256": sha256(decrypted_bytes).hexdigest(),
    }


# ==============================================================================
# Key Generation & Serialization
# ==============================================================================

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


# ==============================================================================
# Ed25519 signer and verifier
# ==============================================================================

@dataclass
class MDMPSigner:
    private_key_path: str | Path
    signed_by: str = "MDMP-Authority-v1"
    key_id: str = "mdmp_pub_v1"
    private_key_passphrase: str | bytes | None = None
    algorithm: str = "ed25519+sha256"

    def sign_card(
        self,
        card: Dict[str, Any],
        *,
        expires_days: int | None = None,
        algorithm: Optional[str] = None,
    ) -> Dict[str, Any]:
        algo = algorithm or self.algorithm
        if algo not in SUPPORTED_SIGNATURE_ALGORITHMS:
            raise MDMPSignatureError(f"Unsupported signature algorithm: {algo}")

        payload = normalize_unsigned_payload(card)
        if expires_days is not None and "expires" not in payload:
            payload["expires"] = _to_iso(_now_utc() + timedelta(days=max(0, int(expires_days))))

        signed_at = _to_iso(_now_utc())
        signed_material = {
            **payload,
            "signature_format_version": CURRENT_SIGNATURE_FORMAT_VERSION,
            "signature_algorithm": algo,
            "signed_by": self.signed_by,
            "key_id": self.key_id,
            "signed_at": signed_at,
        }
        digest = payload_digest(signed_material)
        signer_obj = load_private_key(self.private_key_path, passphrase=self.private_key_passphrase)
        try:
            signature = signer_obj.sign(digest)
        finally:
            del signer_obj

        result = {
            **signed_material,
            "signature": base64.b64encode(signature).decode("utf-8"),
        }
        return result


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
        signature_algorithm = str(signed_card.get("signature_algorithm", "ed25519+sha256")).strip()
        if signature_algorithm not in SUPPORTED_SIGNATURE_ALGORITHMS:
            return {
                "valid": False,
                "tampered": True,
                "error": "unsupported_signature_algorithm",
                "signature_algorithm": signature_algorithm,
            }

        payload = normalize_unsigned_payload(signed_card)
        raw_format_version = signed_card.get("signature_format_version", 1)
        try:
            signature_format_version = int(raw_format_version)
        except (TypeError, ValueError):
            return {
                "valid": False,
                "tampered": True,
                "error": "invalid_signature_format_version",
            }
        if signature_format_version == 1:
            # Compatibility path for cards written before MDMP signature v2.
            signed_material = payload
        elif signature_format_version == CURRENT_SIGNATURE_FORMAT_VERSION:
            required_metadata = ("signed_by", "key_id", "signed_at")
            if any(not str(signed_card.get(field, "")).strip() for field in required_metadata):
                return {
                    "valid": False,
                    "tampered": True,
                    "error": "missing_signed_signature_metadata",
                    "signature_format_version": signature_format_version,
                }
            signed_material = {
                **payload,
                "signature_format_version": signature_format_version,
                "signature_algorithm": signature_algorithm,
                "signed_by": signed_card["signed_by"],
                "key_id": signed_card["key_id"],
                "signed_at": signed_card["signed_at"],
            }
        else:
            return {
                "valid": False,
                "tampered": True,
                "error": "unsupported_signature_format_version",
                "signature_format_version": signature_format_version,
            }
        digest = payload_digest(signed_material)
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
            "signature_algorithm": signature_algorithm,
            "signature_format_version": signature_format_version,
            "pq_verified": False,
            "identity_error": identity_error,
            "expected_signed_by": (
                trusted_signed_by if self.verification_mode == "trust_store" else None
            ),
        }
        if not valid:
            result["error"] = identity_error or "signature_verification_failed_or_card_invalid"
        return result
