from __future__ import annotations

# Modified in the bundled IINTS-SDK distribution to load conformance vectors
# from packaged resources as well as the repository layout.

from datetime import datetime, timezone
import importlib.resources
from pathlib import Path
from typing import Any
import base64
import hashlib
import json

import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mdmp_core.contracts import parse_contract
from mdmp_core.crypto import MDMPSigner, MDMPVerifier, generate_keypair, payload_digest
from mdmp_core.delegate import DelegateVerifier
from mdmp_core.fingerprint import check_fingerprint, compute_fingerprint
from mdmp_core.policy import PolicySpec, evaluate_policy
from mdmp_core.runner import ContractRunner


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _load_vector_file(filename: str) -> dict[str, Any] | None:
    try:
        resource = importlib.resources.files("mdmp_core").joinpath(f"data/conformance/vectors/{filename}")
        if resource.is_file():
            payload = json.loads(resource.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    path = _repo_root() / "conformance" / "vectors" / filename
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    return None


def _default_contract_payload() -> dict[str, Any]:
    return {
        "schema": {
            "name": "mdmp_conformance_dataset",
            "version": "1.0",
            "industry": "health",
            "columns": [
                {"name": "timestamp", "type": "datetime", "required": True},
                {"name": "glucose", "type": "float", "bounds": [40, 400], "required": True},
            ],
        },
        "consent": {
            "ai_training_allowed": True,
            "jurisdiction": "GDPR",
            "anonymized": True,
            "consent_date": "2026-01-01T00:00:00Z",
            "expiry": "2028-01-01T00:00:00Z",
        },
    }


def _default_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-03-01T00:00:00Z",
                "2026-03-01T00:05:00Z",
                "2026-03-01T00:10:00Z",
            ],
            "glucose": [110.0, 112.0, 115.0],
        }
    )


def _check_contract_runner(workdir: Path) -> dict[str, Any]:
    contract = parse_contract(_default_contract_payload())
    result = ContractRunner(contract).run(_default_dataset())
    return {
        "name": "contract_runner",
        "passed": bool(result.is_compliant and result.consent_verified),
        "detail": f"grade={result.grade} score={result.compliance_score}",
        "grade": result.grade,
        "compliance_score": result.compliance_score,
    }


def _check_fingerprint_lifecycle(workdir: Path) -> dict[str, Any]:
    dataset = workdir / "conformance_data.csv"
    df = _default_dataset()
    df.to_csv(dataset, index=False)
    record = compute_fingerprint(dataset, expires_days=7)
    valid_check = check_fingerprint(record, data_path=dataset)
    df.loc[0, "glucose"] = 210.0
    df.to_csv(dataset, index=False)
    stale_check = check_fingerprint(record, data_path=dataset)
    passed = valid_check.get("status") == "valid" and stale_check.get("status") == "stale"
    return {
        "name": "fingerprint_lifecycle",
        "passed": passed,
        "detail": f"initial={valid_check.get('status')} changed={stale_check.get('status')}",
    }


def _check_signature_verification(workdir: Path) -> dict[str, Any]:
    keys_dir = workdir / "keys"
    keys = generate_keypair(output_dir=keys_dir)
    payload = {"grade": "research_grade", "dataset_fingerprint": "sha256:abc123"}
    signed = MDMPSigner(keys["private_key"]).sign_card(payload, expires_days=30)
    verified = MDMPVerifier(keys["public_key"]).verify(signed)
    return {
        "name": "signature_verification",
        "passed": bool(verified.get("valid")),
        "detail": f"valid={verified.get('valid')} key_id={verified.get('key_id')}",
    }


def _check_policy_engine(workdir: Path) -> dict[str, Any]:
    payload = {
        "effective_grade": "research_grade",
        "consent_verified": True,
        "staleness": {"status": "valid"},
        "expired": False,
        "consent_context": {"jurisdiction": "GDPR"},
    }
    policy = PolicySpec(
        min_grade="research_grade",
        require_consent_verified=True,
        require_not_stale=True,
        require_not_expired=True,
        allowed_jurisdictions=("GDPR",),
    )
    evaluation = evaluate_policy(payload, policy)
    return {
        "name": "policy_engine",
        "passed": bool(evaluation.get("passed")),
        "detail": f"failed_checks={evaluation.get('failed_checks', [])}",
    }


def _check_vector_fingerprints() -> dict[str, Any]:
    payload = _load_vector_file("fingerprint.json")
    if payload is None:
        return {
            "name": "vector_fingerprint",
            "passed": False,
            "detail": "missing conformance/vectors/fingerprint.json",
        }
    failures: list[str] = []
    vectors = payload.get("vectors", [])
    for vector in vectors:
        try:
            raw = bytes.fromhex(str(vector.get("input_bytes_hex", "")))
            expected = str(vector.get("expected_fingerprint", ""))
            actual = f"sha256:{hashlib.sha256(raw).hexdigest()}"
            if actual != expected:
                failures.append(f"{vector.get('id')}: expected {expected}, got {actual}")
        except Exception as exc:
            failures.append(f"{vector.get('id')}: exception={exc}")
    return {
        "name": "vector_fingerprint",
        "passed": len(failures) == 0,
        "detail": "all vectors passed" if not failures else "; ".join(failures),
        "total_vectors": len(vectors),
    }


def _check_vector_signing(workdir: Path) -> dict[str, Any]:
    payload = _load_vector_file("signing.json")
    if payload is None:
        return {
            "name": "vector_signing",
            "passed": False,
            "detail": "missing conformance/vectors/signing.json",
        }
    failures: list[str] = []
    vectors = payload.get("vectors", [])

    for vector in vectors:
        vid = str(vector.get("id", "unknown"))
        try:
            seed_hex = str(vector.get("private_seed_hex", ""))
            priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
            pub = priv.public_key()
            pub_raw = pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex()
            expected_pub = str(vector.get("public_key_raw_hex", "")).lower()
            if expected_pub and pub_raw != expected_pub:
                failures.append(f"{vid}: public key mismatch")

            unsigned_card = dict(vector.get("unsigned_card", {}))
            digest = hashlib.sha256(_canonical_bytes(unsigned_card)).digest()
            expected_digest = str(vector.get("expected_digest_hex", "")).lower().strip()
            if expected_digest and digest.hex() != expected_digest:
                failures.append(f"{vid}: digest mismatch")

            computed_sig_b64 = base64.b64encode(priv.sign(digest)).decode("utf-8")
            expected_sig = vector.get("expected_signature_base64")
            if isinstance(expected_sig, str) and expected_sig.strip() and computed_sig_b64 != expected_sig:
                failures.append(f"{vid}: signature mismatch")

            sig_for_verify = str(vector.get("signature_base64_from_vector", expected_sig or computed_sig_b64))
            signed_card = {
                **unsigned_card,
                "signature": sig_for_verify,
                "signature_algorithm": "ed25519+sha256",
                "signed_by": "vector-test",
                "key_id": "vector-key",
                "signed_at": "2026-01-01T00:00:00Z",
            }

            pem_path = workdir / f"{vid}_pub.pem"
            pem_path.write_bytes(
                pub.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            verify = MDMPVerifier(pem_path).verify(signed_card)
            expected_valid = bool(vector.get("expected_valid", True))
            if bool(verify.get("valid")) != expected_valid:
                failures.append(f"{vid}: expected valid={expected_valid}, got {verify.get('valid')}")

            expected_error = vector.get("expected_error_string")
            if expected_error and verify.get("error") != expected_error:
                failures.append(f"{vid}: expected error {expected_error}, got {verify.get('error')}")
        except Exception as exc:
            failures.append(f"{vid}: exception={exc}")

    return {
        "name": "vector_signing",
        "passed": len(failures) == 0,
        "detail": "all vectors passed" if not failures else "; ".join(failures),
        "total_vectors": len(vectors),
    }


def _grade_reference(*, score: float, is_compliant: bool, consent_policy_ok: bool, consent_window_ok: bool) -> tuple[str, str]:
    if not is_compliant:
        return "raw", "non_compliant_schema_or_bounds"
    if not consent_policy_ok:
        return "research_grade", "consent_policy_not_verified"
    if not consent_window_ok:
        return "research_grade", "consent_window_not_valid"
    if score < 75.0:
        return "draft", "low_validation_score"
    if score < 90.0:
        return "research_grade", "medium_validation_score"
    if score < 95.0:
        return "clinical_grade", "high_score_but_not_ai_ready_threshold"
    return "ai_ready", "all_checks_passed_with_valid_consent"


def _check_vector_grading() -> dict[str, Any]:
    payload = _load_vector_file("grading.json")
    if payload is None:
        return {
            "name": "vector_grading",
            "passed": False,
            "detail": "missing conformance/vectors/grading.json",
        }
    failures: list[str] = []
    vectors = payload.get("vectors", [])
    for vector in vectors:
        vid = str(vector.get("id", "unknown"))
        try:
            in_payload = dict(vector.get("input", {}))
            grade, reason = _grade_reference(
                score=float(in_payload.get("score", 0.0)),
                is_compliant=bool(in_payload.get("is_compliant", False)),
                consent_policy_ok=bool(in_payload.get("consent_policy_ok", False)),
                consent_window_ok=bool(in_payload.get("consent_window_ok", False)),
            )
            if grade != vector.get("expected_grade"):
                failures.append(f"{vid}: grade mismatch ({grade} != {vector.get('expected_grade')})")
            if reason != vector.get("expected_reason"):
                failures.append(f"{vid}: reason mismatch ({reason} != {vector.get('expected_reason')})")
        except Exception as exc:
            failures.append(f"{vid}: exception={exc}")
    return {
        "name": "vector_grading",
        "passed": len(failures) == 0,
        "detail": "all vectors passed" if not failures else "; ".join(failures),
        "total_vectors": len(vectors),
    }


def _check_vector_delegation(workdir: Path) -> dict[str, Any]:
    payload = _load_vector_file("delegation.json")
    if payload is None:
        return {
            "name": "vector_delegation",
            "passed": False,
            "detail": "missing conformance/vectors/delegation.json",
        }

    vectors = payload.get("vectors", [])
    vector_by_id = {str(v.get("id")): v for v in vectors}
    failures: list[str] = []

    for vector in vectors:
        vid = str(vector.get("id", "unknown"))
        try:
            source = vector
            if vector.get("base_vector"):
                source = vector_by_id[str(vector.get("base_vector"))]

            root_seed_hex = str(source.get("root_private_seed_hex", ""))
            delegate_seed_hex = str(source.get("delegate_private_seed_hex", ""))
            root_priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(root_seed_hex))
            delegate_priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(delegate_seed_hex))

            root_pub_raw = root_priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex()
            expected_root_pub = str(source.get("root_public_key_raw_hex", "")).lower()
            if expected_root_pub and root_pub_raw != expected_root_pub:
                failures.append(f"{vid}: root public key mismatch")

            delegate_pub_raw = delegate_priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex()
            expected_delegate_pub = str(source.get("delegate_public_key_raw_hex", "")).lower()
            if expected_delegate_pub and delegate_pub_raw != expected_delegate_pub:
                failures.append(f"{vid}: delegate public key mismatch")

            cert_unsigned = dict(source.get("delegate_certificate_unsigned", {}))
            cert_sig_b64 = str(source.get("delegate_certificate_signature_base64", ""))
            cert_sig_calc = base64.b64encode(root_priv.sign(payload_digest(cert_unsigned))).decode("utf-8")
            if cert_sig_b64 and cert_sig_calc != cert_sig_b64:
                failures.append(f"{vid}: delegate certificate signature mismatch")

            cert_signed = {
                **cert_unsigned,
                "signature": cert_sig_b64,
                "signature_algorithm": "ed25519+sha256",
                "signed_by": "MDMP-Authority-v1",
                "key_id": "mdmp_pub_v1",
                "signed_at": "2026-01-01T00:00:01Z",
            }

            card_unsigned = dict(source.get("delegate_card_unsigned", {}))
            if vector.get("tampered_fields"):
                card_unsigned.update(dict(vector.get("tampered_fields", {})))

            card_sig_b64 = str(source.get("delegate_card_signature_base64", ""))
            if not vector.get("base_vector"):
                card_sig_calc = base64.b64encode(delegate_priv.sign(payload_digest(card_unsigned))).decode("utf-8")
                if card_sig_b64 and card_sig_calc != card_sig_b64:
                    failures.append(f"{vid}: delegate card signature mismatch")

            card_signed = {
                **card_unsigned,
                "signature": card_sig_b64,
                "signature_algorithm": "ed25519+sha256",
                "signed_at": "2026-01-01T00:10:01Z",
            }

            root_pub_pem = root_priv.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            pub_path = workdir / f"{vid}_root_pub.pem"
            pub_path.write_bytes(root_pub_pem)

            verify = DelegateVerifier(pub_path).verify(card_signed, cert_signed)
            expected_valid = bool(vector.get("expected_valid", True))
            if bool(verify.get("valid")) != expected_valid:
                failures.append(f"{vid}: expected valid={expected_valid}, got {verify.get('valid')}")

            expected_error = vector.get("expected_error_string")
            if expected_error and verify.get("error") != expected_error:
                failures.append(f"{vid}: expected error {expected_error}, got {verify.get('error')}")
        except Exception as exc:
            failures.append(f"{vid}: exception={exc}")

    return {
        "name": "vector_delegation",
        "passed": len(failures) == 0,
        "detail": "all vectors passed" if not failures else "; ".join(failures),
        "total_vectors": len(vectors),
    }


def run_conformance_suite(workdir: str | Path) -> dict[str, Any]:
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    checks = [
        _check_contract_runner(root),
        _check_fingerprint_lifecycle(root),
        _check_signature_verification(root),
        _check_policy_engine(root),
        _check_vector_fingerprints(),
        _check_vector_signing(root),
        _check_vector_grading(),
        _check_vector_delegation(root),
    ]
    passed = sum(1 for c in checks if bool(c.get("passed")))
    failed = len(checks) - passed
    return {
        "suite": "mdmp_core_conformance",
        "version": "1",
        "executed_utc": _now_iso(),
        "passed": failed == 0,
        "summary": {"total": len(checks), "passed": passed, "failed": failed},
        "checks": checks,
    }


def write_conformance_report(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
