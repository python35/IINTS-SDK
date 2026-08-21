from __future__ import annotations

import base64
import json
from pathlib import Path
import pytest

from mdmp_core.crypto import (
    MDMPSigner,
    MDMPVerifier,
    derive_key_hkdf,
    derive_key_scrypt,
    encrypt_patient_payload,
    decrypt_patient_payload,
    encrypt_cgm_dataset_file,
    decrypt_cgm_dataset_file,
    generate_keypair,
    load_private_key,
    payload_digest,
)
from mdmp_core.exceptions import MDMPSignatureError


def test_derive_key_hkdf_produces_32_byte_key() -> None:
    key1 = derive_key_hkdf("secret-passphrase-123")
    key2 = derive_key_hkdf("secret-passphrase-123")
    key3 = derive_key_hkdf("different-passphrase")

    assert len(key1) == 32
    assert key1 == key2
    assert key1 != key3


def test_scrypt_uses_explicit_random_salt() -> None:
    salt_a = b"a" * 16
    salt_b = b"b" * 16
    key_a = derive_key_scrypt("correct horse battery staple", salt=salt_a)
    key_b = derive_key_scrypt("correct horse battery staple", salt=salt_b)

    assert len(key_a) == 32
    assert key_a != key_b


def test_new_envelopes_reject_short_passphrases() -> None:
    with pytest.raises(MDMPSignatureError, match="at least 12"):
        encrypt_patient_payload({"glucose": 120.0}, "too-short")


def test_encrypt_decrypt_patient_payload_roundtrip() -> None:
    patient_data = {
        "patient_id": "PT-9042",
        "mean_glucose": 142.5,
        "tir_70_180": 81.2,
        "cgm_stream": [110.0, 115.5, 122.0, 130.4, 128.1],
    }
    key = "biomedical-secure-key"
    aad = "study-protocol-diabetes-2026"

    encrypted = encrypt_patient_payload(patient_data, key, associated_data=aad)
    assert encrypted["cipher"] == "chacha20-poly1305"
    assert encrypted["format_version"] == 2
    assert encrypted["kdf"]["name"] == "scrypt"
    assert "nonce" in encrypted
    assert "ciphertext" in encrypted

    # Decrypt with correct key and AAD
    decrypted = decrypt_patient_payload(encrypted, key, associated_data=aad)
    assert decrypted == patient_data


def test_tampered_ciphertext_fails_decryption() -> None:
    patient_data = {"glucose": 120.0}
    key = "passphrase-key"
    encrypted = encrypt_patient_payload(patient_data, key)

    # Tamper with the ciphertext
    raw_ct = bytearray(base64.b64decode(encrypted["ciphertext"]))
    raw_ct[5] ^= 0x01
    tampered_encrypted = dict(encrypted)
    tampered_encrypted["ciphertext"] = base64.b64encode(raw_ct).decode("ascii")

    with pytest.raises(MDMPSignatureError, match="Decryption failed"):
        decrypt_patient_payload(tampered_encrypted, key)


def test_mismatched_associated_data_fails_decryption() -> None:
    patient_data = {"glucose": 120.0}
    key = "passphrase-key"
    encrypted = encrypt_patient_payload(patient_data, key, associated_data="patient-A")

    # Attempt decrypt with wrong patient ID in AAD
    with pytest.raises(MDMPSignatureError, match="Decryption failed"):
        decrypt_patient_payload(encrypted, key, associated_data="patient-B")


def test_v2_envelope_rejects_kdf_metadata_downgrade() -> None:
    encrypted = encrypt_patient_payload({"glucose": 120.0}, "passphrase-key")
    encrypted.pop("kdf")

    with pytest.raises(MDMPSignatureError, match="missing KDF metadata"):
        decrypt_patient_payload(encrypted, "passphrase-key")


def test_file_level_encryption_decryption(tmp_path: Path) -> None:
    cgm_csv = tmp_path / "patient_cgm.csv"
    cgm_csv.write_text("timestamp,glucose,insulin\n0,120.0,0.1\n5,125.0,0.2\n", encoding="utf-8")

    enc_file = tmp_path / "patient_cgm.enc"
    dec_file = tmp_path / "patient_cgm_restored.csv"
    key = "secure-file-key"

    enc_meta = encrypt_cgm_dataset_file(cgm_csv, enc_file, key, associated_data="cgm-pack-v1")
    assert enc_file.exists()
    assert enc_meta["original_sha256"]

    dec_meta = decrypt_cgm_dataset_file(enc_file, dec_file, key, associated_data="cgm-pack-v1")
    assert dec_file.exists()
    assert dec_meta["decrypted_sha256"] == enc_meta["original_sha256"]
    assert dec_file.read_text(encoding="utf-8") == cgm_csv.read_text(encoding="utf-8")


def test_unsupported_post_quantum_label_is_rejected(tmp_path: Path) -> None:
    keypair = generate_keypair(output_dir=tmp_path / "keys")
    signer = MDMPSigner(
        private_key_path=keypair["private_key"],
        signed_by="IINTS-Research-Authority",
        key_id="custom_key_v1",
    )
    verifier = MDMPVerifier(public_key_path=keypair["public_key"])

    passport_card = {
        "dataset_name": "T1D-Empirical-Cohort",
        "sample_count": 1440,
        "mean_glucose": 141.2,
        "grade": "ai_ready",
    }

    with pytest.raises(MDMPSignatureError, match="Unsupported signature algorithm"):
        signer.sign_card(passport_card, algorithm="ed25519+mldsa44_hybrid")

    falsely_labelled = signer.sign_card(passport_card)
    falsely_labelled["signature_algorithm"] = "ed25519+mldsa44_hybrid"
    verification = verifier.verify(falsely_labelled)
    assert verification["valid"] is False
    assert verification["error"] == "unsupported_signature_algorithm"


def test_classic_ed25519_remains_fully_supported(tmp_path: Path) -> None:
    keypair = generate_keypair(output_dir=tmp_path / "keys_classic")
    signer = MDMPSigner(
        private_key_path=keypair["private_key"],
        signed_by="IINTS-Classic-Authority",
        key_id="classic_key_v1",
    )
    verifier = MDMPVerifier(public_key_path=keypair["public_key"])

    card = {"model_name": "DigitalTwin-BergmanODE", "grade": "clinical_grade"}
    signed = signer.sign_card(card, algorithm="ed25519+sha256")
    assert signed["signature_algorithm"] == "ed25519+sha256"
    assert signed["signature_format_version"] == 2

    res = verifier.verify(signed)
    assert res["valid"] is True
    assert res["signature_valid"] is True
    assert res["pq_verified"] is False


@pytest.mark.parametrize("field", ["signed_by", "key_id", "signed_at"])
def test_signature_v2_authenticates_signature_metadata(tmp_path: Path, field: str) -> None:
    keypair = generate_keypair(output_dir=tmp_path / "keys_v2")
    signed = MDMPSigner(
        private_key_path=keypair["private_key"],
        signed_by="Research Authority",
        key_id="research_key_v2",
    ).sign_card({"dataset_name": "CGM cohort", "grade": "research_grade"})

    tampered = dict(signed)
    tampered[field] = f"altered-{field}"
    result = MDMPVerifier(public_key_path=keypair["public_key"]).verify(tampered)

    assert result["valid"] is False
    assert result["signature_valid"] is False


def test_signature_v1_cards_remain_readable(tmp_path: Path) -> None:
    keypair = generate_keypair(output_dir=tmp_path / "keys_v1")
    payload = {"dataset_name": "legacy cohort", "grade": "research_grade"}
    signature = load_private_key(keypair["private_key"]).sign(payload_digest(payload))
    legacy_card = {
        **payload,
        "signature": base64.b64encode(signature).decode("ascii"),
        "signature_algorithm": "ed25519+sha256",
        "signed_by": "Legacy Research Authority",
        "key_id": "legacy_key_v1",
        "signed_at": "2025-01-01T00:00:00Z",
    }

    result = MDMPVerifier(public_key_path=keypair["public_key"]).verify(legacy_card)

    assert result["valid"] is True
    assert result["signature_format_version"] == 1
