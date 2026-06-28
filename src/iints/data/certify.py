from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
import importlib.resources

import pandas as pd

from iints.mdmp.backend import (
    MDMPValidationResult,
    build_mdmp_dashboard_html,
    load_mdmp_contract,
    run_mdmp_validation,
)


STANDARD_DIABETES_CONTRACT = "diabetes_cgm_mdmp_contract.yaml"


def standard_diabetes_contract_path() -> Path:
    """Return the packaged standard diabetes research contract path."""

    resource = importlib.resources.files("iints").joinpath("data").joinpath("contracts").joinpath(STANDARD_DIABETES_CONTRACT)
    return Path(str(resource))


def load_standard_diabetes_contract() -> Any:
    """Load the bundled CGM/insulin/carbs diabetes MDMP contract."""

    return load_mdmp_contract(standard_diabetes_contract_path())


def write_standard_diabetes_contract(output_path: str | Path) -> Path:
    """Write the bundled standard diabetes contract to a user-visible path."""

    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(standard_diabetes_contract_path().read_text(encoding="utf-8"), encoding="utf-8")
    return resolved


def certify_dataset(
    contract: Any | str | Path,
    dataframe: pd.DataFrame,
    *,
    apply_builtin_transforms: bool = True,
) -> MDMPValidationResult:
    """Run the bundled data-certification flow on a DataFrame."""
    resolved_contract = load_mdmp_contract(Path(contract)) if isinstance(contract, (str, Path)) else contract
    return run_mdmp_validation(
        resolved_contract,
        dataframe,
        apply_builtin_transforms=apply_builtin_transforms,
    )


def certify_csv(
    contract_path: str | Path,
    input_csv: str | Path,
    *,
    apply_builtin_transforms: bool = True,
    quick: bool = False,
    quick_rows: int = 5000,
) -> MDMPValidationResult:
    """Load a CSV and certify it against an IINTS data contract."""
    contract = load_mdmp_contract(Path(contract_path))
    dataframe = pd.read_csv(input_csv, nrows=max(1, int(quick_rows)) if quick else None)
    return certify_dataset(
        contract,
        dataframe,
        apply_builtin_transforms=apply_builtin_transforms,
    )


def certification_payload(
    report: Mapping[str, Any] | MDMPValidationResult,
    *,
    quick: bool = False,
    quick_rows: int | None = None,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready report payload with scan-mode metadata."""

    payload = report.to_dict() if isinstance(report, MDMPValidationResult) else dict(report)
    payload["scan_mode"] = "quick" if quick else "full"
    if quick:
        payload["quick_rows_limit"] = int(quick_rows or 0)
        payload["quick_mode_notice"] = (
            "Quick mode only scans the first rows loaded for this run. "
            "Use full mode before publishing final dataset evidence."
        )
        payload["full_dataset_required_for_final_certification"] = True
    else:
        payload["full_dataset_required_for_final_certification"] = False
    if input_path is not None:
        payload["input_path"] = str(input_path)
    return payload


def create_mdmp_certificate_payload(
    report: Mapping[str, Any] | MDMPValidationResult,
    *,
    issued_by: str = "IINTS-AF Local MDMP",
    certificate_id: str | None = None,
) -> dict[str, Any]:
    """Create an MDMP certificate payload from a certification report."""

    from mdmp_core.certification import create_certificate

    payload = report.to_dict() if isinstance(report, MDMPValidationResult) else dict(report)
    cert_input = {
        **payload,
        "grade": payload.get("mdmp_grade"),
        "effective_grade": payload.get("mdmp_grade"),
        "protocol_version": payload.get("mdmp_protocol_version"),
        "intended_use": "research_and_education_only",
    }
    certificate = create_certificate(
        cert_input,
        issued_by=issued_by,
        level=str(payload.get("mdmp_grade", "draft")),
        certificate_id=certificate_id,
    )
    certificate["signature_status"] = "unsigned_sha256_only"
    certificate["verification_note"] = (
        "This certificate has only a deterministic SHA-256 digest. "
        "Use signing_key to produce an Ed25519 signed certificate."
    )
    return certificate


def sign_mdmp_certificate_payload(
    certificate: Mapping[str, Any],
    *,
    signing_key: str | Path,
    signed_by: str = "IINTS-AF Local MDMP",
    key_id: str = "iints_local_mdmp_v1",
    passphrase: str | bytes | None = None,
) -> dict[str, Any]:
    """Sign a certificate with an Ed25519 MDMP key."""

    from mdmp_core.crypto import MDMPSigner

    payload = dict(certificate)
    payload.pop("signature_sha256", None)
    payload.pop("signature_status", None)
    payload["signature_status"] = "ed25519_signed"
    signer = MDMPSigner(
        private_key_path=signing_key,
        signed_by=signed_by,
        key_id=key_id,
        private_key_passphrase=passphrase,
    )
    return signer.sign_card(payload)


def write_mdmp_certificate(
    report: Mapping[str, Any] | MDMPValidationResult,
    output_path: str | Path,
    *,
    issued_by: str = "IINTS-AF Local MDMP",
    signing_key: str | Path | None = None,
    key_id: str = "iints_local_mdmp_v1",
    passphrase_env: str | None = None,
) -> Path:
    """Write an unsigned or Ed25519-signed MDMP certificate JSON."""

    cert = create_mdmp_certificate_payload(report, issued_by=issued_by)
    if signing_key is not None:
        passphrase = os.getenv(passphrase_env) if passphrase_env else None
        cert = sign_mdmp_certificate_payload(
            cert,
            signing_key=signing_key,
            signed_by=issued_by,
            key_id=key_id,
            passphrase=passphrase,
        )
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    return resolved


def render_certification_dashboard(
    report: Mapping[str, Any] | MDMPValidationResult,
    *,
    title: str = "IINTS Data Certification Dashboard",
) -> str:
    """Render a single-file HTML dashboard from a certification report."""
    payload = report.to_dict() if isinstance(report, MDMPValidationResult) else dict(report)
    return build_mdmp_dashboard_html(payload, title=title)


def write_certification_report(
    report: Mapping[str, Any] | MDMPValidationResult,
    output_path: str | Path,
) -> Path:
    """Write a certification report JSON file."""
    resolved = Path(output_path)
    payload = report.to_dict() if isinstance(report, MDMPValidationResult) else dict(report)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2))
    return resolved


def write_certification_dashboard(
    report: Mapping[str, Any] | MDMPValidationResult,
    output_path: str | Path,
    *,
    title: str = "IINTS Data Certification Dashboard",
) -> Path:
    """Write a certification dashboard HTML file."""
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(render_certification_dashboard(report, title=title))
    return resolved
