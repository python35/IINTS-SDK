from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from iints.data.certify import (
    certification_payload,
    certify_dataset,
    load_standard_diabetes_contract,
    write_mdmp_certificate,
)


@dataclass(frozen=True)
class DesktopMDMPCertificateResult:
    """Paths and summary for a desktop-generated MDMP certificate."""

    certificate_path: Path
    report_path: Path
    public_key_path: Path
    private_key_path: Path
    grade: str
    compliance_score: float
    row_count: int


def create_desktop_mdmp_certificate(
    csv_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    quick: bool = True,
    quick_rows: int = 5000,
) -> DesktopMDMPCertificateResult:
    """Certify a loaded desktop CSV with the standard diabetes MDMP contract."""

    source = Path(csv_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Results CSV not found: {source}")

    certificate_dir = Path(output_dir).expanduser().resolve() if output_dir else source.parent / "mdmp_certificates"
    certificate_dir.mkdir(parents=True, exist_ok=True)
    key_dir = certificate_dir / "keys"
    private_key = key_dir / "iints_desktop_mdmp_private.pem"
    public_key = key_dir / "iints_desktop_mdmp_public.pem"
    if not private_key.exists() or not public_key.exists():
        from mdmp_core.crypto import generate_keypair

        generated = generate_keypair(
            output_dir=key_dir,
            private_name=private_key.name,
            public_name=public_key.name,
        )
        private_key = Path(str(generated["private_key"]))
        public_key = Path(str(generated["public_key"]))

    dataframe = pd.read_csv(source, nrows=max(1, int(quick_rows)) if quick else None)
    report = certify_dataset(
        load_standard_diabetes_contract(),
        dataframe,
        apply_builtin_transforms=True,
    )
    payload = certification_payload(
        report,
        quick=quick,
        quick_rows=max(1, int(quick_rows)) if quick else None,
        input_path=source,
    )

    stem = source.stem
    report_path = certificate_dir / f"{stem}_mdmp_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    certificate_path = certificate_dir / f"{stem}_mdmp_certificate.json"
    write_mdmp_certificate(
        payload,
        certificate_path,
        issued_by="IINTS-AF Desktop Local MDMP",
        signing_key=private_key,
        key_id="iints_desktop_mdmp_local_v1",
    )

    return DesktopMDMPCertificateResult(
        certificate_path=certificate_path,
        report_path=report_path,
        public_key_path=public_key,
        private_key_path=private_key,
        grade=report.mdmp_grade,
        compliance_score=report.compliance_score,
        row_count=report.row_count,
    )
