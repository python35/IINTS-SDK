from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

from iints.mdmp.backend import mdmp_grade_meets_minimum


def _load_mdmp_verifier() -> type[Any]:
    try:
        module = importlib.import_module("mdmp_core")
    except Exception as exc:
        raise ImportError(
            "MDMP verification requires the optional standalone package.\n"
            "Install with: pip install iints-sdk-python35[mdmp]\n"
            "or: pip install 'mdmp-protocol>=0.3.0'"
        ) from exc
    verifier_cls = getattr(module, "MDMPVerifier", None)
    if verifier_cls is None:
        raise ImportError("mdmp_core is installed but does not expose MDMPVerifier.")
    return verifier_cls


@dataclass(frozen=True)
class GuardResult:
    cert_path: str
    grade: str
    issued_by: str | None
    verification_mode: str | None
    key_id: str | None
    raw_result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cert_path": self.cert_path,
            "grade": self.grade,
            "issued_by": self.issued_by,
            "verification_mode": self.verification_mode,
            "key_id": self.key_id,
            "raw_result": self.raw_result,
        }


class MDMPGuard:
    """Enforce a valid MDMP-signed artifact before AI analysis can run."""

    DISCLAIMER = "\n\nWARNING: For research use only. Not medical advice."
    RESEARCH_ONLY_NOTICE = "For research use only. Not medical advice."

    def __init__(
        self,
        cert_path: str | Path,
        *,
        minimum_grade: str = "research_grade",
        public_key_path: str | Path | None = None,
        trust_store_path: str | Path | None = None,
    ) -> None:
        self.cert_path = Path(cert_path)
        self.minimum_grade = minimum_grade
        self.public_key_path = str(public_key_path) if public_key_path else None
        self.trust_store_path = str(trust_store_path) if trust_store_path else None
        if self.public_key_path and self.trust_store_path:
            raise ValueError("Use either public_key_path or trust_store_path, not both.")

    def _load_signed_artifact(self) -> dict[str, Any]:
        if not self.cert_path.is_file():
            raise FileNotFoundError(f"MDMP certificate not found: {self.cert_path}")
        try:
            payload = json.loads(self.cert_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"MDMP certificate must be valid JSON: {self.cert_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("MDMP certificate payload must be a JSON object.")
        return payload

    def check(self) -> GuardResult:
        signed_artifact = self._load_signed_artifact()
        verifier_cls = _load_mdmp_verifier()
        verifier = verifier_cls(
            public_key_path=self.public_key_path,
            trust_store_path=self.trust_store_path,
        )
        raw_result = verifier.verify(signed_artifact)
        if not bool(raw_result.get("valid", False)):
            error = raw_result.get("error", "verification_failed")
            raise PermissionError(
                f"MDMP verification failed: {error}. Cannot run AI analysis on uncertified data. "
                f"{self.RESEARCH_ONLY_NOTICE}"
            )

        raw_grade = raw_result.get("grade") or signed_artifact.get("grade") or signed_artifact.get("mdmp_grade") or "raw"
        grade = str(raw_grade).strip().lower()
        if not mdmp_grade_meets_minimum(grade, self.minimum_grade):
            raise PermissionError(
                f"MDMP grade '{grade}' does not meet required minimum '{self.minimum_grade}'. "
                f"Cannot run AI analysis on uncertified data. {self.RESEARCH_ONLY_NOTICE}"
            )

        return GuardResult(
            cert_path=str(self.cert_path),
            grade=grade,
            issued_by=raw_result.get("issued_by"),
            verification_mode=raw_result.get("verification_mode"),
            key_id=raw_result.get("key_id"),
            raw_result=raw_result,
        )

    def wrap(self, response: str) -> str:
        return response.rstrip() + self.DISCLAIMER
