from __future__ import annotations


class MDMPError(Exception):
    """Base exception for MDMP errors."""


class MDMPContractError(MDMPError):
    """Contract parsing or validation error."""


class MDMPFingerprintError(MDMPError):
    """Fingerprint generation or verification error."""


class MDMPSignatureError(MDMPError):
    """Cryptographic signing or verification error."""


class MDMPStalenessError(MDMPError):
    """Dataset staleness related error."""


class MDMPGradeError(MDMPError):
    """Grade assignment or authorization error."""


class MDMPMigrationError(MDMPError):
    """Specification migration error."""


class MDMPPolicyError(MDMPError):
    """Policy evaluation or parsing error."""
