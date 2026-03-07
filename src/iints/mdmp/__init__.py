"""
MDMP public API (separated namespace).

This namespace is intentionally distinct from the broader iints SDK so MDMP
can evolve as its own protocol surface over time.
"""

from ..data.contracts import (
    StreamSpec,
    FeatureSpec,
    LabelSpec,
    ValidationSpec,
    ProcessSpec,
    ModelReadyContract,
    compile_contract,
    parse_contract,
    load_contract_yaml,
)
from ..data.runner import (
    ContractRunner,
    ValidationResult,
    CheckResult,
    MDMP_PROTOCOL_VERSION,
    MDMP_GRADE_ORDER,
    classify_mdmp_grade,
    mdmp_grade_meets_minimum,
    dataframe_fingerprint,
)
from ..data.guardians import mdmp_gate, MDMPGateError
from ..data.synthetic_mirror import generate_synthetic_mirror, SyntheticMirrorArtifact
from ..data.mdmp_visualizer import build_mdmp_dashboard_html
from .backend import (
    MDMPValidationResult,
    MDMPCheckResult,
    MDMP_GRADE_ORDER as BACKEND_MDMP_GRADE_ORDER,
    mdmp_grade_meets_minimum as backend_mdmp_grade_meets_minimum,
    active_mdmp_backend,
    load_mdmp_contract,
    run_mdmp_validation,
    build_mdmp_dashboard_html as build_mdmp_dashboard_html_with_backend,
)

__all__ = [
    "StreamSpec",
    "FeatureSpec",
    "LabelSpec",
    "ValidationSpec",
    "ProcessSpec",
    "ModelReadyContract",
    "compile_contract",
    "parse_contract",
    "load_contract_yaml",
    "ContractRunner",
    "ValidationResult",
    "CheckResult",
    "MDMP_PROTOCOL_VERSION",
    "MDMP_GRADE_ORDER",
    "classify_mdmp_grade",
    "mdmp_grade_meets_minimum",
    "dataframe_fingerprint",
    "mdmp_gate",
    "MDMPGateError",
    "generate_synthetic_mirror",
    "SyntheticMirrorArtifact",
    "build_mdmp_dashboard_html",
    "MDMPValidationResult",
    "MDMPCheckResult",
    "BACKEND_MDMP_GRADE_ORDER",
    "backend_mdmp_grade_meets_minimum",
    "active_mdmp_backend",
    "load_mdmp_contract",
    "run_mdmp_validation",
    "build_mdmp_dashboard_html_with_backend",
]
