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
]

