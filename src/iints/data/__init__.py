"""
IINTS-AF Data Module
Universal data ingestion and quality validation.
"""

from .adapter import DataAdapter
from .column_mapper import ColumnMapper, ColumnMapping
from .importer import (
    ImportResult,
    export_demo_csv,
    export_standard_csv,
    guess_column_mapping,
    import_carelink_csv,
    import_carelink_timeline,
    import_cgm_csv,
    import_cgm_dataframe,
    load_carelink_event_log,
    load_demo_dataframe,
    scenario_from_csv,
    scenario_from_dataframe,
    summarize_carelink_csv,
)
from .quality_checker import DataQualityChecker, QualityReport, DataGap, DataAnomaly
from .realism_validator import (
    REALISM_VERDICT_ORDER,
    MealResponse,
    RealismCheck,
    RealismReport,
    realism_verdict_meets_minimum,
    validate_realism_csv,
    validate_realism_dataset,
    write_realism_report,
)
from .realism_governance import (
    RealDataGateProfile,
    RealDataGateResult,
    STRICT_REAL_DATA_RESEARCH_PROFILE,
    review_real_data_realism,
)
from .evidence import rank_real_data_sources
from .realism_reference import (
    ReferenceBand,
    ReferenceComparison,
    RealismReferenceProfile,
    get_realism_reference,
    list_realism_reference_ids,
    load_realism_reference_registry,
)
from .realism_dashboard import (
    build_realism_dashboard_html,
    write_realism_dashboard,
)
from .certify import (
    certify_csv,
    certify_dataset,
    render_certification_dashboard,
    write_certification_dashboard,
    write_certification_report,
)
from .study_corruption import (
    AVAILABLE_STUDY_CORRUPTIONS,
    apply_study_corruptions,
    write_corrupted_study_csv,
)
from .universal_parser import UniversalParser, StandardDataPack, ParseResult
from .registry import load_dataset_registry, get_dataset, list_dataset_ids, fetch_dataset
from .nightscout import NightscoutConfig, import_nightscout
from .tidepool import (
    TidepoolClient,
    TidepoolConfig,
    fetch_tidepool_dataframe,
    import_tidepool,
    load_openapi_spec,
)
from .medtronic_live import (
    MedtronicLiveClient,
    MedtronicLiveConfig,
    fetch_medtronic_live_dataframe,
    fetch_medtronic_live_timeline,
    import_medtronic_live,
    normalize_medtronic_live_payload,
)
from .contracts import (
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
from .runner import (
    ContractRunner,
    ValidationResult,
    CheckResult,
    MDMP_PROTOCOL_VERSION,
    MDMP_GRADE_ORDER,
    classify_mdmp_grade,
    mdmp_grade_meets_minimum,
    dataframe_fingerprint,
)
from .mdmp_visualizer import build_mdmp_dashboard_html
from .guardians import mdmp_gate, MDMPGateError
from .synthetic_mirror import generate_synthetic_mirror, SyntheticMirrorArtifact

__all__ = [
    'DataAdapter',
    'ColumnMapper',
    'ColumnMapping',
    'ImportResult',
    'export_demo_csv',
    'export_standard_csv',
    'guess_column_mapping',
    'import_carelink_csv',
    'import_carelink_timeline',
    'import_cgm_csv',
    'import_cgm_dataframe',
    'load_carelink_event_log',
    'load_demo_dataframe',
    'scenario_from_csv',
    'scenario_from_dataframe',
    'summarize_carelink_csv',
    'DataQualityChecker',
    'QualityReport',
    'DataGap',
    'DataAnomaly',
    'REALISM_VERDICT_ORDER',
    'MealResponse',
    'RealismCheck',
    'RealismReport',
    'realism_verdict_meets_minimum',
    'validate_realism_csv',
    'validate_realism_dataset',
    'write_realism_report',
    'RealDataGateProfile',
    'RealDataGateResult',
    'STRICT_REAL_DATA_RESEARCH_PROFILE',
    'review_real_data_realism',
    'rank_real_data_sources',
    'ReferenceBand',
    'ReferenceComparison',
    'RealismReferenceProfile',
    'get_realism_reference',
    'list_realism_reference_ids',
    'load_realism_reference_registry',
    'build_realism_dashboard_html',
    'write_realism_dashboard',
    'certify_csv',
    'certify_dataset',
    'render_certification_dashboard',
    'write_certification_dashboard',
    'write_certification_report',
    'AVAILABLE_STUDY_CORRUPTIONS',
    'apply_study_corruptions',
    'write_corrupted_study_csv',
    'UniversalParser',
    'StandardDataPack',
    'ParseResult',
    'load_dataset_registry',
    'get_dataset',
    'list_dataset_ids',
    'fetch_dataset',
    'NightscoutConfig',
    'import_nightscout',
    'TidepoolClient',
    'TidepoolConfig',
    'fetch_tidepool_dataframe',
    'import_tidepool',
    'load_openapi_spec',
    'MedtronicLiveClient',
    'MedtronicLiveConfig',
    'fetch_medtronic_live_dataframe',
    'fetch_medtronic_live_timeline',
    'import_medtronic_live',
    'normalize_medtronic_live_payload',
    'StreamSpec',
    'FeatureSpec',
    'LabelSpec',
    'ValidationSpec',
    'ProcessSpec',
    'ModelReadyContract',
    'compile_contract',
    'parse_contract',
    'load_contract_yaml',
    'ContractRunner',
    'ValidationResult',
    'CheckResult',
    'MDMP_PROTOCOL_VERSION',
    'MDMP_GRADE_ORDER',
    'classify_mdmp_grade',
    'mdmp_grade_meets_minimum',
    'dataframe_fingerprint',
    'build_mdmp_dashboard_html',
    'mdmp_gate',
    'MDMPGateError',
    'generate_synthetic_mirror',
    'SyntheticMirrorArtifact',
]
