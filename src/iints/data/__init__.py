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
    import_cgm_csv,
    import_cgm_dataframe,
    load_demo_dataframe,
    scenario_from_csv,
    scenario_from_dataframe,
)
from .quality_checker import DataQualityChecker, QualityReport, DataGap, DataAnomaly
from .universal_parser import UniversalParser, StandardDataPack, ParseResult
from .registry import load_dataset_registry, get_dataset, list_dataset_ids, fetch_dataset
from .nightscout import NightscoutConfig, import_nightscout
from .tidepool import TidepoolClient, load_openapi_spec
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

__all__ = [
    'DataAdapter',
    'ColumnMapper',
    'ColumnMapping',
    'ImportResult',
    'export_demo_csv',
    'export_standard_csv',
    'guess_column_mapping',
    'import_cgm_csv',
    'import_cgm_dataframe',
    'load_demo_dataframe',
    'scenario_from_csv',
    'scenario_from_dataframe',
    'DataQualityChecker',
    'QualityReport',
    'DataGap',
    'DataAnomaly',
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
    'load_openapi_spec',
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
]
