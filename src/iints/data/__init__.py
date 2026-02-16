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
]
