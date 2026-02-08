"""
IINTS-AF Data Module
Universal data ingestion and quality validation.
"""

from .adapter import DataAdapter
from .column_mapper import ColumnMapper, ColumnMapping
from .importer import (
    ImportResult,
    export_standard_csv,
    import_cgm_csv,
    scenario_from_csv,
    scenario_from_dataframe,
)
from .quality_checker import DataQualityChecker, QualityReport, DataGap, DataAnomaly
from .universal_parser import UniversalParser, StandardDataPack, ParseResult

__all__ = [
    'DataAdapter',
    'ColumnMapper',
    'ColumnMapping',
    'ImportResult',
    'export_standard_csv',
    'import_cgm_csv',
    'scenario_from_csv',
    'scenario_from_dataframe',
    'DataQualityChecker',
    'QualityReport',
    'DataGap',
    'DataAnomaly',
    'UniversalParser',
    'StandardDataPack',
    'ParseResult'
]
