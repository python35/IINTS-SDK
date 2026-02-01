"""
IINTS-AF Data Module
Universal data ingestion and quality validation.
"""

from .adapter import DataAdapter
from .column_mapper import ColumnMapper, ColumnMapping
from .quality_checker import DataQualityChecker, QualityReport, DataGap, DataAnomaly
from .universal_parser import UniversalParser, StandardDataPack, ParseResult

__all__ = [
    'DataAdapter',
    'ColumnMapper',
    'ColumnMapping',
    'DataQualityChecker',
    'QualityReport',
    'DataGap',
    'DataAnomaly',
    'UniversalParser',
    'StandardDataPack',
    'ParseResult'
]

