#!/usr/bin/env python3
"""
Column Mapper - IINTS-AF
Maps various column names to standard IINTS format

Supports 50+ column name variations from different data sources.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
import pandas as pd # Required for type hints like pd.DataFrame


@dataclass
class ColumnMapping:
    """Result of column mapping operation"""
    mapped_columns: Dict[str, str]
    unmapped_columns: List[str]
    confidence: float
    warnings: List[str]


class ColumnMapper:
    """
    Maps various column name aliases to standard IINTS format.
    
    Standard Format: [timestamp, glucose, carbs, insulin]
    
    Supported sources:
    - Ohio T1DM: timestamp, glucose_mg_dl, carbs, insulin
    - OpenAPS/Nightscout: dateString, sg, carbs, insulin
    - Dexcom: timestamp, glucose, meal_carbs, bolus
    - CareLink: Glucose, Carbohydrates, Insulin
    - Custom CSV: Any variation
    """
    
    # Comprehensive column aliases for each standard field
    COLUMN_ALIASES = {
        'timestamp': [
            'timestamp', 'time', 'datetime', 'date', 'dateTime',
            'dateString', 'ts', 't', 'time_minutes', 'time_min',
            'minutes', 'elapsed_time', 'unix_time',
            'clock_time', 'reading_time', 'measurement_time',
            'created_at', 'recorded_at', 'sensor_timestamp'
        ],
        'glucose': [
            'glucose', 'bg', 'glucose_mg_dl', 'glucose_mgdl',
            'sg', 'sensor_glucose', 'cbg', 'blood_glucose',
            'glucose_value', 'glucose_reading', 'glucose_level',
            'glucose_concentration', 'glucose_mgdl', 'Glucose',
            'BG', 'blood_glucose_mg_dl', 'interstitial_glucose',
            'meter_glucose', 'sensor_glucose_mg_dl'
        ],
        'carbs': [
            'carbs', 'carbohydrates', 'cho', 'carb_intake',
            'meal_carbs', 'carbs_grams', 'carbohydrate_intake',
            'carbs_g', 'Carbs', 'carbs_consumed', 'food_carbs',
            'meal_carbohydrates', 'cho grams', 'Carbohydrates',
            'CHO', 'carb_input', 'carbs_input'
        ],
        'insulin': [
            'insulin', 'insulin_delivered', 'insulin_units',
            'bolus', 'total_insulin', 'insulin_dose', 'insulin_value',
            'bolus_insulin', 'basal_insulin', 'Insulin', 'insulin_total',
            'delivered_insulin', 'IOB', 'insulin_on_board', 'correction_bolus',
            'meal_bolus', 'Insulin (U)', 'insulin_units_u'
        ]
    }
    
    # Data source detection patterns
    SOURCE_PATTERNS = {
        'ohio_t1dm': {
            'columns': ['glucose_mg_dl', 'timestamp', 'carbs', 'insulin'],
            'delimiter': ','
        },
        'openaps_nightscout': {
            'columns': ['dateString', 'sg', 'carbs', 'insulin'],
            'delimiter': ','
        },
        'dexcom': {
            'columns': ['timestamp', 'glucose', 'meal_carbs', 'bolus'],
            'delimiter': ','
        },
        'carelink': {
            'columns': ['Glucose', 'Carbohydrates', 'Insulin'],
            'delimiter': ','
        },
        'tandem': {
            'columns': ['Date', 'Sensor Glucose', 'Carbs', 'Insulin'],
            'delimiter': ','
        }
    }
    
    def __init__(self):
        self.mappings_cache: Dict[str, str] = {}
        
    def detect_source(self, columns: List[str]) -> Optional[str]:
        """
        Detect the data source based on column names.
        
        Args:
            columns: List of column names from the data file
            
        Returns:
            Detected source name or None
        """
        columns_lower = [c.lower() for c in columns]
        
        for source, pattern in self.SOURCE_PATTERNS.items():
            matches = 0
            for expected_col in pattern['columns']:
                if any(expected_col.lower() in col for col in columns_lower):
                    matches += 1
            
            if matches >= len(pattern['columns']) * 0.75:  # 75% match threshold
                return source
        
        return None
    
    def normalize_column_name(self, column_name: str) -> str:
        """
        Normalize a column name to lowercase with underscores.
        
        Args:
            column_name: Original column name
            
        Returns:
            Normalized column name
        """
        # Check cache first
        if column_name in self.mappings_cache:
            return self.mappings_cache[column_name]
        
        # Normalize: lowercase, replace spaces/special chars with underscore
        normalized = re.sub(r'[\s\-/]+', '_', column_name.lower())
        normalized = re.sub(r'[()]', '', normalized)
        normalized = normalized.strip('_')
        
        self.mappings_cache[column_name] = normalized
        return normalized
    
    def find_standard_mapping(self, column_name: str) -> Optional[str]:
        """
        Find which standard field this column maps to.
        
        Args:
            column_name: Column name to map
            
        Returns:
            Standard field name or None
        """
        normalized = self.normalize_column_name(column_name)
        
        # Prioritize exact match with normalized aliases
        for standard_field, aliases in self.COLUMN_ALIASES.items():
            if normalized in aliases:
                return standard_field
        
        # Fuzzy match (original logic, but less priority)
        for standard_field, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in normalized or normalized in alias:
                    return standard_field
        
        # Try partial matching (even less priority)
        for standard_field, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                # Check for common patterns
                if 'glucose' in normalized or 'bg' in normalized:
                    if 'glucose' in alias or 'bg' in alias:
                        return standard_field
                if 'carb' in normalized:
                    if 'carb' in alias:
                        return standard_field
                if 'insulin' in normalized or 'bolus' in normalized:
                    if 'insulin' in alias or 'bolus' in alias:
                        return standard_field
                if 'time' in normalized or 'date' in normalized:
                    if 'time' in alias or 'date' in alias:
                        return standard_field
        
        return None
    
    def map_columns(self, columns: List[str]) -> ColumnMapping:
        """
        Map a list of columns to standard IINTS format.
        
        Args:
            columns: List of column names from data file
            
        Returns:
            ColumnMapping with results and confidence score
        """
        mapped_columns: Dict[str, str] = {}
        unmapped_columns: List[str] = []
        warnings: List[str] = []
        
        # Iterate and map, prioritizing earlier matches for a given standard field
        for column in columns:
            standard_field = self.find_standard_mapping(column)
            
            if standard_field:
                # Only map if not already mapped, or if it's a "better" match
                # For simplicity, we'll just take the first match for now.
                # More advanced logic could go here to determine "best" match
                if standard_field not in mapped_columns:
                    mapped_columns[standard_field] = column
                else:
                    warnings.append(
                        f"Skipping duplicate mapping for '{standard_field}': "
                        f"'{column}' (already mapped to '{mapped_columns[standard_field]}')"
                    )
            else:
                unmapped_columns.append(column)
        
        # Calculate confidence
        required_fields = ['timestamp', 'glucose']
        optional_fields = ['carbs', 'insulin']
        
        mapped_required = sum(1 for f in required_fields if f in mapped_columns)
        mapped_optional = sum(1 for f in optional_fields if f in mapped_columns)
        
        confidence = (
            (mapped_required / len(required_fields)) * 0.7 +
            (mapped_optional / len(optional_fields)) * 0.3
        )
        
        # Add warnings for missing optional fields
        for field in optional_fields:
            if field not in mapped_columns:
                warnings.append(f"Optional field '{field}' not found in data")
        
        # Warning for missing required fields
        for field in required_fields:
            if field not in mapped_columns:
                warnings.append(f"CRITICAL: Required field '{field}' not found!")
                confidence -= 0.2
        
        return ColumnMapping(
            mapped_columns=mapped_columns,
            unmapped_columns=unmapped_columns,
            confidence=max(0, confidence),
            warnings=warnings
        )
    
    def apply_mapping(self, df, mapping: ColumnMapping) -> 'pd.DataFrame':
        """
        Apply column mapping to a DataFrame, renaming columns to standard format.
        
        Args:
            df: Pandas DataFrame
            mapping: ColumnMapping result
            
        Returns:
            DataFrame with renamed columns
        """
        import pandas as pd
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Rename columns to standard names
        rename_dict = {v: k for k, v in mapping.mapped_columns.items()}
        result_df = result_df.rename(columns=rename_dict)
        
        # Add missing columns with NaN values
        for field in ['timestamp', 'glucose', 'carbs', 'insulin']:
            if field not in result_df.columns:
                result_df[field] = float('nan')
        
        # Ensure correct column order
        standard_order = ['timestamp', 'glucose', 'carbs', 'insulin']
        existing_cols = [c for c in standard_order if c in result_df.columns]
        other_cols = [c for c in result_df.columns if c not in standard_order]
        result_df = result_df[existing_cols + other_cols]
        
        return result_df
    
    def get_recommended_parser(self, source: str) -> str:
        """
        Get recommended parser configuration for a detected source.
        
        Args:
            source: Detected source name
            
        Returns:
            Parser configuration string
        """
        if source in self.SOURCE_PATTERNS:
            pattern = self.SOURCE_PATTERNS[source]
            return f"use_parser('{source}', delimiter='{pattern['delimiter']}')"
        
        return "use_parser('auto')"
    
    def get_source_info(self, source: str) -> Dict:
        """
        Get information about a detected data source.
        
        Args:
            source: Source name
            
        Returns:
            Dictionary with source information
        """
        if source in self.SOURCE_PATTERNS:
            return {
                'name': source,
                'columns': self.SOURCE_PATTERNS[source]['columns'],
                'delimiter': self.SOURCE_PATTERNS[source]['delimiter'],
                'known_format': True
            }
        
        return {
            'name': source,
            'columns': [],
            'delimiter': ',',
            'known_format': False
        }


def demo_column_mapping():
    """Demonstrate column mapping functionality"""
    print("=" * 70)
    print("COLUMN MAPPER DEMONSTRATION")
    print("=" * 70)
    
    mapper = ColumnMapper()
    
    # Test case 1: Ohio T1DM format
    print("\n Test Case 1: Ohio T1DM Format")
    print("-" * 50)
    ohio_columns = ['timestamp', 'glucose_mg_dl', 'carbs', 'insulin', 'heart_rate']
    mapping = mapper.map_columns(ohio_columns)
    
    print(f"Input columns: {ohio_columns}")
    print(f"Mapped: {mapping.mapped_columns}")
    print(f"Unmapped: {mapping.unmapped_columns}")
    print(f"Confidence: {mapping.confidence:.1%}")
    if mapping.warnings:
        print("Warnings:")
        for w in mapping.warnings:
            print(f"    {w}")
    
    # Test case 2: OpenAPS/Nightscout format
    print("\n Test Case 2: OpenAPS/Nightscout Format")
    print("-" * 50)
    openaps_columns = ['dateString', 'sg', 'carbs', 'insulin', 'iob', 'cob']
    mapping = mapper.map_columns(openaps_columns)
    
    print(f"Input columns: {openaps_columns}")
    print(f"Mapped: {mapping.mapped_columns}")
    print(f"Confidence: {mapping.confidence:.1%}")
    
    # Test case 3: Dexcom format
    print("\n Test Case 3: Dexcom Format")
    print("-" * 50)
    dexcom_columns = ['timestamp', 'glucose', 'meal_carbs', 'bolus', 'isig']
    mapping = mapper.map_columns(dexcom_columns)
    
    print(f"Input columns: {dexcom_columns}")
    print(f"Mapped: {mapping.mapped_columns}")
    print(f"Confidence: {mapping.confidence:.1%}")
    
    # Test case 4: Custom format with variations
    print("\n Test Case 4: Custom Format (Variations)")
    print("-" * 50)
    custom_columns = ['Time (min)', 'BG', 'CHO grams', 'Insulin (U)', 'Activity']
    mapping = mapper.map_columns(custom_columns)
    
    print(f"Input columns: {custom_columns}")
    print(f"Mapped: {mapping.mapped_columns}")
    print(f"Confidence: {mapping.confidence:.1%}")
    if mapping.warnings:
        print("Warnings:")
        for w in mapping.warnings:
            print(f"    {w}")
    
    # Test source detection
    print("\n Source Detection")
    print("-" * 50)
    for test_columns, expected_source in [
        (ohio_columns, 'ohio_t1dm'),
        (openaps_columns, 'openaps_nightscout'),
        (dexcom_columns, 'dexcom')
    ]:
        detected = mapper.detect_source(test_columns)
        status = "✓" if detected == expected_source else "✗"
        print(f"{status} Columns: {test_columns[:3]}... → Detected: {detected}")
    
    print("\n" + "=" * 70)
    print("COLUMN MAPPER DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_column_mapping()

