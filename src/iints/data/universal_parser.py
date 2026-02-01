#!/usr/bin/env python3
"""
Universal Data Parser - IINTS-AF
Universal ingestion engine for any CSV/JSON data format.

This is the "Universal Data Bridge" - it accepts any data format and
converts it to the standard IINTS format: [Time, Glucose, Carbs, Insulin]
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime

from .column_mapper import ColumnMapper, ColumnMapping
from .quality_checker import DataQualityChecker, QualityReport


@dataclass
class StandardDataPack:
    """
    Standard data format for IINTS-AF.
    
    All data is converted to this format before simulation.
    """
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_report: Optional[QualityReport] = None
    source_file: Optional[str] = None
    data_format: str = "standard"
    
    def __post_init__(self):
        """Ensure standard columns exist"""
        required_cols = ['timestamp', 'glucose']
        optional_cols = ['carbs', 'insulin']
        
        for col in required_cols:
            if col not in self.data.columns:
                self.data[col] = float('nan')
        
        for col in optional_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0
    
    @property
    def duration_hours(self) -> float:
        """Get data duration in hours"""
        if 'timestamp' not in self.data.columns or len(self.data) < 2:
            return 0.0
        return (self.data['timestamp'].max() - self.data['timestamp'].min()) / 60.0
    
    @property
    def data_points(self) -> int:
        """Get number of data points"""
        return len(self.data)
    
    @property
    def confidence_score(self) -> float:
        """Get simulation confidence score"""
        if self.quality_report:
            return self.quality_report.overall_score
        return 0.85  # Default if no quality report


@dataclass
class ParseResult:
    """Result of a parse operation"""
    success: bool
    data_pack: Optional[StandardDataPack]
    errors: List[str]
    warnings: List[str]
    parse_time_seconds: float
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'data_pack': {
                'data_points': self.data_pack.data_points if self.data_pack else 0,
                'duration_hours': self.data_pack.duration_hours if self.data_pack else 0,
                'confidence_score': self.data_pack.confidence_score if self.data_pack else 0
            } if self.data_pack else None,
            'errors': self.errors,
            'warnings': self.warnings,
            'parse_time_seconds': self.parse_time_seconds
        }


class UniversalParser:
    """
    Universal data parser for IINTS-AF.
    
    Accepts any CSV or JSON file and converts it to standard format.
    Handles:
    - Automatic column detection and mapping
    - Multiple date/time formats
    - Various glucose unit conversions
    - Data quality assessment
    
    Usage:
        parser = UniversalParser()
        result = parser.parse("patient_data.csv")
        if result.success:
            data = result.data_pack
            # Use data for simulation
    """
    
    # Supported date formats
    DATE_FORMATS = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
        '%m/%d/%Y',
        '%H:%M:%S',
        '%H:%M',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        None,  # Let pandas infer
    ]
    
    # Glucose unit conversions (to mg/dL)
    GLUCOSE_CONVERSIONS = {
        'mg/dl': 1.0,
        'mg/dL': 1.0,
        'mmol/l': 18.0182,
        'mmol/L': 18.0182,
    }
    
    def __init__(self, 
                 auto_validate: bool = True,
                 expected_interval: int = 5):
        """
        Initialize universal parser.
        
        Args:
            auto_validate: Whether to automatically run quality checks
            expected_interval: Expected time between readings in minutes
        """
        self.column_mapper = ColumnMapper()
        self.quality_checker = DataQualityChecker(
            expected_interval=expected_interval,
            source_type='cgm'
        )
        self.auto_validate = auto_validate
        self.expected_interval = expected_interval
        
    def detect_format(self, file_path: str) -> str:
        """
        Detect file format from extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected format ('csv', 'json', 'parquet', 'unknown')
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return 'csv'
        elif suffix == '.json':
            return 'json'
        elif suffix == '.parquet':
            return 'parquet'
        else:
            return 'unknown'
    
    def detect_delimiter(self, file_path: str) -> Optional[str]:
        """
        Detect CSV delimiter by analyzing the file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Detected delimiter or None
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        # Common delimiters to check
        delimiters = [',', ';', '\t', '|']
        detected = None
        max_count = 0
        
        for delimiter in delimiters:
            count = first_line.count(delimiter)
            if count > max_count:
                max_count = count
                detected = delimiter
        
        return detected
    
    def parse_datetime(self, value: Any) -> Optional[float]:
        """
        Parse datetime value to minutes from start.
        
        Args:
            value: Datetime value to parse
            
        Returns:
            Minutes from start or None if parsing fails
        """
        if pd.isna(value):
            return None
        
        # If already numeric, assume minutes
        if isinstance(value, (int, float)):
            return float(value)
        
        # If string, try to parse
        if isinstance(value, str):
            value = value.strip()
            
            try:
                # Use pandas for robust datetime parsing
                dt = pd.to_datetime(value)
                # Return minutes from midnight. The normalize_timestamps function
                # will convert this to minutes from the start of the series.
                return dt.hour * 60 + dt.minute + dt.second / 60 + dt.microsecond / 1_000_000 / 60
            except ValueError:
                # If parsing as a date fails, it might be a time-only format
                # that pandas couldn't infer.
                try:
                    parts = value.split(':')
                    if len(parts) >= 2:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2]) if len(parts) > 2 else 0.0
                        if 0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60:
                             return hours * 60 + minutes + seconds / 60
                except (ValueError, IndexError):
                    return None  # Could not parse as time

        return None
    
    def parse_glucose(self, value: Any, unit: str = 'mg/dL') -> Optional[float]:
        """
        Parse glucose value and convert to mg/dL.
        
        Args:
            value: Glucose value to parse
            unit: Unit of the value
            
        Returns:
            Glucose in mg/dL or None if parsing fails
        """
        if pd.isna(value):
            return None
        
        try:
            glucose_mgdl = float(value)
            
            # Convert if necessary
            if unit.lower() in ['mmol/l', 'mmol/l']:
                glucose_mgdl *= 18.0182  # Convert mmol/L to mg/dL
            
            return glucose_mgdl
        except (ValueError, TypeError):
            return None
    
    def parse_csv(self, file_path: str) -> pd.DataFrame:
        """
        Parse CSV file with automatic delimiter detection.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Parsed DataFrame
        """
        # Detect delimiter
        delimiter = self.detect_delimiter(file_path)
        
        # Read CSV
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            na_values=['', 'NA', 'N/A', 'null', 'NULL', 'NaN', 'nan'],
            keep_default_na=True
        )
        
        return df
    
    def parse_json(self, file_path: str) -> pd.DataFrame:
        """
        Parse JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed DataFrame
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            # Check for common nested structures
            if 'data' in data and isinstance(data['data'], list):
                df = pd.json_normalize(data['data'])
            elif 'readings' in data and isinstance(data['readings'], list):
                df = pd.json_normalize(data['readings'])
            elif 'entries' in data and isinstance(data['entries'], list):
                df = pd.json_normalize(data['entries'])
            else:
                df = pd.json_normalize(data)
        else:
            raise ValueError(f"Unexpected JSON structure in {file_path}")
        
        return df
    
    def convert_to_standard(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Convert DataFrame to standard IINTS format.
        
        Args:
            df: Input DataFrame with original columns
            
        Returns:
            DataFrame with standard columns [timestamp, glucose, carbs, insulin]
        """
        result_df = df.copy()
        
        # Map columns to standard format
        mapping = self.column_mapper.map_columns(list(result_df.columns))
        
        if mapping.mapped_columns:
            # Rename columns to standard names
            rename_dict = {v: k for k, v in mapping.mapped_columns.items()}
            result_df = result_df.rename(columns=rename_dict)
        
        # Parse timestamp column
        if 'timestamp' in result_df.columns:
            result_df['timestamp'] = result_df['timestamp'].apply(self.parse_datetime)
        
        # Ensure glucose is numeric
        if 'glucose' in result_df.columns:
            result_df['glucose'] = pd.to_numeric(result_df['glucose'], errors='coerce')
        
        # Ensure carbs and insulin are numeric
        if 'carbs' in result_df.columns:
            result_df['carbs'] = pd.to_numeric(result_df['carbs'], errors='coerce').fillna(0)
        else:
            result_df['carbs'] = 0.0
            
        if 'insulin' in result_df.columns:
            result_df['insulin'] = pd.to_numeric(result_df['insulin'], errors='coerce').fillna(0)
        else:
            result_df['insulin'] = 0.0
        
        # Select and order standard columns
        standard_cols = ['timestamp', 'glucose', 'carbs', 'insulin']
        # Fix: Exclude columns that have been mapped
        mapped_original_cols = list(mapping.mapped_columns.values())
        other_cols = [c for c in df.columns if c not in mapped_original_cols and c not in standard_cols]
        
        # We only want to keep the standard columns in the final dataframe
        result_df = result_df[standard_cols]
        
        return result_df, mapping.mapped_columns
    
    def normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize timestamps to minutes from start.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with normalized timestamps
        """
        if 'timestamp' not in df.columns:
            return df
        
        # If timestamps are already in minutes, ensure they're numeric
        if df['timestamp'].max() < 1440:  # Less than 24 hours in minutes
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            return df
        
        # Otherwise, convert to minutes from start
        try:
            # Convert to datetime first
            if df['timestamp'].dtype == object:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Calculate minutes from start
            start_time = df['timestamp'].min()
            df['timestamp'] = (df['timestamp'] - start_time).dt.total_seconds() / 60
            
            return df
        except Exception:
            return df
    
    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
        """
        Validate data and return quality report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (cleaned DataFrame, QualityReport)
        """
        report = self.quality_checker.check(df)
        return df, report
    
    def _clean_data_based_on_report(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        """
        Cleans the DataFrame based on the QualityReport by setting anomalous glucose values to NaN.
        """
        cleaned_df = df.copy()

        for anomaly in report.anomalies:
            if anomaly.anomaly_type in ['impossible_value', 'outlier', 'rapid_change']:
                # Set the anomalous glucose value to NaN
                # Ensure 'glucose' column exists and index is valid
                if 'glucose' in cleaned_df.columns and anomaly.index in cleaned_df.index:
                    cleaned_df.loc[anomaly.index, 'glucose'] = np.nan
        
        return cleaned_df

    def parse(self, 
              file_path: str,
              validate: Optional[bool] = None,
              metadata: Optional[Dict] = None) -> ParseResult:
        """
        Main entry point for parsing data files.
        
        Args:
            file_path: Path to the data file
            validate: Override auto_validate setting
            metadata: Optional metadata to add to the data pack
            
        Returns:
            ParseResult with data pack or errors
        """
        import time
        start_time = time.time()
        
        errors: List[str] = []
        warnings: List[str] = []
        
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            return ParseResult(
                success=False,
                data_pack=None,
                errors=[f"File not found: {file_path}"],
                warnings=[],
                parse_time_seconds=time.time() - start_time
            )
        
        # Detect format
        file_format = self.detect_format(file_path)
        if file_format == 'unknown':
            return ParseResult(
                success=False,
                data_pack=None,
                errors=[f"Unsupported file format: {path.suffix}"],
                warnings=[],
                parse_time_seconds=time.time() - start_time
            )
        
        try:
            # Parse based on format
            if file_format == 'csv':
                df = self.parse_csv(file_path)
            elif file_format == 'json':
                df = self.parse_json(file_path)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            # Log column mapping info
            mapping = self.column_mapper.map_columns(list(df.columns))
            if mapping.warnings:
                warnings.extend(mapping.warnings)
            
            if mapping.confidence < 0.5:
                warnings.append(f"Low column mapping confidence: {mapping.confidence:.1%}")
            
            # Convert to standard format
            df, column_mapping = self.convert_to_standard(df)
            
            # Check for required columns
            if 'glucose' not in df.columns or df['glucose'].isna().all():
                return ParseResult(
                    success=False,
                    data_pack=None,
                    errors=["No valid glucose data found in file"],
                    warnings=warnings,
                    parse_time_seconds=time.time() - start_time
                )
            
            # Normalize timestamps
            df = self.normalize_timestamps(df)
            
            # Create data pack
            data_pack = StandardDataPack(
                data=df,
                metadata=metadata or {},
                source_file=str(path.absolute())
            )
            
            # Validate and check quality
            if validate if validate is not None else self.auto_validate:
                original_df = df.copy() # Keep original for comparison if needed
                df, quality_report = self.validate_and_clean(original_df)
                
                # Clean data based on the quality report
                cleaned_df = self._clean_data_based_on_report(df, quality_report)
                data_pack.data = cleaned_df # Store the cleaned DataFrame
                data_pack.quality_report = quality_report
                
                if quality_report.overall_score < 0.5:
                    warnings.append(
                        f"Data quality is low ({quality_report.overall_score:.1%}). "
                        f"Simulation results may be unreliable."
                    )
                
                # Add quality warnings
                warnings.extend(quality_report.warnings)
            
            # Add metadata
            data_pack.metadata.update({
                'source_file': str(path.absolute()),
                'source_format': file_format,
                'column_mapping': column_mapping,
                'data_points': len(df),
                'duration_hours': data_pack.duration_hours
            })
            
            return ParseResult(
                success=True,
                data_pack=data_pack,
                errors=errors,
                warnings=warnings,
                parse_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                data_pack=None,
                errors=[f"Parse error: {str(e)}"],
                warnings=warnings,
                parse_time_seconds=time.time() - start_time
            )
    
    def parse_string(self,
                     content: str,
                     format_type: str = 'csv',
                     validate: bool = True) -> ParseResult:
        """
        Parse data from a string instead of a file.
        
        Args:
            content: String containing data
            format_type: Format of the data ('csv' or 'json')
            validate: Whether to run quality checks
            
        Returns:
            ParseResult with data pack or errors
        """
        import io
        import time
        start_time = time.time()
        
        errors: List[str] = []
        warnings: List[str] = []
        
        try:
            # Parse from string
            if format_type == 'csv':
                df = pd.read_csv(io.StringIO(content))
            elif format_type == 'json':
                data = json.loads(content)
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                else:
                    df = pd.json_normalize(data)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Convert to standard format
            df, column_mapping = self.convert_to_standard(df)
            
            # Check for required columns before further processing
            if 'glucose' not in df.columns or df['glucose'].isna().all():
                return ParseResult(
                    success=False,
                    data_pack=None,
                    errors=["No valid glucose data found in input string"],
                    warnings=warnings,
                    parse_time_seconds=time.time() - start_time
                )
            
            # Normalize timestamps
            df = self.normalize_timestamps(df)

            # Create data pack
            data_pack = StandardDataPack(
                data=df,
                metadata={'source': 'string_input', 'column_mapping': column_mapping},
                source_file=None
            )
            
            # Validate if requested
            if validate:
                # Keep original for comparison if needed for cleaning
                original_df_for_quality_check = df.copy() 
                _, quality_report = self.validate_and_clean(original_df_for_quality_check)
                
                # Clean data based on the quality report
                cleaned_df = self._clean_data_based_on_report(df.copy(), quality_report) # Pass a copy to avoid modifying df in place if it's used elsewhere
                data_pack.data = cleaned_df
                data_pack.quality_report = quality_report
                warnings.extend(quality_report.warnings)

                if quality_report.overall_score < 0.5:
                    warnings.append(
                        f"Data quality is low ({quality_report.overall_score:.1%}). "
                        f"Simulation results may be unreliable."
                    )
            
            return ParseResult(
                success=True,
                data_pack=data_pack,
                errors=errors,
                warnings=warnings,
                parse_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                data_pack=None,
                errors=[f"Parse error: {str(e)}"],
                warnings=warnings,
                parse_time_seconds=time.time() - start_time
            )


def demo_universal_parser():
    """Demonstrate universal parsing functionality"""
    print("=" * 70)
    print("UNIVERSAL DATA PARSER DEMONSTRATION")
    print("=" * 70)
    
    parser = UniversalParser()
    
    # Demo 1: Parse existing Ohio data
    print("\n Demo 1: Parse Ohio T1DM Dataset")
    print("-" * 50)
    
    ohio_path = Path("data_packs/public/ohio_t1dm/patient_559/timeseries.csv")
    if ohio_path.exists():
        result = parser.parse(str(ohio_path))
        
        if result.success:
            data_pack = result.data_pack
            print(f" Successfully parsed {data_pack.data_points} data points")
            print(f"   Duration: {data_pack.duration_hours:.1f} hours")
            print(f"   Confidence Score: {data_pack.confidence_score:.1%}")
            print(f"\n   Data Preview:")
            print(data_pack.data.head(3).to_string())
            
            if data_pack.quality_report:
                print(f"\n   Quality Report:")
                print(f"   - Completeness: {data_pack.quality_report.completeness_score:.1%}")
                print(f"   - Consistency: {data_pack.quality_report.consistency_score:.1%}")
                print(f"   - Validity: {data_pack.quality_report.validity_score:.1%}")
        else:
            print(f" Parse failed: {result.errors}")
    
    # Demo 2: Parse synthetic data
    print("\n\n Demo 2: Parse Synthetic Data")
    print("-" * 50)
    
    synthetic_csv = """timestamp,glucose_mg_dl,carbs_grams,insulin_units
0,120,0,0
5,125,0,0.5
10,130,30,0
15,140,0,1.0
20,145,0,0
25,150,0,0
30,148,0,0
35,145,0,0
40,140,0,0
45,135,0,0
50,130,0,0"""
    
    result = parser.parse_string(synthetic_csv, format_type='csv')
    
    if result.success:
        data_pack = result.data_pack
        print(f" Successfully parsed synthetic data")
        print(f"   Data Points: {data_pack.data_points}")
        print(f"   Duration: {data_pack.duration_hours:.1f} hours")
        print(f"   Confidence: {data_pack.confidence_score:.1%}")
        print(f"\n   Data Preview:")
        print(data_pack.data.to_string())
        
        if result.warnings:
            print(f"\n   Warnings:")
            for w in result.warnings:
                print(f"   {w}")
    
    # Demo 3: Data with issues
    print("\n\n Demo 3: Data with Quality Issues")
    print("-" * 50)
    
    problematic_csv = """timestamp,glucose,carbs,insulin
0,120,0,0
5,700,0,0
10,130,0,0
15,,0,0
20,140,0,0
25,145,0,0
30,50,0,0
35,140,0,0
40,155,0,0"""
    
    result = parser.parse_string(problematic_csv, format_type='csv')
    
    if result.success:
        data_pack = result.data_pack
        print(f" Parsed with quality issues detected")
        print(f"   Confidence: {data_pack.confidence_score:.1%}")
        print(f"\n   Warnings:")
        for w in result.warnings:
            print(f"   {w}")
    else:
        print(f" Parse failed: {result.errors}")
    
    # Demo 4: Different column names
    print("\n\n Demo 4: Different Column Names (Custom Format)")
    print("-" * 50)
    
    custom_csv = """Time (min),BG Value,Carbohydrates,Insulin (U)
0,115,0,0
5,118,0,0.5
10,125,30,0
15,135,0,1.0
20,140,0,0"""
    
    result = parser.parse_string(custom_csv, format_type='csv')
    
    if result.success:
        data_pack = result.data_pack
        print(f" Successfully parsed custom format")
        print(f"   Data Points: {data_pack.data_points}")
        print(f"   Column Mapping: {data_pack.metadata.get('column_mapping', {})}")
        print(f"\n   Data Preview:")
        print(data_pack.data.to_string())
    
    
    # Demo 5: Cleaning data with physiological feasibility check
    print("\n\n Demo 5: Cleaning Data with Physiological Feasibility Check")
    print("-" * 50)
    
    dirty_csv = """Time,Glucose.Level,Meal.Carbs,Delivered.Insulin
0,120,0,0
5,800,0,0
10,130,0,0
15,140,0,0
20,250,0,0
25,155,0,0
"""
    
    result = parser.parse_string(dirty_csv, format_type='csv')
    
    if result.success:
        data_pack = result.data_pack
        print("Successfully parsed and cleaned dirty data")
        print(f"   Data Points: {data_pack.data_points}")
        print(f"   Column Mapping: {data_pack.metadata.get('column_mapping', {})}")
        print("\n   Original Data Preview:")
        # To show the original data, we can re-parse without validation
        original_result = parser.parse_string(dirty_csv, format_type='csv', validate=False)
        if original_result.success:
            print(original_result.data_pack.data.to_string())
        
        print("\n   Cleaned Data Preview (anomalies set to NaN):")
        print(data_pack.data.to_string())
        
        if result.warnings:
            print("\n   Warnings:")
            for w in result.warnings:
                print(f"   {w}")
    else:
        print(f" Parse failed: {result.errors}")

    print("\n" + "=" * 70)
    print("UNIVERSAL DATA PARSER DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_universal_parser()