#!/usr/bin/env python3
"""
Data Quality Checker - IINTS-AF
Validates data quality and calculates confidence scores with gap detection.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    overall_score: float  # 0.0 - 1.0
    completeness_score: float  # Data coverage percentage
    consistency_score: float  # Temporal consistency
    validity_score: float  # Value range validation
    gaps: List['DataGap']
    anomalies: List['DataAnomaly']
    warnings: List[str]
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            'overall_score': self.overall_score,
            'completeness_score': self.completeness_score,
            'consistency_score': self.consistency_score,
            'validity_score': self.validity_score,
            'gaps': [g.to_dict() for g in self.gaps],
            'anomalies': [a.to_dict() for a in self.anomalies],
            'warnings': self.warnings,
            'summary': self.summary
        }


@dataclass
class DataGap:
    """Represents a gap in the data"""
    start_time: float
    end_time: float
    duration_minutes: float
    data_points_missing: int
    percentage_of_total: float
    time_range_description: str
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_minutes': self.duration_minutes,
            'data_points_missing': self.data_points_missing,
            'percentage_of_total': self.percentage_of_total,
            'time_range_description': self.time_range_description
        }
    
    def get_warning_message(self) -> str:
        """Generate human-readable warning message"""
        return (
            f"[WARN] DATA GAP DETECTED: {self.percentage_of_total:.1f}% of data missing "
            f"({self.data_points_missing} points) between {self.time_range_description} "
            f"({self.duration_minutes:.0f} minutes)"
        )


@dataclass
class DataAnomaly:
    """Represents an anomalous data point"""
    index: int
    timestamp: float
    value: float
    anomaly_type: str  # 'outlier', 'impossible_value', 'rapid_change'
    severity: str  # 'low', 'medium', 'high'
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'value': self.value,
            'anomaly_type': self.anomaly_type,
            'severity': self.severity,
            'description': self.description
        }


class DataQualityChecker:
    """
    Validates data quality and calculates confidence scores.
    
    Performs comprehensive checks:
    - Completeness: Detects missing data and gaps
    - Consistency: Validates temporal sampling
    - Validity: Checks value ranges
    
    Outputs confidence score and detailed warnings.
    """
    
    # Physiological limits for glucose values
    GLUCOSE_LIMITS = {
        'minimum': 20,    # mg/dL - physiologically possible minimum
        'maximum': 600,   # mg/dL - physiologically possible maximum
        'critical_low': 54,   # mg/dL - clinically significant low
        'critical_high': 350  # mg/dL - clinically significant high
    }
    
    PHYSIOLOGICAL_RATES = {
        'max_glucose_change_per_min': 19.9 # mg/dL/min - Detecting changes of 20 mg/dL/min or more
    }
    
    # Expected sampling intervals (in minutes)
    EXPECTED_INTERVALS = {
        'cgm': 5,      # Continuous Glucose Monitor
        'bg_meter': 60,  # Blood glucose meter
        'manual': 240    # Manual logging
    }
    
    def __init__(self, expected_interval: int = 5, source_type: str = 'cgm'):
        """
        Initialize quality checker.
        
        Args:
            expected_interval: Expected time between readings in minutes
            source_type: Data source type ('cgm', 'bg_meter', 'manual')
        """
        self.expected_interval = expected_interval
        self.source_type = source_type
        
    def check_completeness(self, df: pd.DataFrame) -> Tuple[float, List[DataGap]]:
        """
        Check data completeness and detect gaps.
        
        Args:
            df: DataFrame with timestamp and glucose columns
            
        Returns:
            Tuple of (completeness_score, list of gaps)
        """
        if 'timestamp' not in df.columns:
            return 1.0, []  # Can't check without timestamp
        
        timestamps = df['timestamp'].dropna().sort_values().astype(float)
        
        if len(timestamps) < 2:
            return 1.0, []
        
        # Calculate expected number of readings
        time_span = timestamps.iloc[-1] - timestamps.iloc[0]
        expected_readings = int((time_span / self.expected_interval) + 1)
        actual_readings = len(timestamps)
        
        # Completeness score
        completeness = min(1.0, actual_readings / expected_readings)
        
        # Detect gaps
        gaps = self._detect_gaps(timestamps, time_span, actual_readings, int(expected_readings))
        
        return completeness, gaps
    
    def _detect_gaps(self, 
                     timestamps: pd.Series, 
                     time_span: float,
                     actual_readings: int,
                     expected_readings: int) -> List[DataGap]:
        """Detect gaps in the data"""
        gaps: List[DataGap] = []
        
        if actual_readings < 2:
            return gaps
        
        # Calculate time differences between consecutive readings
        time_diffs = timestamps.diff().dropna().astype(float)
        
        # Threshold for gap detection (3x expected interval)
        gap_threshold = float(self.expected_interval * 3)
        
        # Find gap locations
        gap_indices = time_diffs[time_diffs > gap_threshold].index
        
        for idx in gap_indices:
            # Get timestamps around the gap
            before_idx = idx - 1
            after_idx = idx
            
            start_time = timestamps.loc[before_idx]
            end_time = timestamps.loc[after_idx]
            
            gap_duration = end_time - start_time
            points_missing = int(gap_duration / self.expected_interval) - 1
            gap_percentage = (points_missing / expected_readings) * 100 if expected_readings > 0 else 0
            
            # Create time range description
            start_minutes = int(start_time)
            end_minutes = int(end_time)
            hours_start = start_minutes // 60
            mins_start = start_minutes % 60
            hours_end = end_minutes // 60
            mins_end = end_minutes % 60
            
            time_range_desc = f"{hours_start:02d}:{mins_start:02d} - {hours_end:02d}:{mins_end:02d}"
            
            gap = DataGap(
                start_time=start_time,
                end_time=end_time,
                duration_minutes=gap_duration,
                data_points_missing=points_missing,
                percentage_of_total=gap_percentage,
                time_range_description=time_range_desc
            )
            gaps.append(gap)
        
        return gaps
    
    def check_consistency(self, df: pd.DataFrame) -> float:
        """
        Check temporal consistency of data.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Consistency score (0.0 - 1.0)
        """
        if 'timestamp' not in df.columns:
            return 1.0
        
        timestamps = df['timestamp'].dropna().sort_values()
        
        if len(timestamps) < 3:
            return 1.0
        
        # Calculate time differences
        time_diffs = timestamps.diff().dropna()
        
        if len(time_diffs) == 0:
            return 1.0
        
        # Check for irregular intervals
        mean_interval = time_diffs.mean()
        std_interval = time_diffs.std()
        
        # Coefficient of variation
        cv = std_interval / mean_interval if mean_interval > 0 else 0
        
        # Score based on CV (lower is better)
        if cv < 0.1:  # Very consistent
            return 1.0
        elif cv < 0.25:  # Mostly consistent
            return 0.9
        elif cv < 0.5:  # Somewhat inconsistent
            return 0.7
        else:  # Very inconsistent
            return 0.5
    
    def check_validity(self, df: pd.DataFrame) -> Tuple[float, List[DataAnomaly]]:
        """
        Check data validity and detect anomalies.
        
        Args:
            df: DataFrame with glucose column
            
        Returns:
            Tuple of (validity_score, list of anomalies)
        """
        anomalies: List[DataAnomaly] = []
        
        if 'glucose' not in df.columns:
            return 1.0, anomalies
        
        glucose = df['glucose'].dropna()
        
        if len(glucose) == 0:
            return 1.0, anomalies
        
        # Check for impossible values
        for idx, value in glucose.items():
            if value < self.GLUCOSE_LIMITS['minimum']:
                anomalies.append(DataAnomaly(
                    index=int(idx), # type: ignore
                    timestamp=float(df.at[idx, 'timestamp']), # type: ignore
                    value=value,
                    anomaly_type='impossible_value',
                    severity='high',
                    description=f"Glucose {value:.1f} mg/dL below physiological minimum ({self.GLUCOSE_LIMITS['minimum']})"
                ))
            elif value > self.GLUCOSE_LIMITS['maximum']:
                anomalies.append(DataAnomaly(
                    index=int(idx), # type: ignore
                    timestamp=float(df.at[idx, 'timestamp']), # type: ignore
                    value=value,
                    anomaly_type='impossible_value',
                    severity='high',
                    description=f"Glucose {value:.1f} mg/dL above physiological maximum ({self.GLUCOSE_LIMITS['maximum']})"
                ))
        
        # Check for outliers using IQR method
        q1 = glucose.quantile(0.25)
        q3 = glucose.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # Using 3*IQR for extreme outliers
        upper_bound = q3 + 3 * iqr
        
        for idx, value in glucose.items():
            if value < lower_bound or value > upper_bound:
                severity = 'low' if (abs(value - glucose.median()) < 3 * iqr) else 'medium'
                anomalies.append(DataAnomaly(
                    index=int(idx), # type: ignore
                    timestamp=float(df.at[idx, 'timestamp']), # type: ignore
                    value=value,
                    anomaly_type='outlier',
                    severity=severity,
                    description=f"Outlier glucose value {value:.1f} mg/dL"
                ))
        
        # Check for rapid glucose changes (physiologically impossible)
        if 'timestamp' in df.columns:
            glucose_with_time = df[['timestamp', 'glucose']].dropna().sort_values('timestamp')
            if len(glucose_with_time) >= 2:
                time_diff = glucose_with_time['timestamp'].diff() # type: ignore
                glucose_diff = glucose_with_time['glucose'].diff() # type: ignore
                
                # Rate of change in mg/dL per minute
                rate_of_change = glucose_diff / time_diff
                
                # Use the new class attribute
                max_rate = self.PHYSIOLOGICAL_RATES['max_glucose_change_per_min']
                rapid_change_mask = rate_of_change.abs() > max_rate
                
                for idx in rate_of_change[rapid_change_mask].index:
                    change = glucose_diff.loc[idx] # type: ignore
                    time_delta = time_diff.loc[idx] # type: ignore
                    actual_rate = rate_of_change.loc[idx] # type: ignore
                    
                    direction = "rise" if change > 0 else "drop"
                    description = (f"Impossible glucose {direction} of {actual_rate:.1f} mg/dL/min "
                                   f"(changed by {change:.1f} in {time_delta:.1f} min)")

                    anomalies.append(DataAnomaly(
                        index=int(idx), # type: ignore
                        timestamp=float(df.at[idx, 'timestamp']), # type: ignore
                        value=glucose_with_time.loc[idx, 'glucose'], # type: ignore
                        anomaly_type='rapid_change',
                        severity='high',
                        description=description
                    ))
        
        # Calculate validity score
        total_points = len(glucose)
        invalid_points = len(anomalies)
        
        if total_points == 0:
            return 1.0, anomalies
        
        validity = 1.0 - (invalid_points / total_points)
        
        return max(0.0, validity), anomalies
    
    def check(self, df: pd.DataFrame) -> QualityReport:
        """
        Perform comprehensive data quality check.
        
        Args:
            df: DataFrame to check
            
        Returns:
            QualityReport with all findings
        """
        warnings = []
        
        # Run all checks
        completeness, gaps = self.check_completeness(df)
        consistency = self.check_consistency(df)
        validity, anomalies = self.check_validity(df)
        
        # Calculate overall score (weighted average)
        overall = (
            completeness * 0.4 +
            consistency * 0.3 +
            validity * 0.3
        )
        
        # Generate warnings
        for gap in gaps:
            warnings.append(gap.get_warning_message())
            warnings.append(
                f"   [INFO] Simulation confidence score decreases to {max(0, overall - gap.percentage_of_total * 0.01):.2f}"
            )
        
        for anomaly in anomalies:
            if anomaly.severity == 'high':
                warnings.append(
                    f"[WARN] CRITICAL ANOMALY: {anomaly.description} at index {anomaly.index}"
                )
            elif anomaly.severity == 'medium':
                warnings.append(
                    f"[WARN] ANOMALY: {anomaly.description} at index {anomaly.index}"
                )
        
        # Summary generation
        if overall >= 0.9:
            summary = "Excellent data quality"
        elif overall >= 0.75:
            summary = "Good data quality with minor issues"
        elif overall >= 0.5:
            summary = "Moderate data quality - use with caution"
        elif overall >= 0.25:
            summary = "Poor data quality - significant gaps detected"
        else:
            summary = "Critical data quality issues - simulation may be unreliable"
        
        return QualityReport(
            overall_score=overall,
            completeness_score=completeness,
            consistency_score=consistency,
            validity_score=validity,
            gaps=gaps,
            anomalies=anomalies,
            warnings=warnings,
            summary=summary
        )
    
    def get_confidence_score(self, df: pd.DataFrame) -> float:
        """
        Get overall confidence score for simulation.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        report = self.check(df)
        return report.overall_score
    
    def print_report(self, report: QualityReport):
        """Print formatted quality report"""
        print("\n" + "=" * 70)
        print("DATA QUALITY REPORT")
        print("=" * 70)
        
        # Overall score with visual indicator
        score_bar = "█" * int(report.overall_score * 20) + "░" * (20 - int(report.overall_score * 20))
        print(f"\nOverall Score: [{score_bar}] {report.overall_score:.1%}")
        print(f"Summary: {report.summary}")
        
        # Component scores
        print(f"\nComponent Scores:")
        print(f"  Completeness:  {report.completeness_score:.1%}")
        print(f"  Consistency:   {report.consistency_score:.1%}")
        print(f"  Validity:      {report.validity_score:.1%}")
        
        # Gaps
        if report.gaps:
            print(f"\nData Gaps Found: {len(report.gaps)}")
            for gap in report.gaps:
                print(f"   {gap.get_warning_message()}")
        
        # Anomalies
        high_anomalies = [a for a in report.anomalies if a.severity == 'high']
        medium_anomalies = [a for a in report.anomalies if a.severity == 'medium']
        
        if high_anomalies:
            print(f"\nCRITICAL Anomalies: {len(high_anomalies)}")
            for anomaly in high_anomalies:
                print(f"   {anomaly.description}")
        
        if medium_anomalies:
            print(f"\nWarnings: {len(medium_anomalies)}")
            for anomaly in medium_anomalies[:5]:  # Show first 5
                print(f"   {anomaly.description}")
            if len(medium_anomalies) > 5:
                print(f"   ... and {len(medium_anomalies) - 5} more")
        
        # Warnings
        if report.warnings:
            print(f"\n{'='*70}")
            print("WARNINGS")
            print("=" * 70)
            for warning in report.warnings:
                print(f"  {warning}")
        
        print("\n" + "=" * 70)


def demo_quality_checker():
    """Demonstrate data quality checking"""
    print("=" * 70)
    print("DATA QUALITY CHECKER DEMONSTRATION")
    print("=" * 70)
    
    checker = DataQualityChecker(expected_interval=5, source_type='cgm')
    
    # Test case 1: Clean data
    print("\nTest Case 1: Clean Data (Simulated)")
    print("-" * 50)
    
    # Generate clean data
    np.random.seed(42)
    timestamps = np.arange(0, 480, 5)  # 8 hours, 5-min intervals
    glucose = 120 + 30 * np.sin(timestamps / 60) + np.random.normal(0, 5, len(timestamps))
    glucose = np.clip(glucose, 40, 400)  # Keep within reasonable range
    
    clean_df = pd.DataFrame({
        'timestamp': timestamps,
        'glucose': glucose,
        'carbs': np.random.choice([0, 30, 60], len(timestamps), p=[0.8, 0.15, 0.05]),
        'insulin': np.random.choice([0, 1, 2], len(timestamps), p=[0.7, 0.2, 0.1])
    })
    
    report = checker.check(clean_df)
    checker.print_report(report)
    
    # Test case 2: Data with gaps
    print("\n\nTest Case 2: Data with Gaps (14:00-16:00)")
    print("-" * 50)
    
    gap_df = clean_df.copy()
    
    # Remove data points between 14:00 and 16:00 (in minutes from start)
    # Assuming 14:00 = 840 minutes, but our data is only 0-480
    # Let's create a scenario where gap is in the middle
    gap_start_idx = 100  # Around index 100
    gap_end_idx = 130
    
    gap_df = gap_df.drop(range(gap_start_idx, gap_end_idx))
    gap_df = gap_df.reset_index(drop=True)
    
    report = checker.check(gap_df)
    checker.print_report(report)
    
    # Test case 3: Data with anomalies
    print("\n\nTest Case 3: Data with Anomalies")
    print("-" * 50)
    
    anomaly_df = clean_df.copy()
    
    # Add impossible values
    anomaly_df.loc[50, 'glucose'] = 15  # Too low
    anomaly_df.loc[100, 'glucose'] = 700  # Too high
    
    # Add outlier
    anomaly_df.loc[75, 'glucose'] = 400
    
    report = checker.check(anomaly_df)
    checker.print_report(report)
    
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECKER DEMONSTRATION COMPLETE")
