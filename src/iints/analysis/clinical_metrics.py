#!/usr/bin/env python3
"""
Clinical Metrics Calculator - IINTS-AF
Standard clinical metrics for diabetes algorithm comparison.

Calculates:
- Time-in-Range (TIR) 70-180 mg/dL
- Time-in-Range 70-140 mg/dL (Tight)
- Time-in-Range 70-110 mg/dL (Very Tight)
- Glucose Management Indicator (GMI)
- Coefficient of Variation (CV)
- hypoglycemia Index (HI)
- Various other clinical benchmarks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ClinicalMetricsResult:
    """Comprehensive clinical metrics result"""
    # Time-in-Range metrics
    tir_70_180: float  # Percentage of time in 70-180 mg/dL
    tir_70_140: float  # Percentage of time in 70-140 mg/dL
    tir_70_110: float  # Percentage of time in 70-110 mg/dL
    tir_below_70: float  # Time below 70 mg/dL
    tir_below_54: float  # Time below 54 mg/dL
    tir_above_180: float  # Time above 180 mg/dL
    tir_above_250: float  # Time above 250 mg/dL
    
    # Glucose variability
    cv: float  # Coefficient of variation
    sd: float  # Standard deviation
    
    # Glucose management
    gmi: float  # Glucose Management Indicator
    mean_glucose: float
    median_glucose: float
    
    # Hypoglycemia metrics
    hi: float  # Hypoglycemia Index
    lbgi: float  # Low Blood Glucose Index
    hbgi: float  # High Blood Glucose Index
    
    # Additional metrics
    readings_per_day: float
    data_coverage: float
    
    def to_dict(self) -> Dict:
        return {
            'tir_70_180': self.tir_70_180,
            'tir_70_140': self.tir_70_140,
            'tir_70_110': self.tir_70_110,
            'tir_below_70': self.tir_below_70,
            'tir_below_54': self.tir_below_54,
            'tir_above_180': self.tir_above_180,
            'tir_above_250': self.tir_above_250,
            'cv': self.cv,
            'sd': self.sd,
            'gmi': self.gmi,
            'mean_glucose': self.mean_glucose,
            'median_glucose': self.median_glucose,
            'hi': self.hi,
            'lbgi': self.lbgi,
            'hbgi': self.hbgi,
            'readings_per_day': self.readings_per_day,
            'data_coverage': self.data_coverage
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Time-in-Range (70-180): {self.tir_70_180:.1f}%",
            f"Time-in-Range (70-140): {self.tir_70_140:.1f}%",
            f"Time <70 mg/dL: {self.tir_below_70:.1f}%",
            f"Time <54 mg/dL: {self.tir_below_54:.1f}%",
            f"Time >180 mg/dL: {self.tir_above_180:.1f}%",
            f"Time >250 mg/dL: {self.tir_above_250:.1f}%",
            f"Glucose Management Indicator: {self.gmi:.1f}%",
            f"Coefficient of Variation: {self.cv:.1f}%",
            f"Mean Glucose: {self.mean_glucose:.1f} mg/dL",
            f"Hypoglycemia Index: {self.hi:.2f}",
            f"Low BG Index: {self.lbgi:.2f}",
            f"High BG Index: {self.hbgi:.2f}"
        ]
        return '\n'.join(lines)
    
    def get_rating(self) -> str:
        """Get overall rating based on TIR and CV"""
        # TIR rating
        if self.tir_70_180 >= 70:
            tir_rating = "Excellent"
        elif self.tir_70_180 >= 50:
            tir_rating = "Good"
        elif self.tir_70_180 >= 30:
            tir_rating = "Fair"
        else:
            tir_rating = "Poor"
        
        # CV rating (lower is better)
        if self.cv <= 36:
            cv_rating = "Stable"
        elif self.cv <= 50:
            cv_rating = "Moderate Variability"
        else:
            cv_rating = "High Variability"
        
        return f"{tir_rating} ({tir_rating}) | {cv_rating}"


class ClinicalMetricsCalculator:
    """
    Calculate standard clinical metrics for diabetes management.
    
    Based on:
    - International Consensus on Time-in-Range
    - ATTD (Advanced Technologies & Treatments for Diabetes) guidelines
    - ADA (American Diabetes Association) Standards of Care
    """
    
    # Clinical thresholds (mg/dL)
    THRESHOLDS = {
        'very_low': 54,
        'low': 70,
        'target_low': 70,
        'target_high': 180,
        'high': 250,
        'very_high': 350
    }
    
    def __init__(self, 
                 target_range: Tuple[float, float] = (70, 180),
                 tight_range: Tuple[float, float] = (70, 140)):
        """
        Initialize calculator.
        
        Args:
            target_range: Target glucose range (low, high)
            tight_range: Tight target range (low, high)
        """
        self.target_range = target_range
        self.tight_range = tight_range
        
    def calculate_tir(self, 
                      glucose: pd.Series, 
                      low: float, 
                      high: float) -> float:
        """
        Calculate Time-in-Range percentage.
        
        Args:
            glucose: Series of glucose values
            low: Lower threshold
            high: Upper threshold
            
        Returns:
            Percentage of time in range (0-100)
        """
        if len(glucose) == 0:
            return 0.0
        
        in_range = ((glucose >= low) & (glucose <= high)).sum()
        return (in_range / len(glucose)) * 100
    
    def calculate_all_tir_metrics(self, glucose: pd.Series) -> Dict[str, float]:
        """Calculate all TIR-related metrics"""
        return {
            'tir_70_180': self.calculate_tir(glucose, 70, 180),
            'tir_70_140': self.calculate_tir(glucose, 70, 140),
            'tir_70_110': self.calculate_tir(glucose, 70, 110),
            'tir_below_70': self.calculate_tir(glucose, 0, 70),
            'tir_below_54': self.calculate_tir(glucose, 0, 54),
            'tir_above_180': self.calculate_tir(glucose, 180, 600),
            'tir_above_250': self.calculate_tir(glucose, 250, 600)
        }
    
    def calculate_gmi(self, glucose: pd.Series) -> float:
        """
        Calculate Glucose Management Indicator (GMI).
        
        GMI is an estimate of HbA1c based on mean glucose.
        Formula: GMI (%) = 3.31 + (0.02392 × mean glucose in mg/dL)
        
        Args:
            glucose: Series of glucose values
            
        Returns:
            GMI percentage
        """
        if len(glucose) == 0:
            return 0.0
        
        mean_glucose = glucose.mean()
        gmi = 3.31 + (0.02392 * mean_glucose)
        return min(max(gmi, 0), 15)  # Clamp to realistic range
    
    def calculate_cv(self, glucose: pd.Series) -> float:
        """
        Calculate Coefficient of Variation (CV).
        
        CV = (SD / Mean) × 100
        
        Lower CV indicates more stable glucose.
        Target: CV ≤ 36% (from consensus)
        
        Args:
            glucose: Series of glucose values
            
        Returns:
            CV percentage
        """
        if len(glucose) == 0:
            return 0.0
        
        mean = glucose.mean()
        if mean == 0:
            return 0.0
        
        std = glucose.std()
        return (std / mean) * 100
    
    def calculate_hypoglycemia_index(self, 
                                     glucose: pd.Series, 
                                     timestamp: Optional[pd.Series] = None) -> float:
        """
        Calculate Hypoglycemia Index (HI).
        
        HI measures severity and duration of hypoglycemia events.
        
        Args:
            glucose: Series of glucose values
            timestamp: Optional series of timestamps (in minutes)
            
        Returns:
            Hypoglycemia Index value
        """
        if len(glucose) == 0:
            return 0.0
        
        # Find hypoglycemia episodes (glucose < 70)
        low_glucose = glucose[glucose < 70]
        
        if len(low_glucose) == 0:
            return 0.0
        
        # Calculate severity-weighted index
        # Lower glucose values contribute more
        severity = (70 - low_glucose) / 70  # Normalized severity (0-1)
        severity = severity.clip(0, 1)
        
        # Sum of severity scores
        hi = severity.sum()
        
        return hi / len(glucose) * 100 if len(glucose) > 0 else 0
    
    def calculate_lbgi(self, glucose: pd.Series) -> float:
        """
        Calculate Low Blood Glucose Index (LBGI).
        
        LBGI quantifies the frequency and extent of low glucose readings.
        Based on: Kovatchev et al. (2000)
        
        Args:
            glucose: Series of glucose values
            
        Returns:
            LBGI value
        """
        if len(glucose) == 0:
            return 0.0
        
        # Transform glucose to risk space
        # BG Risk function: f(BG) = 1.509 × (ln(BG)^1.084 - 5.381)
        lbgi = 0.0
        
        for bg in glucose:
            if bg > 0:
                try:
                    risk = 1.509 * ((np.log(bg) ** 1.084) - 5.381)
                    if risk < 0:
                        # Low glucose risk
                        lbgi += risk ** 2
                except (ValueError, OverflowError):
                    pass
        
        return lbgi / len(glucose)
    
    def calculate_hbgi(self, glucose: pd.Series) -> float:
        """
        Calculate High Blood Glucose Index (HBGI).
        
        HBGI quantifies the frequency and extent of high glucose readings.
        Based on: Kovatchev et al. (2000)
        
        Args:
            glucose: Series of glucose values
            
        Returns:
            HBGI value
        """
        if len(glucose) == 0:
            return 0.0
        
        hbgi = 0.0
        
        for bg in glucose:
            if bg > 0:
                try:
                    risk = 1.509 * ((np.log(bg) ** 1.084) - 5.381)
                    if risk > 0:
                        # High glucose risk
                        hbgi += risk ** 2
                except (ValueError, OverflowError):
                    pass
        
        return hbgi / len(glucose)
    
    def calculate_readings_per_day(self, 
                                   glucose: pd.Series, 
                                   duration_hours: float) -> float:
        """
        Calculate average number of readings per day.
        
        Args:
            glucose: Series of glucose values
            duration_hours: Duration of data in hours
            
        Returns:
            Readings per day
        """
        if duration_hours == 0:
            return len(glucose)
        
        readings_per_hour = len(glucose) / duration_hours
        return readings_per_hour * 24
    
    def calculate_data_coverage(self,
                                glucose: pd.Series,
                                expected_interval_minutes: int = 5,
                                duration_hours: float = 24) -> float:
        """
        Calculate data coverage percentage.
        
        Args:
            glucose: Series of glucose values
            expected_interval_minutes: Expected time between readings
            duration_hours: Duration of data in hours
            
        Returns:
            Data coverage percentage
        """
        if duration_hours == 0:
            return 0.0
        
        expected_readings = (duration_hours * 60) / expected_interval_minutes
        actual_readings = len(glucose)
        
        return min((actual_readings / expected_readings) * 100, 100)
    
    def calculate(self, 
                  glucose: pd.Series,
                  timestamp: Optional[pd.Series] = None,
                  duration_hours: Optional[float] = None) -> ClinicalMetricsResult:
        """
        Calculate all clinical metrics.
        
        Args:
            glucose: Series of glucose values
            timestamp: Optional series of timestamps (in minutes)
            duration_hours: Optional duration in hours (calculated if not provided)
            
        Returns:
            ClinicalMetricsResult with all calculated values
        """
        # Calculate duration if not provided
        if duration_hours is None and timestamp is not None and len(timestamp) > 1:
            duration_hours = (timestamp.max() - timestamp.min()) / 60
        elif duration_hours is None:
            # Estimate from reading frequency (assume 5-min intervals)
            duration_hours = len(glucose) * 5 / 60
        
        # Remove NaN values for calculations
        clean_glucose = glucose.dropna()
        
        # Calculate all metrics
        tir_metrics = self.calculate_all_tir_metrics(clean_glucose)
        
        result = ClinicalMetricsResult(
            # TIR metrics
            tir_70_180=tir_metrics['tir_70_180'],
            tir_70_140=tir_metrics['tir_70_140'],
            tir_70_110=tir_metrics['tir_70_110'],
            tir_below_70=tir_metrics['tir_below_70'],
            tir_below_54=tir_metrics['tir_below_54'],
            tir_above_180=tir_metrics['tir_above_180'],
            tir_above_250=tir_metrics['tir_above_250'],
            
            # Variability
            cv=self.calculate_cv(clean_glucose),
            sd=float(clean_glucose.std()),
            
            # Management
            gmi=self.calculate_gmi(clean_glucose),
            mean_glucose=float(clean_glucose.mean()),
            median_glucose=float(clean_glucose.median()),
            
            # Hypoglycemia
            hi=self.calculate_hypoglycemia_index(clean_glucose, timestamp),
            lbgi=self.calculate_lbgi(clean_glucose),
            hbgi=self.calculate_hbgi(clean_glucose),
            
            # Additional
            readings_per_day=self.calculate_readings_per_day(clean_glucose, duration_hours),
            data_coverage=self.calculate_data_coverage(clean_glucose, duration_hours=duration_hours)
        )
        
        return result
    
    def compare_metrics(self, 
                        metrics1: ClinicalMetricsResult,
                        metrics2: ClinicalMetricsResult) -> Dict[str, Tuple[float, str]]:
        """
        Compare two sets of metrics.
        
        Args:
            metrics1: First set of metrics
            metrics2: Second set of metrics
            
        Returns:
            Dictionary of differences
        """
        comparison = {}
        
        # TIR comparison (higher is better)
        tir_1 = metrics1.tir_70_180
        tir_2 = metrics2.tir_70_180
        diff = tir_1 - tir_2
        comparison['tir_70_180'] = (diff, 'better' if diff > 0 else 'worse' if diff < 0 else 'equal')
        
        # CV comparison (lower is better)
        cv_1 = metrics1.cv
        cv_2 = metrics2.cv
        diff = cv_2 - cv_1  # Invert so positive = better
        comparison['cv'] = (abs(diff), 'better' if diff > 0 else 'worse' if diff < 0 else 'equal')
        
        # GMI comparison (lower is better)
        gmi_1 = metrics1.gmi
        gmi_2 = metrics2.gmi
        diff = gmi_2 - gmi_1  # Invert
        comparison['gmi'] = (abs(diff), 'better' if diff > 0 else 'worse' if diff < 0 else 'equal')
        
        # Hypoglycemia comparison (lower is better)
        lbgi_1 = metrics1.lbgi
        lbgi_2 = metrics2.lbgi
        diff = lbgi_2 - lbgi_1  # Invert
        comparison['lbgi'] = (abs(diff), 'better' if diff > 0 else 'worse' if diff < 0 else 'equal')
        
        return comparison


def demo_clinical_metrics():
    """Demonstrate clinical metrics calculation"""
    print("=" * 70)
    print("CLINICAL METRICS CALCULATOR DEMONSTRATION")
    print("=" * 70)
    
    calculator = ClinicalMetricsCalculator()
    
    # Generate sample glucose data
    np.random.seed(42)
    n_points = 288  # 24 hours at 5-min intervals
    
    # Simulate realistic glucose patterns
    time = np.arange(n_points)
    base_glucose = 120 + 30 * np.sin(time / (24 * 12 / (2 * np.pi)))  # Daily pattern
    glucose = base_glucose + np.random.normal(0, 15, n_points)
    
    # Add some excursions
    glucose[50:60] = np.random.uniform(200, 280, 10)  # Morning high
    glucose[140:150] = np.random.uniform(50, 65, 10)  # Afternoon low
    
    # Clip to realistic range
    glucose = np.clip(glucose, 40, 400)
    
    df = pd.DataFrame({
        'timestamp': time * 5,  # 5-minute intervals in minutes
        'glucose': glucose
    })
    
    # Calculate metrics
    print("\n Glucose Data Analysis")
    print("-" * 50)
    print(f"Data points: {len(df)}")
    print(f"Duration: {df['timestamp'].max() / 60:.1f} hours")
    print(f"Mean Glucose: {df['glucose'].mean():.1f} mg/dL")
    print(f"Std Deviation: {df['glucose'].std():.1f} mg/dL")
    
    result = calculator.calculate(
        glucose=df['glucose'],
        timestamp=df['timestamp'],
        duration_hours=24
    )
    
    print("\n Clinical Metrics")
    print("-" * 50)
    print(result.get_summary())
    
    print("\n Overall Rating")
    print("-" * 50)
    print(result.get_rating())
    
    # Compare with hypothetical better algorithm
    print("\n\n Comparison with Hypothetical Improved Algorithm")
    print("-" * 50)
    
    # Simulate improved algorithm (lower glucose, less variability)
    improved_glucose = np.clip(glucose - 10 + np.random.normal(0, 10, n_points), 40, 400)
    improved_result = calculator.calculate(
        glucose=pd.Series(improved_glucose),
        timestamp=df['timestamp'],
        duration_hours=24
    )
    
    comparison = calculator.compare_metrics(result, improved_result)
    
    print(f"Original TIR: {result.tir_70_180:.1f}% → Improved TIR: {improved_result.tir_70_180:.1f}%")
    print(f"Original CV: {result.cv:.1f}% → Improved CV: {improved_result.cv:.1f}%")
    print(f"Original GMI: {result.gmi:.1f}% → Improved GMI: {improved_result.gmi:.1f}%")
    
    print("\n Improvements")
    print("-" * 50)
    for metric, (diff, status) in comparison.items():
        sign = '+' if diff > 0 else ''
        print(f"  {metric}: {sign}{diff:.2f} ({status})")
    
    print("\n" + "=" * 70)
    print("CLINICAL METRICS DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_clinical_metrics()

