import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ReliabilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    passed: bool
    reliability_score: float  # 0-100%
    level: ReliabilityLevel
    issues: List[str]
    warnings: List[str]

class DataIntegrityValidator:
    """Validates data integrity for reverse engineering analysis."""
    
    def __init__(self):
        # Physiological limits
        self.max_glucose_rate = 10  # mg/dL per minute
        self.glucose_range = (20, 600)  # Physiologically possible range
        self.max_insulin_per_step = 5.0  # Units per 5-min step
        
    def validate_glucose_data(self, glucose_values: List[float], timestamps: List[float]) -> ValidationResult:
        """Validate glucose data for physiological plausibility."""
        issues = []
        warnings: List[str] = []
        score = 100.0
        
        # Check range
        for i, glucose in enumerate(glucose_values):
            if not (self.glucose_range[0] <= glucose <= self.glucose_range[1]):
                issues.append(f"Glucose {glucose} at step {i} outside physiological range")
                score -= 10
        
        # Check rate of change
        for i in range(1, len(glucose_values)):
            if len(timestamps) > i:
                time_diff = timestamps[i] - timestamps[i-1]
                glucose_diff = abs(glucose_values[i] - glucose_values[i-1])
                rate = glucose_diff / time_diff if time_diff > 0 else float('inf')
                
                if rate > self.max_glucose_rate:
                    issues.append(f"Impossible glucose rate: {rate:.1f} mg/dL/min at step {i}")
                    score -= 15
        
        # Check for missing values (NaN)
        nan_count = sum(1 for g in glucose_values if pd.isna(g))
        if nan_count > 0:
            warnings.append(f"{nan_count} missing glucose values detected")
            score -= nan_count * 5
        
        # Determine reliability level
        if score >= 90:
            level = ReliabilityLevel.HIGH
        elif score >= 70:
            level = ReliabilityLevel.MEDIUM
        elif score >= 50:
            level = ReliabilityLevel.LOW
        else:
            level = ReliabilityLevel.CRITICAL
        
        return ValidationResult(
            passed=len(issues) == 0,
            reliability_score=max(0, score),
            level=level,
            issues=issues,
            warnings=warnings
        )
    
    def validate_insulin_data(self, insulin_values: List[float]) -> ValidationResult:
        """Validate insulin delivery data."""
        issues = []
        warnings: List[str] = []
        score = 100.0
        
        for i, insulin in enumerate(insulin_values):
            if insulin < 0:
                issues.append(f"Negative insulin {insulin} at step {i}")
                score -= 20
            elif insulin > self.max_insulin_per_step:
                warnings.append(f"High insulin dose {insulin} at step {i}")
                score -= 5
        
        level = ReliabilityLevel.HIGH if score >= 90 else ReliabilityLevel.MEDIUM if score >= 70 else ReliabilityLevel.LOW
        
        return ValidationResult(
            passed=len(issues) == 0,
            reliability_score=max(0, score),
            level=level,
            issues=issues,
            warnings=warnings
        )

class AlgorithmicDriftDetector:
    """Detects when AI algorithms drift from safe baseline behavior."""
    
    def __init__(self, drift_threshold=0.5):  # 50% difference threshold
        self.drift_threshold = drift_threshold
        
    def detect_drift(self, ai_outputs: List[float], baseline_outputs: List[float]) -> ValidationResult:
        """Compare AI outputs against rule-based baseline."""
        issues = []
        warnings: List[str] = []
        score = 100.0
        
        if len(ai_outputs) != len(baseline_outputs):
            issues.append("AI and baseline output lengths don't match")
            return ValidationResult(False, 0, ReliabilityLevel.CRITICAL, issues, warnings)
        
        drift_count = 0
        extreme_drift_count = 0
        
        for i, (ai_val, baseline_val) in enumerate(zip(ai_outputs, baseline_outputs)):
            if baseline_val == 0:
                if ai_val > 0.1:  # AI gives insulin when baseline gives none
                    drift_count += 1
                    warnings.append(f"AI delivers insulin ({ai_val:.2f}) when baseline gives none at step {i}")
            else:
                relative_diff = abs(ai_val - baseline_val) / baseline_val
                if relative_diff > self.drift_threshold:
                    drift_count += 1
                    if relative_diff > 1.0:  # 100% difference
                        extreme_drift_count += 1
                        issues.append(f"Extreme drift: AI={ai_val:.2f}, Baseline={baseline_val:.2f} at step {i}")
                    else:
                        warnings.append(f"Drift detected: {relative_diff:.1%} difference at step {i}")
        
        # Calculate score based on drift frequency
        drift_rate = drift_count / len(ai_outputs)
        extreme_drift_rate = extreme_drift_count / len(ai_outputs)
        
        score -= drift_rate * 50  # Penalize drift
        score -= extreme_drift_rate * 30  # Extra penalty for extreme drift
        
        if extreme_drift_count > 0:
            level = ReliabilityLevel.CRITICAL
        elif drift_rate > 0.3:
            level = ReliabilityLevel.LOW
        elif drift_rate > 0.1:
            level = ReliabilityLevel.MEDIUM
        else:
            level = ReliabilityLevel.HIGH
        
        return ValidationResult(
            passed=extreme_drift_count == 0,
            reliability_score=max(0, score),
            level=level,
            issues=issues,
            warnings=warnings
        )

class StatisticalReliabilityChecker:
    """Checks statistical reliability of Monte Carlo results."""
    
    def __init__(self, min_runs=10, max_cv=0.3):  # Max 30% coefficient of variation
        self.min_runs = min_runs
        self.max_cv = max_cv
        
    def check_monte_carlo_reliability(self, results: List[List[float]]) -> ValidationResult:
        """Check if Monte Carlo results are statistically reliable."""
        issues = []
        warnings: List[str] = []
        score = 100.0
        
        if len(results) < self.min_runs:
            issues.append(f"Insufficient runs: {len(results)} < {self.min_runs}")
            score -= 30
        
        # Calculate coefficient of variation for each time step
        if len(results) > 1:
            results_array = np.array(results)
            means = np.mean(results_array, axis=0)
            stds = np.std(results_array, axis=0)
            
            # Avoid division by zero
            cvs = np.divide(stds, means, out=np.zeros_like(stds), where=means!=0)
            
            high_variance_steps = np.sum(cvs > self.max_cv)
            if high_variance_steps > 0:
                variance_rate = high_variance_steps / len(cvs)
                if variance_rate > 0.5:
                    issues.append(f"High variance in {variance_rate:.1%} of time steps")
                    score -= 40
                else:
                    warnings.append(f"Moderate variance in {variance_rate:.1%} of time steps")
                    score -= 20
        
        level = ReliabilityLevel.HIGH if score >= 90 else ReliabilityLevel.MEDIUM if score >= 70 else ReliabilityLevel.LOW
        
        return ValidationResult(
            passed=len(issues) == 0,
            reliability_score=max(0, score),
            level=level,
            issues=issues,
            warnings=warnings
        )

class ReverseEngineeringValidator:
    """Main validator for reverse engineering analysis."""
    
    def __init__(self):
        self.data_validator = DataIntegrityValidator()
        self.drift_detector = AlgorithmicDriftDetector()
        self.reliability_checker = StatisticalReliabilityChecker()
        
    def validate_simulation_results(self, simulation_df: pd.DataFrame, 
                                  baseline_results: Optional[List[float]] = None,
                                  monte_carlo_results: Optional[List[List[float]]] = None) -> Dict[str, ValidationResult]:
        """Comprehensive validation of simulation results."""
        
        results = {}
        
        # 1. Data integrity validation
        glucose_values = simulation_df['glucose_actual_mgdl'].tolist()
        timestamps = simulation_df['time_minutes'].tolist()
        insulin_values = simulation_df['delivered_insulin_units'].tolist()
        
        results['glucose_integrity'] = self.data_validator.validate_glucose_data(glucose_values, timestamps)
        results['insulin_integrity'] = self.data_validator.validate_insulin_data(insulin_values)
        
        # 2. Algorithmic drift detection
        if baseline_results:
            results['algorithmic_drift'] = self.drift_detector.detect_drift(insulin_values, baseline_results)
        
        # 3. Statistical reliability
        if monte_carlo_results:
            results['statistical_reliability'] = self.reliability_checker.check_monte_carlo_reliability(monte_carlo_results)
        
        return results
    
    def generate_reliability_report(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive reliability report."""
        
        overall_score = np.mean([result.reliability_score for result in validation_results.values()])
        
        all_issues = []
        all_warnings = []
        
        for category, result in validation_results.items():
            all_issues.extend([f"{category}: {issue}" for issue in result.issues])
            all_warnings.extend([f"{category}: {warning}" for warning in result.warnings])
        
        # Determine overall reliability
        if overall_score >= 90:
            overall_level = ReliabilityLevel.HIGH
            recommendation = "Results are highly reliable for reverse engineering analysis"
        elif overall_score >= 70:
            overall_level = ReliabilityLevel.MEDIUM
            recommendation = "Results are moderately reliable - consider additional validation"
        elif overall_score >= 50:
            overall_level = ReliabilityLevel.LOW
            recommendation = "Results have low reliability - use with caution"
        else:
            overall_level = ReliabilityLevel.CRITICAL
            recommendation = "Results are unreliable - do not use for analysis"
        
        return {
            "overall_reliability_score": overall_score,
            "overall_level": overall_level.value,
            "recommendation": recommendation,
            "total_issues": len(all_issues),
            "total_warnings": len(all_warnings),
            "issues": all_issues,
            "warnings": all_warnings,
            "category_scores": {category: result.reliability_score for category, result in validation_results.items()}
        }