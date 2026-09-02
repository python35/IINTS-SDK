import numpy as np
import pandas as pd

from iints.core import glycemic_risk

class DiabetesMetrics:
    """Professional diabetes metrics for algorithm evaluation."""
    
    @staticmethod
    def time_in_range(glucose_values, lower=70, upper=180):
        """Calculate Time In Range (TIR) percentage."""
        in_range = (glucose_values >= lower) & (glucose_values <= upper)
        return (in_range.sum() / len(glucose_values)) * 100
    
    @staticmethod
    def coefficient_of_variation(glucose_values):
        """Calculate CV - variability metric.

        Uses the sample standard deviation (ddof=1), matching
        iints.core.clinical_metrics so the two modules report the same CV.
        """
        values = np.asarray(glucose_values, dtype=float)
        return (np.std(values, ddof=1) / np.mean(values)) * 100

    @staticmethod
    def blood_glucose_risk_index(glucose_values, risk_type='high'):
        """Calculate LBGI or HBGI.

        Delegates to iints.core.glycemic_risk, the single definition used
        across the SDK. The previous inline version split the branches at a
        rounded 112.5 mg/dL; the canonical version splits on the sign of
        f(BG), which is the exact same boundary without the rounding.
        """
        if risk_type == 'low':
            return glycemic_risk.lbgi(glucose_values)
        return glycemic_risk.hbgi(glucose_values)
    
    @staticmethod
    def calculate_all_metrics(df, baseline=120):
        """Calculate comprehensive metrics suite."""
        glucose = df['glucose_actual_mgdl']
        
        return {
            "peak_glucose_mgdl": glucose.max(),
            "tir_percentage": DiabetesMetrics.time_in_range(glucose),
            "cv_percentage": DiabetesMetrics.coefficient_of_variation(glucose),
            "lbgi": DiabetesMetrics.blood_glucose_risk_index(glucose, 'low'),
            "hbgi": DiabetesMetrics.blood_glucose_risk_index(glucose, 'high'),
            "mean_glucose": glucose.mean(),
            "glucose_std": glucose.std()
        }