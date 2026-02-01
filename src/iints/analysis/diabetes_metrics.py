import numpy as np
import pandas as pd

class DiabetesMetrics:
    """Professional diabetes metrics for algorithm evaluation."""
    
    @staticmethod
    def time_in_range(glucose_values, lower=70, upper=180):
        """Calculate Time In Range (TIR) percentage."""
        in_range = (glucose_values >= lower) & (glucose_values <= upper)
        return (in_range.sum() / len(glucose_values)) * 100
    
    @staticmethod
    def coefficient_of_variation(glucose_values):
        """Calculate CV - variability metric."""
        return (np.std(glucose_values) / np.mean(glucose_values)) * 100
    
    @staticmethod
    def blood_glucose_risk_index(glucose_values, risk_type='high'):
        """Calculate LBGI or HBGI."""
        def risk_function(bg):
            if risk_type == 'low':
                return 10 * (1.509 * (np.log(bg)**1.084 - 5.381))**2 if bg < 112.5 else 0
            else:  # high
                return 10 * (1.509 * (np.log(bg)**1.084 - 5.381))**2 if bg > 112.5 else 0
        
        risks = [risk_function(bg) for bg in glucose_values]
        return np.mean(risks)
    
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