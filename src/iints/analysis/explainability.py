import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Union, cast

class ExplainabilityAnalyzer:
    """
    Provides tools for analyzing and explaining the behavior of insulin algorithms.
    This module implements 'AI as explainability tool' by performing sensitivity analysis.
    """
    def __init__(self, simulation_results: pd.DataFrame):
        self.results = simulation_results

    def calculate_glucose_variability(self) -> Dict[str, float]:
        """
        Calculates basic metrics for glucose variability.
        """
        glucose_actual = self.results['glucose_actual_mgdl']
        return {
            "mean_glucose": glucose_actual.mean(),
            "std_dev_glucose": glucose_actual.std(),
            "min_glucose": glucose_actual.min(),
            "max_glucose": glucose_actual.max(),
            "time_in_range_70_180": (
                ((glucose_actual >= 70) & (glucose_actual <= 180)).sum() / len(glucose_actual)
            ) * 100
        }

    def analyze_insulin_response(self, algorithm_type: str = "") -> Dict[str, Any]:
        """
        Analyzes insulin delivery patterns.
        """
        total_insulin = self.results['delivered_insulin_units'].sum()
        basal_insulin = self.results['basal_insulin_units'].sum()
        bolus_insulin = self.results['bolus_insulin_units'].sum()
        correction_bolus = self.results['correction_bolus_units'].sum() # May not exist for all algorithms

        return {
            "total_insulin_delivered": total_insulin,
            "total_basal_insulin": basal_insulin,
            "total_bolus_insulin": bolus_insulin,
            "total_correction_bolus": correction_bolus,
            "algorithm_type": algorithm_type # For context
        }

    def perform_sensitivity_analysis(self,
                                     algorithm_instance: Any,
                                     parameter_name: str,
                                     original_value: float,
                                     perturbations: List[float],
                                     simulation_run_func: Callable[[Any, List[Any], int], pd.DataFrame],
                                     fixed_events: Optional[List[Any]] = None,
                                     duration_minutes: int = 1440 # 24 hours
                                    ) -> Dict[float, Dict[str, Union[float, str]]]:
        """
        Performs a basic sensitivity analysis by perturbing one algorithm parameter
        and observing the change in glucose metrics.

        Args:
            algorithm_instance (Any): An instance of the algorithm to test.
            parameter_name (str): The name of the parameter to perturb (e.g., 'carb_ratio').
            original_value (float): The original value of the parameter.
            perturbations (List[float]): A list of values to test for the parameter.
            simulation_run_func (Callable): A function that takes an algorithm instance,
                                            a list of fixed events, and duration_minutes,
                                            then returns simulation results (pd.DataFrame).
            fixed_events (List[Any]): Optional list of events to apply in each simulation run.
            duration_minutes (int): Duration for each simulation run.

        Returns:
            Dict[float, Dict[str, float]]: A dictionary where keys are the perturbed parameter values
                                            and values are dictionaries of glucose metrics.
        """
        original_settings = algorithm_instance.settings.copy()
        analysis_results = {}

        for p_value in perturbations:
            # Temporarily modify the parameter
            algorithm_instance.settings[parameter_name] = p_value
            algorithm_instance.reset() # Ensure algorithm state is reset

            try:
                # Call the provided simulation run function
                perturbed_results_df = simulation_run_func(algorithm_instance, fixed_events or [], duration_minutes)
                temp_analyzer = ExplainabilityAnalyzer(perturbed_results_df)
                analysis_results[p_value] = temp_analyzer.calculate_glucose_variability()
            except Exception as e:
                print(f"Error during sensitivity analysis for {parameter_name}={p_value}: {e}")
                analysis_results[p_value] = {"error": cast(Any, str(e))}

        # Restore original settings
        algorithm_instance.settings = original_settings
        algorithm_instance.reset()

        return analysis_results # type: ignore