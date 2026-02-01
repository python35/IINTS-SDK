from typing import Dict, Any, Optional
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput

class CorrectionBolus(InsulinAlgorithm):
    """
    An insulin algorithm that calculates a meal bolus based on carbohydrates
    and adds a correction bolus if current glucose is above a target.

    This algorithm introduces sensitivity to current glucose levels.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings)
        # Default settings, can be overridden by 'settings' dict
        self.default_settings = {
            "fixed_basal_rate": 0.8,    # Units per hour
            "carb_ratio": 10.0,         # Grams of carbs per unit of insulin
            "insulin_sensitivity_factor": 50.0, # mg/dL per unit of insulin
            "target_glucose": 100.0,    # mg/dL
            "max_bolus": 10.0           # Maximum single bolus in Units
        }
        # Merge default settings with provided settings
        self.settings = {**self.default_settings, **self.settings}

        # Validate essential settings
        if not all(k in self.settings for k in ["fixed_basal_rate", "carb_ratio", "insulin_sensitivity_factor", "target_glucose"]):
            raise ValueError("CorrectionBolus algorithm requires 'fixed_basal_rate', 'carb_ratio', 'insulin_sensitivity_factor', and 'target_glucose' in settings.")

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        self.why_log = [] # Clear the log for this prediction cycle

        basal_rate_units_per_minute = self.settings["fixed_basal_rate"] / 60.0
        basal_insulin = basal_rate_units_per_minute * data.time_step
        self._log_reason(f"Basal insulin calculated ({self.settings['fixed_basal_rate']} U/hr)", "basal_delivery", basal_insulin)

        carb_intake = data.carb_intake
        meal_bolus = 0.0
        if carb_intake > 0:
            meal_bolus = carb_intake / self.settings["carb_ratio"]
            self._log_reason(f"Meal bolus calculated for {carb_intake:.0f}g carbs (CR: {self.settings['carb_ratio']})", "meal_response", meal_bolus)
        else:
            self._log_reason("No meal bolus needed (no carb intake)", "meal_response", 0.0)


        correction_bolus = 0.0
        if data.current_glucose > self.settings["target_glucose"]:
            glucose_deviation = data.current_glucose - self.settings["target_glucose"]
            correction_bolus = glucose_deviation / self.settings["insulin_sensitivity_factor"]
            self._log_reason(f"Correction bolus calculated for high glucose (Current: {data.current_glucose:.0f}mg/dL, Target: {self.settings['target_glucose']:.0f}mg/dL, ISF: {self.settings['insulin_sensitivity_factor']})", "glucose_correction", correction_bolus)
        else:
            self._log_reason(f"No correction bolus needed (glucose at or below target: {self.settings['target_glucose']:.0f}mg/dL)", "glucose_correction", 0.0)


        # Ensure total bolus does not exceed max_bolus
        total_bolus_before_cap = meal_bolus + correction_bolus
        if total_bolus_before_cap > self.settings["max_bolus"]:
            original_meal_bolus = meal_bolus
            original_correction_bolus = correction_bolus
            
            # Prioritize meal bolus, then cap correction bolus
            if meal_bolus < self.settings["max_bolus"]:
                remaining_capacity = self.settings["max_bolus"] - meal_bolus
                correction_bolus = min(correction_bolus, remaining_capacity)
            else:
                meal_bolus = self.settings["max_bolus"] # Cap meal bolus too
                correction_bolus = 0.0 # No correction if meal bolus maxed out
            
            self._log_reason(f"Total bolus capped at {self.settings['max_bolus']}. Original: Meal={original_meal_bolus:.2f}, Corr={original_correction_bolus:.2f}. Adjusted: Meal={meal_bolus:.2f}, Corr={correction_bolus:.2f}.", "safety_constraint", self.settings['max_bolus'])

        total_bolus = meal_bolus + correction_bolus # Recalculate after capping
        total_insulin_delivered = basal_insulin + total_bolus

        self._log_reason(f"Final total insulin delivered: {total_insulin_delivered:.2f} U (Basal: {basal_insulin:.2f}, Meal Bolus: {meal_bolus:.2f}, Correction Bolus: {correction_bolus:.2f})", "final_decision", total_insulin_delivered)

        return {
            "basal_insulin": basal_insulin,
            "meal_bolus": meal_bolus,
            "correction_bolus": correction_bolus,
            "total_insulin_delivered": total_insulin_delivered
        }

    def __str__(self):
        return (f"CorrectionBolus Algorithm:\n"
                f"  Fixed Basal Rate: {self.settings['fixed_basal_rate']} U/hr\n"
                f"  Carb Ratio (CR): {self.settings['carb_ratio']} g/U\n"
                f"  Insulin Sensitivity Factor (ISF): {self.settings['insulin_sensitivity_factor']} mg/dL/U\n"
                f"  Target Glucose: {self.settings['target_glucose']} mg/dL")
