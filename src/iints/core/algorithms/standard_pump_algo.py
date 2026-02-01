from iints import InsulinAlgorithm, AlgorithmInput, AlgorithmMetadata
from typing import Dict, Any, Optional

class StandardPumpAlgorithm(InsulinAlgorithm):
    """
    A simplified algorithm representing a standard insulin pump.
    It delivers a fixed basal rate and a simple bolus based on carbs,
    with minimal correction for high glucose.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings)
        self.set_algorithm_metadata(AlgorithmMetadata(
            name="Standard Pump",
            author="IINTS-AF Team",
            description="A basic insulin pump algorithm with fixed basal and simple carb/correction bolus.",
            algorithm_type="rule_based"
        ))
        # Default settings for a standard pump
        self.isf = self.settings.get('isf', 50.0)  # Insulin Sensitivity Factor (mg/dL per Unit)
        self.icr = self.settings.get('icr', 10.0)  # Insulin to Carb Ratio (grams per Unit)
        self.basal_rate_per_hour = self.settings.get('basal_rate_per_hour', 0.8) # U/hr
        self.target_glucose = self.settings.get('target_glucose', 100.0) # mg/dL

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        self.why_log = [] # Clear log for each prediction

        total_insulin = 0.0
        bolus_insulin = 0.0
        basal_insulin = 0.0
        correction_bolus = 0.0
        meal_bolus = 0.0

        # 1. Basal Insulin
        # Convert hourly basal rate to dose for the current time step
        basal_insulin = (self.basal_rate_per_hour / 60.0) * data.time_step
        total_insulin += basal_insulin
        self._log_reason("Fixed basal insulin", "basal", basal_insulin, f"Delivered {basal_insulin:.2f} units for {data.time_step} min.")

        # 2. Meal Bolus
        if data.carb_intake > 0:
            meal_bolus = data.carb_intake / self.icr
            total_insulin += meal_bolus
            self._log_reason("Meal bolus for carb intake", "carb_intake", data.carb_intake, f"Delivered {meal_bolus:.2f} units for {data.carb_intake}g carbs.")
        
        # 3. Correction Bolus (simple)
        # Only correct if glucose is significantly above target and not too much IOB
        if data.current_glucose > self.target_glucose + 20 and data.insulin_on_board < 1.0:
            correction_bolus = (data.current_glucose - self.target_glucose) / self.isf / 2 # Correct half the difference
            if correction_bolus > 0:
                total_insulin += correction_bolus
                self._log_reason("Correction bolus for high glucose", "glucose_level", data.current_glucose, f"Delivered {correction_bolus:.2f} units to correct {data.current_glucose} mg/dL.")
        
        # Ensure no negative insulin delivery
        total_insulin = max(0.0, total_insulin)

        self._log_reason(f"Final insulin decision: {total_insulin:.2f} units", "decision", total_insulin)

        return {
            "total_insulin_delivered": total_insulin,
            "bolus_insulin": bolus_insulin,
            "basal_insulin": basal_insulin,
            "correction_bolus": correction_bolus,
            "meal_bolus": meal_bolus,
        }
