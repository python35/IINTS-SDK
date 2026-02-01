from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput, AlgorithmResult, AlgorithmMetadata
from typing import Dict, Any

class MyTestAlgorithm(InsulinAlgorithm):
    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__(settings)
        self.set_algorithm_metadata(AlgorithmMetadata(
            name="MyTest",
            author="GeminiCLI",
            description="A new custom insulin algorithm.",
            algorithm_type="rule_based" # Change as appropriate
        ))
        # Initialize any specific state or parameters for your algorithm here

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        # --- YOUR ALGORITHM LOGIC GOES HERE ---
        # This is a basic placeholder. Implement your actual insulin prediction logic.

        # Example: Deliver 0.1 units if glucose is above 120 mg/dL
        total_insulin = 0.0
        bolus_insulin = 0.0
        basal_insulin = 0.0
        correction_bolus = 0.0
        meal_bolus = 0.0

        if data.current_glucose > 120:
            correction_bolus = (data.current_glucose - 120) / self.isf / 5 # Example: 1 unit per 50 mg/dL above 120
            total_insulin += correction_bolus
            self._log_reason(f"Correcting high glucose", "glucose_level", data.current_glucose, f"Delivered {correction_bolus:.2f} units to reduce {data.current_glucose} mg/dL")

        if data.carb_intake > 0:
            meal_bolus = data.carb_intake / self.icr
            total_insulin += meal_bolus
            self._log_reason(f"Meal intake detected", "carb_intake", data.carb_intake, f"Delivered {meal_bolus:.2f} units for {data.carb_intake}g carbs")

        # Simulate basal rate (e.g., a continuous small delivery)
        # For simplicity, let's assume a fixed basal delivery over the time step
        # You might integrate this with your overall basal strategy
        # basal_insulin = 0.01 * data.time_step # Example: 0.01 units per minute basal
        # total_insulin += basal_insulin
        # self._log_reason(f"Maintaining basal rate", "basal", data.time_step, f"Delivered {basal_insulin:.2f} units basal")


        # Ensure no negative insulin delivery
        total_insulin = max(0.0, total_insulin)

        # Store important decisions in the why_log for transparency
        self._log_reason(f"Final insulin decision: {total_insulin:.2f} units", "decision", total_insulin)

        return {
            "total_insulin_delivered": total_insulin,
            "bolus_insulin": bolus_insulin, # You might differentiate between meal and correction bolus here
            "basal_insulin": basal_insulin,
            "correction_bolus": correction_bolus,
            "meal_bolus": meal_bolus,
        }
