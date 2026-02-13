from iints import InsulinAlgorithm, AlgorithmInput, AlgorithmResult, AlgorithmMetadata
from typing import Dict, Any

class {{ALGO_NAME}}(InsulinAlgorithm):
    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__(settings)
        self.set_algorithm_metadata(AlgorithmMetadata(
            name="{{ALGO_NAME}}",
            author="{{AUTHOR_NAME}}",
            description="A new custom insulin algorithm.",
            algorithm_type="rule_based" # Change as appropriate
        ))
        # Initialize any specific state or parameters for your algorithm here

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        # --- SAFETY-FIRST STARTER LOGIC ---
        # This template is intentionally conservative to avoid hypoglycemia.

        self.why_log = []

        current_glucose = data.current_glucose
        iob = data.insulin_on_board
        carbs = data.carb_intake

        previous_glucose = self.state.get("previous_glucose", current_glucose)
        glucose_trend = (current_glucose - previous_glucose) / max(data.time_step, 1)
        self.state["previous_glucose"] = current_glucose

        total_insulin = 0.0
        bolus_insulin = 0.0
        basal_insulin = 0.0
        correction_bolus = 0.0
        meal_bolus = 0.0

        # Hard safety cutoff
        if current_glucose < 90:
            self._log_reason("Glucose below 90 mg/dL; holding insulin.", "safety_cutoff", current_glucose)
            return {
                "total_insulin_delivered": 0.0,
                "bolus_insulin": 0.0,
                "basal_insulin": 0.0,
                "correction_bolus": 0.0,
                "meal_bolus": 0.0,
            }

        # If glucose is falling quickly, avoid correction bolus
        if glucose_trend < -1.0:
            self._log_reason(
                f"Glucose dropping at {glucose_trend:.2f} mg/dL/min; skipping correction bolus.",
                "safety_trend",
                glucose_trend,
            )
        else:
            # Conservative correction only if quite high
            if current_glucose > 180:
                correction_bolus = (current_glucose - 140) / self.isf
                correction_bolus = min(max(correction_bolus, 0.0), 0.5)
                total_insulin += correction_bolus
                self._log_reason(
                    f"Conservative correction bolus {correction_bolus:.2f} U.",
                    "correction",
                    current_glucose,
                )

        # Meal bolus (capped)
        if carbs > 0:
            meal_bolus = min(carbs / self.icr, 2.0)
            total_insulin += meal_bolus
            self._log_reason(
                f"Meal bolus {meal_bolus:.2f} U for {carbs:.0f} g carbs.",
                "meal_bolus",
                carbs,
            )

        # Optional: cap total insulin based on IOB
        if iob > 2.0:
            total_insulin = min(total_insulin, 0.2)
            self._log_reason("High IOB; capping total insulin to 0.2 U.", "iob_cap", iob)

        total_insulin = max(0.0, total_insulin)
        bolus_insulin = total_insulin

        self._log_reason(f"Final insulin decision: {total_insulin:.2f} units", "decision", total_insulin)

        return {
            "total_insulin_delivered": total_insulin,
            "bolus_insulin": bolus_insulin,
            "basal_insulin": basal_insulin,
            "correction_bolus": correction_bolus,
            "meal_bolus": meal_bolus,
        }
