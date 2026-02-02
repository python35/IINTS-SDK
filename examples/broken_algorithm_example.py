# IINTS-AF RESEARCH SDK: "BAD DEVELOPER" TEST ALGORITHM
#
# This algorithm is intentionally designed to be faulty and dangerous.
# Its purpose is to test the robustness of the IINTS-AF Safety Supervisor
# and the clarity of the Traceability Log.
#
# It calculates a reasonable dose and then multiplies it by 10.
#
# -----------------------------------------------------------------------------

from typing import Dict, Any, List

from iints.core.algorithms.base_algorithm import InsulinAlgorithm, AlgorithmInput, AlgorithmMetadata, WhyLogEntry

Dose = Dict[str, float]


class BrokenAlgorithm(InsulinAlgorithm):
    """
    An intentionally faulty algorithm that recommends a dangerously high insulin dose.
    It serves as a test case for the IINTS-AF Safety Supervisor.
    """

    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        """Provides metadata for the "Broken Algorithm" test case."""
        return AlgorithmMetadata(
            name="!!! BROKEN ALGORITHM (TEST) !!!",
            version="0.1.0",
            author="Test Engineer",
            description="DANGER: This algorithm intentionally calculates a 10x overdose to test safety systems.",
            algorithm_type="rule_based"
        )

    def predict_insulin(self, data: AlgorithmInput) -> Dose:
        """
        Calculates a standard dose and then multiplies it by 10 to create a
        dangerous recommendation for testing purposes.
        """
        self.why_log = []

        # Standard parameters
        target_glucose = 110
        insulin_sensitivity_factor = self.isf
        carb_to_insulin_ratio = self.icr
        
        # Standard calculation for correction and meal bolus
        correction_bolus = 0.0
        if data.current_glucose > target_glucose:
            correction_bolus = (data.current_glucose - target_glucose) / insulin_sensitivity_factor

        meal_bolus = 0.0
        if data.carb_intake > 0:
            meal_bolus = data.carb_intake / carb_to_insulin_ratio
            
        total_calculated_dose = meal_bolus + correction_bolus

        # --- FAULTY LOGIC ---
        # This is where the "bad developer" makes a critical error.
        # We multiply the reasonable dose by 10 to simulate a dangerous overdose.
        final_bolus = total_calculated_dose * 10
        
        self._log_reason(
            f"Calculated base dose of {total_calculated_dose:.2f}U.",
            "calculation",
            total_calculated_dose
        )
        self._log_reason(
            "!!! DANGEROUS LOGIC: Multiplying dose by 10 for test !!!",
            "custom_error",
            final_bolus,
            clinical_impact="[DANGER] This will cause a severe overdose if not caught by safety systems."
        )

        # --- DO NOT add safety logic here for the test ---
        # We want to see if the external Safety Supervisor catches this.

        dose: Dose = {
            'total_insulin_delivered': max(0, final_bolus)
        }

        return dose

    def reset(self):
        """Resets the algorithm's state."""
        self.state = {}
        self.why_log = []
        
