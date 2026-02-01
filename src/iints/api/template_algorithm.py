# IINTS-AF RESEARCH SDK: ALGORITHM TEMPLATE
#
# Welcome to the IINTS-AF Research SDK! This file is your starting point
# for creating your own insulin delivery algorithm.
#
# HOW IT WORKS:
# 1. COPY THIS FILE: Copy this template into a new file within this same
#     `user` directory (e.g., `MyAwesomeAlgorithm.py`).
# 2. RENAME THE CLASS: Change the class name from `TemplateAlgorithm` to something
#     unique (e.g., `MyAwesomeAlgorithm`).
# 3. WRITE YOUR LOGIC: Implement your insulin calculation logic in the
#     `predict_insulin` method.
# 4. RUN THE BATTLE: Your new algorithm will be automatically detected
#     and will appear as an option in the IINTS-AF terminal.
#
# Happy coding!
#
# -----------------------------------------------------------------------------

from typing import Dict, Any, List

# All algorithms must inherit from the InsulinAlgorithm base class.
# It provides the necessary structure and methods that the IINTS-AF simulator expects.
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput, AlgorithmMetadata, WhyLogEntry

# A "Dose" is a dictionary containing the insulin breakdown. The simulator
# currently only requires 'total_insulin_delivered'. The other keys are for
# more detailed logging and analysis.
Dose = Dict[str, float]


class TemplateAlgorithm(InsulinAlgorithm):
    """
    This is a template for a user-defined insulin delivery algorithm.
    It demonstrates the required structure and provides a basic, safe implementation
    of a correction and meal bolus logic.
    """

    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        """
        REQUIRED: This method provides metadata about your algorithm.
        The simulator uses this to identify and display your algorithm in the UI.
        """
        return AlgorithmMetadata(
            name="Template Algorithm",  # The name that will appear in the UI
            version="1.0.0",
            author="Your Name Here",
            description="A basic example algorithm demonstrating the IINTS-AF SDK.",
            algorithm_type="rule_based" # 'rule_based', 'ml', or 'hybrid'
        )

    def predict_insulin(self, data: AlgorithmInput) -> Dose:
        """
        REQUIRED: This is the core method of your algorithm.
        The simulator calls this method at each time step (e.g., every 5 minutes).

        Args:
            data (AlgorithmInput): An object containing the current patient data.
                - data.current_glucose (float): The latest glucose reading in mg/dL.
                - data.insulin_on_board (float): Estimated insulin still active from previous doses.
                - data.carb_intake (float): Carbohydrates from a meal announced at this time step (in grams).
                - data.time_step (float): The duration since the last call, in minutes.
                - data.current_time (float): The current simulation time in minutes from the start.
                - data.patient_state (Dict): A dictionary for you to store custom state between function calls (e.g., tracking glucose trends).

        Returns:
            Dose (Dict[str, float]): A dictionary specifying the insulin dose to be delivered.
                                     Must contain at least the 'total_insulin_delivered' key.
        """
        # This function is called at every time step.
        # First, we clear the reasoning log from the previous step.
        self.why_log = []

        # --- Example: Accessing Patient Data ---
        current_glucose = data.current_glucose
        iob = data.insulin_on_board
        carbs = data.carb_intake

        # --- Your Custom State Management (Optional) ---
        # The `self.state` dictionary persists across calls for this simulation run.
        # You can use it to track trends, previous states, etc.
        previous_glucose = self.state.get('previous_glucose', current_glucose)
        glucose_trend = (current_glucose - previous_glucose) / data.time_step # mg/dL/min
        self.state['previous_glucose'] = current_glucose # Update state for the next call

        # ----------------------------------------------------------------------
        # --- YOUR CUSTOM LOGIC START ---
        # ----------------------------------------------------------------------

        # Initialize insulin components
        correction_bolus = 0.0
        meal_bolus = 0.0
        basal_rate = 0.0 # This simple example does not use a dynamic basal rate

        # Define algorithm parameters
        # In a real algorithm, these would be managed more robustly.
        target_glucose = 110  # mg/dL
        insulin_sensitivity_factor = self.isf # mg/dL per Unit of insulin (set in base class)
        carb_to_insulin_ratio = self.icr # grams of carb per Unit of insulin (set in base class)

        # --- Log Key Data for Traceability ---
        # This is how you make your algorithm's reasoning transparent!
        self._log_reason(f"Current glucose is {current_glucose:.0f} mg/dL.", "glucose_level", current_glucose)
        self._log_reason(f"Glucose trend is {glucose_trend:+.1f} mg/dL/min.", "velocity", glucose_trend)
        self._log_reason(f"Active insulin (IOB) is {iob:.2f} U.", "insulin_on_board", iob)


        # --- Example Correction Bolus Logic ---
        # Only correct if glucose is above the target
        if current_glucose > target_glucose:
            glucose_diff = current_glucose - target_glucose
            # Standard correction formula: (Current - Target) / ISF
            correction_bolus = glucose_diff / insulin_sensitivity_factor
            self._log_reason(
                f"Calculated correction bolus of {correction_bolus:.2f} U for high glucose.",
                "calculation",
                correction_bolus,
                clinical_impact="Aims to bring glucose back to target range."
            )
        else:
             self._log_reason("No correction bolus needed (glucose is at or below target).", "calculation")


        # --- Example Meal Bolus Logic ---
        # Deliver a bolus if carbs are announced
        if carbs > 0:
            # Standard meal bolus formula: Carbs / ICR
            meal_bolus = carbs / carb_to_insulin_ratio
            self._log_reason(
                f"Calculated meal bolus of {meal_bolus:.2f} U for {carbs}g carbs.",
                "calculation",
                meal_bolus,
                clinical_impact="Aims to cover carbohydrate intake from a meal."
            )
            
        # --- Example Safety Guard: Insulin Stacking ---
        # Reduce the calculated dose by the amount of insulin already on board (IOB).
        # This is a crucial safety feature to prevent hypoglycemia from "stacking" insulin.
        total_calculated_dose = meal_bolus + correction_bolus
        
        if total_calculated_dose > iob:
            final_bolus = total_calculated_dose - iob
            self._log_reason(
                f"Reducing dose by IOB ({iob:.2f} U). Final bolus: {final_bolus:.2f} U.",
                "safety",
                final_bolus,
                clinical_impact="Prevents insulin stacking and reduces hypoglycemia risk."
            )
        else:
            final_bolus = 0.0 # Don't give a negative dose
            self._log_reason(
                f"Calculated dose ({total_calculated_dose:.2f} U) is less than IOB ({iob:.2f} U). No bolus advised.",
                "safety",
                final_bolus,
                clinical_impact="High IOB indicates sufficient active insulin; delivering more could be dangerous."
            )

        # --- Example Safety Guard: Hypoglycemia Prevention ---
        # If glucose is low or dropping fast, suspend all insulin delivery.
        if current_glucose < 80 or glucose_trend < -3.0:
            final_bolus = 0.0
            basal_rate = 0.0
            self._log_reason(
                "SUSPENDING INSULIN: Glucose is low or dropping rapidly.",
                "safety",
                current_glucose,
                clinical_impact="Critical safety action to prevent severe hypoglycemia."
            )

        # ----------------------------------------------------------------------
        # --- YOUR CUSTOM LOGIC END ---
        # ----------------------------------------------------------------------

        # The final dose must be assembled into a dictionary.
        # The simulator only strictly requires 'total_insulin_delivered'.
        # The other keys are for detailed analysis and logging.
        dose: Dose = {
            'basal_insulin': basal_rate,
            'meal_bolus': meal_bolus,
            'correction_bolus': correction_bolus,
            'total_insulin_delivered': max(0, final_bolus + basal_rate) # Ensure no negative insulin
        }

        return dose

    def reset(self):
        """
        This method is called by the simulator at the start of a new simulation.
        Use it to reset any internal state of your algorithm.
        """
        # Reset the state dictionary for the new simulation run.
        self.state = {}
        # Also clear the reasoning log.
        self.why_log = []
        print(f"[{self.get_algorithm_metadata().name}] has been reset.")
