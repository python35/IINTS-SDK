from typing import Any, Dict, Optional

class IndependentSupervisor:
    """
    Safety supervisor that operates independently to validate insulin delivery.
    Enforces hard constraints on insulin delivery to prevent hypoglycemia and other hazards.
    """
    def __init__(self):
        self.max_bolus = 25.0  # Default max bolus
        self.min_glucose_threshold = 70.0

    def validate_insulin_dose(self, proposed_dose: float, current_glucose: float, 
                              active_insulin: float, time_since_last_dose: float) -> float:
        """
        Validates the proposed insulin dose against safety constraints.
        """
        if current_glucose < self.min_glucose_threshold:
            return 0.0
        
        return min(proposed_dose, self.max_bolus)

# Alias for backward compatibility as the codebase migrates
SafetySupervisor = IndependentSupervisor

class InputValidator:
    """
    Validates simulation inputs.
    """
    pass