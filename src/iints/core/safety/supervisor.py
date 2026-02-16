from typing import Any, Dict, Optional

from iints.core.safety.config import SafetyConfig

class IndependentSupervisor:
    """
    Safety supervisor that operates independently to validate insulin delivery.
    Enforces hard constraints on insulin delivery to prevent hypoglycemia and other hazards.
    """
    def __init__(self, safety_config: Optional[SafetyConfig] = None):
        if safety_config is None:
            safety_config = SafetyConfig()
        self.max_bolus = safety_config.max_insulin_per_bolus
        self.min_glucose_threshold = safety_config.hypo_cutoff

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
