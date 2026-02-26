from typing import Optional

from iints.core.safety.config import SafetyConfig
from iints.core.supervisor import IndependentSupervisor as FullSupervisor

class IndependentSupervisor(FullSupervisor):
    """
    Safety supervisor that operates independently to validate insulin delivery.
    Enforces hard constraints on insulin delivery to prevent hypoglycemia and other hazards.
    """
    def __init__(self, safety_config: Optional[SafetyConfig] = None):
        super().__init__(safety_config=safety_config)

    def validate_insulin_dose(
        self,
        proposed_dose: float,
        current_glucose: float,
        active_insulin: float,
        time_since_last_dose: float,
    ) -> float:
        """
        Backward-compatible API that routes through the full supervisor.
        """
        result = self.evaluate_safety(
            current_glucose=current_glucose,
            proposed_insulin=proposed_dose,
            current_time=0.0,
            current_iob=active_insulin,
        )
        return result["approved_insulin"]

# Alias for backward compatibility as the codebase migrates
SafetySupervisor = IndependentSupervisor

class InputValidator:
    """
    Validates simulation inputs.
    """
    pass
