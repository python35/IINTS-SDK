from typing import Optional

from iints.core.safety.config import SafetyConfig
from iints.core.safety.input_validator import InputValidator
from iints.core.supervisor import IndependentSupervisor as FullSupervisor

class IndependentSupervisor(FullSupervisor):
    """
    Safety supervisor that operates independently to validate insulin delivery.

    It enforces hard constraints on the dose an algorithm may deliver: it caps, reduces or
    blocks a proposed dose. That is not the same as preventing hypoglycemia. The supervisor
    acts only on the dose in front of it and cannot withdraw insulin already on board, so a
    low-glucose episode driven by earlier delivery, a missed meal or exercise remains
    possible with every constraint active. In this project's own supervisor-off ablation
    (research/EUCYS_REPORT.md), enabling supervision raised time below 70 mg/dL by 2.57
    points (95% CI +1.47 to +3.67) while raising Time in Range, so supervision is a
    trade-off mechanism and must be reported as one, not as a hypoglycemia guarantee.
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
