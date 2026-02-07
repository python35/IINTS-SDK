from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput, AlgorithmMetadata


class ConstantDoseAlgorithm(InsulinAlgorithm):
    """Always returns a fixed insulin dose for CI and smoke testing."""

    def __init__(self, dose: float = 0.5, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings)
        self.dose = max(0.0, float(dose))

    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="ConstantDoseAlgorithm",
            version="1.0",
            author="IINTS-AF SDK",
            algorithm_type="Mock",
            description="Always delivers a fixed insulin dose for CI testing.",
            requires_training=False,
            supported_scenarios=["baseline", "testing"],
        )

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        return {
            "total_insulin_delivered": self.dose,
            "basal_insulin": self.dose,
            "bolus_insulin": 0.0,
            "meal_bolus": 0.0,
            "correction_bolus": 0.0,
            "uncertainty": 0.0,
            "fallback_triggered": False,
        }


class RandomDoseAlgorithm(InsulinAlgorithm):
    """Returns a random dose within a safe range for stochastic testing."""

    def __init__(self, max_dose: float = 1.0, seed: Optional[int] = None, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings)
        self.max_dose = max(0.0, float(max_dose))
        self.rng = np.random.default_rng(seed)

    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="RandomDoseAlgorithm",
            version="1.0",
            author="IINTS-AF SDK",
            algorithm_type="Mock",
            description="Produces randomized doses for stress-testing and CI.",
            requires_training=False,
            supported_scenarios=["baseline", "testing"],
        )

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        dose = float(self.rng.uniform(0.0, self.max_dose))
        return {
            "total_insulin_delivered": dose,
            "basal_insulin": dose,
            "bolus_insulin": 0.0,
            "meal_bolus": 0.0,
            "correction_bolus": 0.0,
            "uncertainty": 0.0,
            "fallback_triggered": False,
        }
