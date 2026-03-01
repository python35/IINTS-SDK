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


class RunawayAIAlgorithm(InsulinAlgorithm):
    """Delivers a maximal bolus during glucose decline to stress-test supervisor."""

    def __init__(
        self,
        max_bolus: float = 5.0,
        trigger_glucose: float = 140.0,
        settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(settings)
        self.max_bolus = max(0.0, float(max_bolus))
        self.trigger_glucose = float(trigger_glucose)

    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="RunawayAIAlgorithm",
            version="1.0",
            author="IINTS-AF SDK",
            algorithm_type="Chaos",
            description="Forces max bolus during falling glucose to test safety supervisor.",
            requires_training=False,
            supported_scenarios=["chaos", "runaway_ai"],
        )

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        trend = data.glucose_trend_mgdl_min
        current_glucose = data.current_glucose
        if current_glucose is None:
            dose = 0.0
        elif current_glucose <= self.trigger_glucose or (trend is not None and trend < 0):
            dose = self.max_bolus
        else:
            dose = 0.0
        return {
            "total_insulin_delivered": dose,
            "basal_insulin": 0.0,
            "bolus_insulin": dose,
            "meal_bolus": 0.0,
            "correction_bolus": dose,
            "uncertainty": 0.0,
            "fallback_triggered": False,
        }


class StackingAIAlgorithm(InsulinAlgorithm):
    """Delivers repeated boluses over consecutive steps to simulate stacking."""

    def __init__(
        self,
        bolus_units: float = 4.0,
        stack_steps: int = 3,
        trigger_glucose: float = 180.0,
        settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(settings)
        self.bolus_units = max(0.0, float(bolus_units))
        self.stack_steps = max(1, int(stack_steps))
        self.trigger_glucose = float(trigger_glucose)
        self._remaining = 0

    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="StackingAIAlgorithm",
            version="1.0",
            author="IINTS-AF SDK",
            algorithm_type="Chaos",
            description="Stacks multiple boluses across consecutive steps.",
            requires_training=False,
            supported_scenarios=["chaos", "stacking"],
        )

    def reset(self) -> None:
        super().reset()
        self._remaining = 0

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        if self._remaining <= 0 and data.current_glucose >= self.trigger_glucose:
            self._remaining = self.stack_steps

        if self._remaining > 0:
            dose = self.bolus_units
            self._remaining -= 1
        else:
            dose = 0.0

        return {
            "total_insulin_delivered": dose,
            "basal_insulin": 0.0,
            "bolus_insulin": dose,
            "meal_bolus": 0.0,
            "correction_bolus": dose,
            "uncertainty": 0.0,
            "fallback_triggered": False,
        }
