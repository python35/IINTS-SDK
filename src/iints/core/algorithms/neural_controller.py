from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from iints.api.base_algorithm import AlgorithmInput, AlgorithmMetadata, InsulinAlgorithm
from iints.research.neural_control import (
    instantiate_neural_controller_model,
    load_neural_controller,
)


class ExperimentalNeuralController(InsulinAlgorithm):
    """Research-only PyTorch policy trained from safety-supervised teacher actions."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings)
        model_path = self.settings.get("model_path")
        if not model_path:
            raise ValueError("ExperimentalNeuralController requires settings['model_path'].")
        self.model_path = Path(str(model_path))
        self.payload = load_neural_controller(self.model_path)
        self.model = instantiate_neural_controller_model(self.payload)
        self.max_output_units = float(self.settings.get("max_output_units", self.payload["max_output_units"]))
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="Experimental Neural Controller",
                author="IINTS-AF Team",
                description=(
                    "Research-only PyTorch controller trained from safety-supervised teacher actions. "
                    "Not a clinically validated dosing system."
                ),
                algorithm_type="ml",
                requires_training=True,
            )
        )

    def _feature_vector(self, data: AlgorithmInput) -> np.ndarray:
        values = {
            "glucose_actual_mgdl": data.current_glucose,
            "glucose_trend_mgdl_min": data.glucose_trend_mgdl_min or 0.0,
            "patient_iob_units": data.insulin_on_board,
            "patient_cob_grams": data.carbs_on_board,
            "effective_isf": data.isf or self.isf,
            "effective_icr": data.icr or self.icr,
            "effective_basal_rate_u_per_hr": data.basal_rate_u_per_hr or 0.0,
            "carb_intake_grams": data.carb_intake,
        }
        raw = np.asarray([float(values[column]) for column in self.payload["feature_columns"]], dtype=float)
        mean = np.asarray(self.payload["feature_mean"], dtype=float)
        std = np.asarray(self.payload["feature_std"], dtype=float)
        return (raw - mean) / std

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        self.why_log = []
        try:
            import torch
        except Exception as exc:  # pragma: no cover - import failure is already validated on construction
            raise RuntimeError("Torch is required to run ExperimentalNeuralController.") from exc

        features = self._feature_vector(data)
        with torch.no_grad():
            tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
            prediction = float(self.model(tensor).reshape(-1)[0].item())
        prediction = max(0.0, min(prediction, self.max_output_units))
        if data.current_glucose < 70.0 or float(data.glucose_trend_mgdl_min or 0.0) <= -2.0:
            prediction = 0.0
            self._log_reason(
                "Neural policy output zeroed by local hypo guard",
                "safety",
                prediction,
                "Research-only controller does not dose into low or rapidly falling glucose.",
            )
        else:
            self._log_reason(
                "Local neural policy proposed insulin",
                "ml_policy",
                prediction,
                "Prediction remains subject to the deterministic safety supervisor.",
            )
        return {
            "total_insulin_delivered": prediction,
            "bolus_insulin": prediction,
            "basal_insulin": 0.0,
            "meal_bolus": 0.0,
            "correction_bolus": prediction,
        }
