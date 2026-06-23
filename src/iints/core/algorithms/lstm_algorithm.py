try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False
import os
import math
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput
from .correction_bolus import CorrectionBolus
from iints.core.safety.config import (
    CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN,
    CONTROLLER_HIGH_IOB_GUARD_UNITS,
    CONTROLLER_HYPO_GUARD_MGDL,
    ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS,
)

if _TORCH_AVAILABLE:
    # Define the LSTM model with Dropout
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(p=dropout_prob)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # x shape: (batch_size, sequence_length, input_size)
            lstm_out, _ = self.lstm(x)
            # Use the output from the last time step and apply dropout
            out = self.dropout(lstm_out[:, -1, :])
            output = self.fc(out)
            return output
else:
    class LSTMModel:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Torch is required for LSTMModel. Install with `pip install iints[torch]`.")

if _TORCH_AVAILABLE:
    class LSTMInsulinAlgorithm(InsulinAlgorithm):
        """
        An insulin algorithm that uses a simple LSTM model with Monte Carlo Dropout
        for uncertainty estimation. If uncertainty is high, it falls back to a rule-based
        algorithm.
        """
        def __init__(self, settings: Optional[Dict[str, Any]] = None):
            super().__init__(settings)
            self.default_settings = {
                "input_features": 7,
                "hidden_size": 50,
                "output_size": 1,
                "dropout_prob": 0.5,
                "model_path": os.path.join(os.path.dirname(__file__), 'trained_lstm_model.pth'),
                "mc_samples": 50,
                "uncertainty_threshold": 0.5, # This threshold may need tuning
                "mc_seed": 42,
                "model_output_scale_units": 1.0,
                "max_model_candidate_units": ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS,
            }
            self.settings = {**self.default_settings, **(settings or {})}

            self.model = LSTMModel(
                self.settings["input_features"],
                self.settings["hidden_size"],
                self.settings["output_size"],
                self.settings["dropout_prob"]
            )
            
            self.model_loaded = False
            # A missing checkpoint must fail closed. Randomly initialized
            # networks are never allowed to produce an insulin candidate.
            if os.path.exists(self.settings['model_path']):
                print(f"Loading trained model from {self.settings['model_path']}")
                self.model.load_state_dict(
                    torch.load(self.settings['model_path'], map_location="cpu", weights_only=True)
                )
                self.model.eval()
                self.model_loaded = True
            else:
                print(
                    f"Warning: Trained model not found at {self.settings['model_path']}. "
                    "LSTM dosing is disabled; deterministic fallback will be used."
                )

            # Instantiate fallback algorithm
            self.fallback_algo = CorrectionBolus()
            self.reset()

        def reset(self):
            """Resets the algorithm's internal state."""
            super().reset()

        def _fallback(self, data: AlgorithmInput, reason: str) -> Dict[str, Any]:
            self._log_reason(reason, "safety_fallback", reason)
            if reason == "fixed_controller_safety_guard":
                result: Dict[str, Any] = {
                    "total_insulin_delivered": 0.0,
                    "bolus_insulin": 0.0,
                    "basal_insulin": 0.0,
                }
            else:
                result = self.fallback_algo.predict_insulin(data)
            result["fallback_triggered"] = True
            result["fallback_reason"] = reason
            result["model_checkpoint_loaded"] = self.model_loaded
            if reason != "fixed_controller_safety_guard":
                self.why_log.extend(self.fallback_algo.get_why_log())
            return result

        def _feature_tensor(self, data: AlgorithmInput):
            features = [
                float(data.current_glucose),
                float(data.glucose_trend_mgdl_min or 0.0),
                float(data.predicted_glucose_30min or data.current_glucose),
                float(data.carb_intake),
                float(data.insulin_on_board),
                float(data.isf or self.isf),
                float(data.icr or self.icr),
            ]
            if len(features) != int(self.settings["input_features"]):
                raise ValueError("LSTM input_features must match the seven documented SDK features")
            return torch.tensor(features, dtype=torch.float32).reshape(1, 1, len(features))

        def _deterministic_uncertainty(self, input_tensor) -> float:
            predictions = []
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.settings["mc_seed"]))
                self.model.train()
                with torch.no_grad():
                    for _ in range(int(self.settings["mc_samples"])):
                        predictions.append(float(self.model(input_tensor).item()))
            self.model.eval()
            return float(np.std(np.asarray(predictions, dtype=float)))

        def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
            self.why_log = [] # Clear the log for this prediction cycle

            if not self.model_loaded:
                return self._fallback(data, "missing_model_checkpoint")

            trend = float(data.glucose_trend_mgdl_min or 0.0)
            predicted = float(data.predicted_glucose_30min or data.current_glucose)
            if (
                float(data.current_glucose) <= CONTROLLER_HYPO_GUARD_MGDL
                or predicted <= CONTROLLER_HYPO_GUARD_MGDL
                or trend <= CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN
                or float(data.insulin_on_board) >= CONTROLLER_HIGH_IOB_GUARD_UNITS
            ):
                return self._fallback(data, "fixed_controller_safety_guard")

            input_tensor = self._feature_tensor(data)

            self._log_reason("LSTM input tensor created", "data_preparation", input_tensor.tolist())

            # The point prediction is deterministic. MC dropout is used only
            # as a seeded uncertainty gate and never as the delivered value.
            self.model.eval()
            with torch.no_grad():
                raw_prediction = float(self.model(input_tensor).item())
            std_dev = self._deterministic_uncertainty(input_tensor)
            self._log_reason(
                f"Seeded uncertainty estimate generated (std dev: {std_dev:.4f})",
                "uncertainty_quantification",
                {"std_dev": std_dev, "seed": int(self.settings["mc_seed"])},
            )

            if not math.isfinite(raw_prediction) or not math.isfinite(std_dev):
                return self._fallback(data, "non_finite_model_output")
            if std_dev > self.settings['uncertainty_threshold']:
                result = self._fallback(data, "uncertainty_above_fixed_threshold")
                result["uncertainty"] = std_dev
                return result

            scaled_candidate = raw_prediction * float(self.settings["model_output_scale_units"])
            max_candidate = max(0.0, float(self.settings["max_model_candidate_units"]))
            total_insulin_delivered = min(max_candidate, max(0.0, scaled_candidate))
            self._log_reason(
                "Loaded LSTM candidate accepted after deterministic scaling and hard cap",
                "lstm_prediction",
                total_insulin_delivered,
            )

            self.state['last_prediction'] = total_insulin_delivered
            self.state['raw_prediction'] = raw_prediction
            self.state['uncertainty'] = std_dev

            return {
                "total_insulin_delivered": total_insulin_delivered,
                "predicted_insulin_raw": raw_prediction,
                "uncertainty": std_dev,
                "fallback_triggered": False,
                "model_checkpoint_loaded": True,
                "uncertainty_seed": int(self.settings["mc_seed"]),
                "model_output_scale_units": float(self.settings["model_output_scale_units"]),
                "hard_cap_units_per_step": max_candidate,
            }

        def __str__(self):
            return (f"Hybrid LSTM/Rule-Based Algorithm:\n"
                    f"  Model Path: {self.settings['model_path']}\n"
                    f"  MC Samples: {self.settings['mc_samples']}\n"
                    f"  Uncertainty Threshold: {self.settings['uncertainty_threshold']}")
else:
    class LSTMInsulinAlgorithm(InsulinAlgorithm):  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Torch is required for LSTMInsulinAlgorithm. Install with `pip install iints[torch]`.")

        def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
            raise ImportError("Torch is required for LSTMInsulinAlgorithm. Install with `pip install iints[torch]`.")
