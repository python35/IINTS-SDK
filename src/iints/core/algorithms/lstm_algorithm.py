try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False
import os
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput
from .correction_bolus import CorrectionBolus

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
            }
            self.settings = {**self.default_settings, **(settings or {})}

            self.model = LSTMModel(
                self.settings["input_features"],
                self.settings["hidden_size"],
                self.settings["output_size"],
                self.settings["dropout_prob"]
            )
            
            # Load the trained model if it exists
            if os.path.exists(self.settings['model_path']):
                print(f"Loading trained model from {self.settings['model_path']}")
                self.model.load_state_dict(torch.load(self.settings['model_path'], weights_only=True))
            else:
                print(f"Warning: Trained model not found at {self.settings['model_path']}. LSTM will make random predictions.")

            # Instantiate fallback algorithm
            self.fallback_algo = CorrectionBolus()
            self.reset()

        def reset(self):
            """Resets the algorithm's internal state."""
            super().reset()

        def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
            self.why_log = [] # Clear the log for this prediction cycle

            placeholder_input = [3, data.current_glucose, 72, 29, 32, 0.47, 33]
            input_tensor = torch.tensor(placeholder_input, dtype=torch.float32).reshape(1, 1, self.settings['input_features'])

            self._log_reason("LSTM input tensor created", "data_preparation", input_tensor.tolist())

            # --- Monte Carlo Dropout ---
            self.model.train() # Enable dropout
            predictions = []
            with torch.no_grad():
                for _ in range(self.settings['mc_samples']):
                    pred = self.model(input_tensor).item()
                    predictions.append(pred)
            self.model.eval() # Disable dropout for future use if any

            mc_predictions_array = np.array(predictions)
            mean_prediction = np.mean(mc_predictions_array)
            std_dev = np.std(mc_predictions_array)
            self._log_reason(f"Monte Carlo Dropout predictions generated (mean: {mean_prediction:.4f}, std dev: {std_dev:.4f})", "uncertainty_quantification", {'mean': mean_prediction, 'std_dev': std_dev})


            # --- Hybrid Safety Controller ---
            if std_dev > self.settings['uncertainty_threshold']:
                self._log_reason(f"High uncertainty detected ({std_dev:.3f} > {self.settings['uncertainty_threshold']}). Falling back to rule-based algorithm.", "safety_fallback", std_dev)
                fallback_result = self.fallback_algo.predict_insulin(data)
                fallback_result['uncertainty'] = std_dev
                fallback_result['fallback_triggered'] = True
                # Extend fallback algo's log
                self.why_log.extend(self.fallback_algo.get_why_log())
                return fallback_result

            total_insulin_delivered = max(0.0, mean_prediction * 10) # Arbitrary scaling for demo
            self._log_reason(f"LSTM prediction accepted (uncertainty: {std_dev:.4f}). Delivered insulin scaled from raw prediction.", "lstm_prediction", total_insulin_delivered)

            self.state['last_prediction'] = total_insulin_delivered
            self.state['raw_prediction'] = mean_prediction
            self.state['uncertainty'] = std_dev

            return {
                "total_insulin_delivered": total_insulin_delivered,
                "predicted_insulin_raw": mean_prediction,
                "uncertainty": std_dev,
                "fallback_triggered": False
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
