try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput
from .lstm_algorithm import LSTMInsulinAlgorithm
from .correction_bolus import CorrectionBolus

if _TORCH_AVAILABLE:
    class HybridInsulinAlgorithm(InsulinAlgorithm):
        """Hybrid algorithm that switches between LSTM and rule-based based on uncertainty."""
        
        def __init__(self, uncertainty_threshold=0.15, mc_samples=30):
            super().__init__()
            self.lstm_algo = LSTMInsulinAlgorithm(
                settings={
                    "uncertainty_threshold": uncertainty_threshold,
                    "mc_samples": mc_samples,
                }
            )
            self.rule_algo = CorrectionBolus()
            self.uncertainty_threshold = uncertainty_threshold
            self.mc_samples = mc_samples
            self.switch_count = 0
            self.lstm_count = 0
            
        def calculate_uncertainty(self, data: AlgorithmInput):
            """Calculate seeded uncertainty using the LSTM's documented inputs."""
            if not self.lstm_algo.model_loaded:
                return float("inf")
            input_tensor = self.lstm_algo._feature_tensor(data)
            return self.lstm_algo._deterministic_uncertainty(input_tensor)
        
        def predict_insulin(self, data: AlgorithmInput):
            self.why_log = [] # Clear the log for this prediction cycle

            if not self.lstm_algo.model_loaded:
                self.switch_count += 1
                self._log_reason(
                    "LSTM checkpoint unavailable; switching to deterministic rule-based algorithm",
                    "decision_switch",
                    "Rule-Based",
                )
                insulin_output = self.rule_algo.predict_insulin(data)
                insulin_output["uncertainty"] = None
                insulin_output["fallback_triggered"] = True
                insulin_output["fallback_reason"] = "missing_model_checkpoint"
                self.why_log.extend(self.rule_algo.get_why_log())
                return insulin_output

            insulin_output = self.lstm_algo.predict_insulin(data)
            uncertainty = insulin_output.get("uncertainty")
            self._log_reason(
                f"Seeded uncertainty result: {uncertainty}",
                "uncertainty_quantification",
                uncertainty,
            )
            if bool(insulin_output.get("fallback_triggered")):
                self.switch_count += 1
                self._log_reason(
                    "LSTM safety gate selected deterministic fallback",
                    "decision_switch",
                    "Rule-Based",
                )
            else:
                self.lstm_count += 1
                self._log_reason(
                    "Loaded LSTM candidate passed deterministic gates",
                    "decision_switch",
                    "LSTM",
                )
            self.why_log.extend(self.lstm_algo.get_why_log())
            return insulin_output
        
        def reset(self):
            self.lstm_algo.reset()
            self.rule_algo.reset()
            self.switch_count = 0
            self.lstm_count = 0
        
        def get_state(self):
            """Get current algorithm state."""
            return {
                "switch_count": self.switch_count,
                "lstm_count": self.lstm_count,
                "lstm_usage": self.lstm_count / (self.switch_count + self.lstm_count) if (self.switch_count + self.lstm_count) > 0 else 0
            }
        
        def get_stats(self):
            total = self.switch_count + self.lstm_count
            return {
                "lstm_usage": self.lstm_count / total if total > 0 else 0,
                "rule_usage": self.switch_count / total if total > 0 else 0,
                "total_decisions": total
            }
else:
    class HybridInsulinAlgorithm(InsulinAlgorithm):  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Torch is required for HybridInsulinAlgorithm. Install with `pip install iints[torch]`.")
