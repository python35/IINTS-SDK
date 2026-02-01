import torch
import numpy as np
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput
from .lstm_algorithm import LSTMInsulinAlgorithm
from .correction_bolus import CorrectionBolus

class HybridInsulinAlgorithm(InsulinAlgorithm):
    """Hybrid algorithm that switches between LSTM and rule-based based on uncertainty."""
    
    def __init__(self, uncertainty_threshold=0.15, mc_samples=30):
        super().__init__()
        self.lstm_algo = LSTMInsulinAlgorithm()
        self.rule_algo = CorrectionBolus()
        self.uncertainty_threshold = uncertainty_threshold
        self.mc_samples = mc_samples
        self.switch_count = 0
        self.lstm_count = 0
        
    def calculate_uncertainty(self, data: AlgorithmInput):
        """Calculate uncertainty using MC Dropout."""
        if not hasattr(self.lstm_algo, 'model') or self.lstm_algo.model is None:
            return 1.0  # High uncertainty if model not loaded
            
        # Use the same input format as LSTM algorithm
        placeholder_input = [3, data.current_glucose, 72, 29, 32, 0.47, 33]
        input_tensor = torch.tensor(placeholder_input, dtype=torch.float32).reshape(1, 1, 7)
        
        self.lstm_algo.model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.lstm_algo.model(input_tensor).item()
                predictions.append(pred)
        
        self.lstm_algo.model.eval()  # Disable dropout
        return np.std(predictions) / (np.mean(predictions) + 1e-8)  # Coefficient of variation
    
    def predict_insulin(self, data: AlgorithmInput):
        self.why_log = [] # Clear the log for this prediction cycle

        uncertainty = self.calculate_uncertainty(data)
        self._log_reason(f"Calculated uncertainty: {uncertainty:.4f}", "uncertainty_quantification", uncertainty)
        
        if uncertainty > self.uncertainty_threshold:
            self.switch_count += 1
            self._log_reason(f"Uncertainty ({uncertainty:.4f}) exceeds threshold ({self.uncertainty_threshold:.4f}). Switching to Rule-Based Algorithm.", "decision_switch", "Rule-Based")
            insulin_output = self.rule_algo.predict_insulin(data)
            insulin_output["uncertainty"] = uncertainty
            # Append child algorithm's why_log to hybrid's why_log
            self.why_log.extend(self.rule_algo.get_why_log())
            return insulin_output
        else:
            self.lstm_count += 1
            self._log_reason(f"Uncertainty ({uncertainty:.4f}) is within threshold ({self.uncertainty_threshold:.4f}). Using LSTM Algorithm.", "decision_switch", "LSTM")
            insulin_output = self.lstm_algo.predict_insulin(data)
            insulin_output["uncertainty"] = uncertainty
            # Append child algorithm's why_log to hybrid's why_log
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