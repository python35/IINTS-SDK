import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from iints.core.safety.config import (
    CONTROLLER_HYPO_GUARD_MGDL,
    ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS,
)

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


def _safe_torch_load_weights(path: str):
    if torch is None:  # pragma: no cover
        raise ImportError("Torch required for model loading.")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch build does not support secure weights-only loading. "
            "Upgrade torch to a version that supports `weights_only=True`."
        ) from exc


@dataclass
class ClinicalConstraints:
    """Research-sandbox constraints; not clinical dosing guidance."""
    max_glucose_rate = 5.0  # mg/dL per minute (ISPAD guidelines)
    min_glucose = 54  # mg/dL (severe hypoglycemia threshold)
    max_glucose = 400  # mg/dL (DKA threshold)
    max_insulin_bolus = ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS
    target_range = (70, 180)  # mg/dL (ADA guidelines)


if _TORCH_AVAILABLE:
    class ClinicalTeacher:
        """Generate deterministic synthetic rule labels for sandbox tests."""

        def __init__(self, seed: int = 42):
            self.seed = int(seed)
            self.clinical_protocols = {
                'correction_factor': 50,  # mg/dL per unit (adult)
                'carb_ratio': 15,  # grams per unit
                'target_glucose': 120,  # mg/dL
                'hypoglycemia_threshold': 70,
                'hyperglycemia_threshold': 250
            }

        def generate_clinical_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
            """Generate reproducible synthetic labels; never clinical evidence."""
            rng = np.random.default_rng(self.seed)
            X = []
            y = []

            for _ in range(n_samples):
                glucose = float(np.clip(rng.normal(150, 40), 70, 350))
                carbs = float(rng.choice([0, 30, 45, 60, 75]))
                trend = float(rng.uniform(-2.0, 2.0))
                predicted = glucose + trend * 30.0
                iob = float(rng.uniform(0.0, 4.0))

                correction = max(0, (glucose - self.clinical_protocols['target_glucose']) /
                                 self.clinical_protocols['correction_factor'])
                meal_bolus = carbs / self.clinical_protocols['carb_ratio'] if carbs > 0 else 0
                total_insulin = min(
                    correction + meal_bolus,
                    ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS,
                )

                if glucose <= CONTROLLER_HYPO_GUARD_MGDL or predicted <= CONTROLLER_HYPO_GUARD_MGDL:
                    total_insulin = 0

                X.append([glucose, trend, predicted, carbs, iob, 50.0, 15.0])
                y.append(total_insulin)

            return np.array(X), np.array(y)

        def evaluate_clinical_safety(self, predicted_insulin: float, glucose: float) -> float:
            """Evaluate predicted insulin dose against clinical safety criteria."""
            safety_score = 100.0

            expected_correction = max(0, (glucose - self.clinical_protocols['target_glucose']) / 50)

            if abs(predicted_insulin - expected_correction) > expected_correction * 1.0:
                safety_score -= 15

            if glucose < 70 and predicted_insulin > 0.5:
                safety_score -= 20

            if predicted_insulin > ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS or predicted_insulin < 0:
                safety_score -= 50

            return max(0, safety_score)


    class AutonomousLearningSystem:
        """Research-only candidate trainer; it never promotes a model to deployment."""

        def __init__(self, model_path: str):
            self.model_path = model_path
            self.clinical_teacher = ClinicalTeacher()
            self.learning_history: List[Dict[str, Any]] = []
            self.safety_threshold = 70.0

        def continuous_learning_cycle(self, validation_errors: List[Dict]) -> bool:
            """Train a deterministic candidate that still requires external review."""
            print("Starting autonomous learning cycle...")

            torch.manual_seed(42)
            torch.use_deterministic_algorithms(True)

            clinical_X, clinical_y = self.clinical_teacher.generate_clinical_training_data(500)

            from iints.core.algorithms.lstm_algorithm import LSTMModel
            model = LSTMModel(input_size=7, hidden_size=50, output_size=1)

            if os.path.exists(self.model_path):
                model.load_state_dict(_safe_torch_load_weights(self.model_path))

            improved_model = self._clinical_fine_tuning(model, clinical_X, clinical_y)
            safety_score = self._validate_clinical_safety(improved_model, clinical_X, clinical_y)

            if safety_score > self.safety_threshold:
                torch.save(improved_model.state_dict(), self.model_path + '.candidate')
                self.learning_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'safety_score': safety_score,
                    'scenarios_learned': len(validation_errors)
                })
                print(
                    f"Research candidate saved for external validation. "
                    f"Heuristic score: {safety_score:.1f}%"
                )
                return True

            print(f"Model not improved. Safety score: {safety_score:.1f}%")
            return False

        def _clinical_fine_tuning(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
            """Fine-tune model on clinical data."""
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            X_tensor = torch.FloatTensor(X).unsqueeze(1)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)

            for _ in range(50):
                optimizer.zero_grad()
                predictions = model(X_tensor)

                mse_loss = nn.MSELoss()(predictions, y_tensor)
                safety_penalty = self._calculate_safety_penalty(predictions, X_tensor)
                total_loss = mse_loss + 0.5 * safety_penalty

                total_loss.backward()
                optimizer.step()

            model.eval()
            return model

        def _calculate_safety_penalty(self, predictions: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            """Calculate penalty for unsafe predictions."""
            glucose = X[:, 0, 0]
            pred_insulin = predictions[:, 0]
            low_glucose_penalty = torch.relu(pred_insulin - 0.1) * (
                glucose < CONTROLLER_HYPO_GUARD_MGDL
            ).float() * 10.0
            cap_penalty = torch.relu(
                pred_insulin - ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS
            ) * 5.0
            return (low_glucose_penalty + cap_penalty).mean()

        def _validate_clinical_safety(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
            """Validate model against clinical safety criteria."""
            model.eval()
            safety_scores = []

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).unsqueeze(1)
                predictions = model(X_tensor)

                for i in range(len(predictions)):
                    pred_insulin = predictions[i].item()
                    glucose = X[i, 0]

                    safety_score = self.clinical_teacher.evaluate_clinical_safety(pred_insulin, glucose)
                    safety_scores.append(safety_score)

            return float(np.mean(safety_scores))

        def get_learning_report(self) -> Dict:
            """Generate report on autonomous learning progress."""
            if not self.learning_history:
                return {"status": "No learning cycles completed"}

            latest = self.learning_history[-1]
            return {
                "total_learning_cycles": len(self.learning_history),
                "latest_safety_score": latest['safety_score'],
                "latest_timestamp": str(latest['timestamp']),
                "scenarios_learned": latest['scenarios_learned']
            }
else:
    class ClinicalTeacher:  # type: ignore
        def __init__(self):
            raise ImportError("Torch is required for ClinicalTeacher. Install with `pip install iints[torch]`.")

    class AutonomousLearningSystem:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Torch is required for AutonomousLearningSystem. Install with `pip install iints[torch]`.")
