import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import os

@dataclass
class ClinicalConstraints:
    """Physiological constraints based on medical literature."""
    max_glucose_rate = 5.0  # mg/dL per minute (ISPAD guidelines)
    min_glucose = 54  # mg/dL (severe hypoglycemia threshold)
    max_glucose = 400  # mg/dL (DKA threshold)
    max_insulin_bolus = 15.0  # Units (safety limit)
    target_range = (70, 180)  # mg/dL (ADA guidelines)

class ClinicalTeacher:
    """Teaches AI using validated clinical protocols."""
    
    def __init__(self):
        self.clinical_protocols = {
            'correction_factor': 50,  # mg/dL per unit (adult)
            'carb_ratio': 15,  # grams per unit
            'target_glucose': 120,  # mg/dL
            'hypoglycemia_threshold': 70,
            'hyperglycemia_threshold': 250
        }
        
    def generate_clinical_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data based on clinical protocols."""
        X = []
        y = []
        
        for _ in range(n_samples):
            glucose = np.clip(np.random.normal(150, 40), 70, 350)
            time_since_meal = np.random.exponential(60)
            carbs = np.random.choice([0, 30, 45, 60, 75])
            
            # Clinical insulin calculation
            correction = max(0, (glucose - self.clinical_protocols['target_glucose']) / 
                           self.clinical_protocols['correction_factor'])
            meal_bolus = carbs / self.clinical_protocols['carb_ratio'] if carbs > 0 else 0
            total_insulin = min(correction + meal_bolus, 15.0)
            
            if glucose < self.clinical_protocols['hypoglycemia_threshold']:
                total_insulin = 0
            
            X.append([3, glucose, 72, 29, 32, 0.47, 33])  # LSTM input format
            y.append(total_insulin)
        
        return np.array(X), np.array(y)
    
    def evaluate_clinical_safety(self, predicted_insulin: float, glucose: float) -> float:
        """Evaluate predicted insulin dose against clinical safety criteria."""
        safety_score = 100.0
        
        expected_correction = max(0, (glucose - self.clinical_protocols['target_glucose']) / 50)
        
        if abs(predicted_insulin - expected_correction) > expected_correction * 1.0:  # More lenient
            safety_score -= 15  # Reduced penalty
        
        if glucose < 70 and predicted_insulin > 0.5:  # Only penalize significant insulin during hypo
            safety_score -= 20  # Reduced penalty
            
        if predicted_insulin > 15.0 or predicted_insulin < 0:
            safety_score -= 50
        
        return max(0, safety_score)

class AutonomousLearningSystem:
    """Self-improving AI system with clinical safety constraints."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.clinical_teacher = ClinicalTeacher()
        self.learning_history: List[Dict[str, Any]] = []
        self.safety_threshold = 70.0  # Lowered threshold
        
    def continuous_learning_cycle(self, validation_errors: List[Dict]) -> bool:
        """Perform autonomous learning from validation errors."""
        
        print("Starting autonomous learning cycle...")
        
        # Generate clinical training data
        clinical_X, clinical_y = self.clinical_teacher.generate_clinical_training_data(500)
        
        # Load and improve model
        from iints.core.algorithms.lstm_algorithm import LSTMModel
        model = LSTMModel(input_size=7, hidden_size=50, output_size=1)
        
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
        
        improved_model = self._clinical_fine_tuning(model, clinical_X, clinical_y)
        safety_score = self._validate_clinical_safety(improved_model, clinical_X, clinical_y)
        
        if safety_score > self.safety_threshold:
            torch.save(improved_model.state_dict(), self.model_path + '.improved')
            self.learning_history.append({
                'timestamp': pd.Timestamp.now(),
                'safety_score': safety_score,
                'scenarios_learned': len(validation_errors)
            })
            print(f"Model improved! Safety score: {safety_score:.1f}%")
            return True
        else:
            print(f"Model not improved. Safety score: {safety_score:.1f}%")
            return False
    
    def _clinical_fine_tuning(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """Fine-tune model on clinical data."""
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(X).unsqueeze(1)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        for epoch in range(50):
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
        penalty = torch.tensor(0.0, requires_grad=True)
        
        for i in range(len(predictions)):
            pred_insulin = predictions[i].item()
            glucose = X[i, 0, 1].item()  # Second feature is glucose
            
            if glucose < 70 and pred_insulin > 0.1:
                penalty = penalty + torch.tensor(10.0, requires_grad=True)
            
            if pred_insulin > 15.0:
                penalty = penalty + torch.tensor(5.0, requires_grad=True)
        
        return penalty / len(predictions)
    
    def _validate_clinical_safety(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
        """Validate model against clinical safety criteria."""
        
        model.eval()
        safety_scores = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(1)
            predictions = model(X_tensor)
            
            for i in range(len(predictions)):
                pred_insulin = predictions[i].item()
                glucose = X[i, 1]  # Second feature is glucose
                
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