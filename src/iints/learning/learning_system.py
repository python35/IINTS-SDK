#!/usr/bin/env python3
"""
IINTS-AF Legacy Learning Storage
Stores externally validated parameters; mock learning is intentionally disabled.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class LearningSystem:
    """Store externally produced research parameters with validation metadata."""
    
    def __init__(self):
        self.models_dir = Path("models/learned_parameters")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.learning_history = []
        
    def save_learned_parameters(self, patient_id: str, parameters: Dict, performance_metrics: Dict):
        """Save patient-specific learned parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        learning_data = {
            "patient_id": patient_id,
            "timestamp": timestamp,
            "learned_parameters": parameters,
            "performance_metrics": performance_metrics,
            "learning_session": len(self.learning_history) + 1
        }
        
        # Save to file
        filename = f"patient_{patient_id}_learned_{timestamp}.json"
        filepath = self.models_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(learning_data, f, indent=2)
            
        self.learning_history.append(learning_data)
        return str(filepath)
    
    def load_learned_parameters(self, patient_id: str) -> Optional[Dict]:
        """Load most recent learned parameters for patient"""
        pattern = f"patient_{patient_id}_learned_*.json"
        files = list(self.models_dir.glob(pattern))
        
        if not files:
            return None
            
        # Get most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def simulate_learning_process(self, patient_id: str, glucose_data: List[float]) -> Tuple[Dict, List[float]]:
        """Reject the former random mock so it cannot create research claims."""

        raise RuntimeError(
            "Random mock learning is disabled. Use the deterministic calibration engine or "
            "the versioned glucose-model training pipeline with measured validation data."
        )
    
    def validate_learning_safety(self, parameters: Dict, patient_id: str) -> Tuple[bool, str]:
        """Safety validation of learned parameters"""
        
        weights = parameters.get("neural_weights", {})
        
        # Safety thresholds
        if weights.get("insulin_sensitivity", 1.0) > 2.0:
            return False, "Learning rejected: Insulin sensitivity exceeds safety threshold"
        
        if weights.get("basal_rate", 1.0) > 3.0:
            return False, "Learning rejected: Basal rate adaptation too aggressive"
            
        if parameters.get("final_loss", 1.0) > 0.5:
            return False, "Learning rejected: Model convergence insufficient"
        
        return True, "Learning validated: All safety constraints satisfied"
    
    def get_learning_status(self, patient_id: str) -> str:
        """Get learning status for patient"""
        learned_data = self.load_learned_parameters(patient_id)
        
        if not learned_data:
            return "Status: Base model (No patient-specific learning)"
        
        timestamp = learned_data["timestamp"]
        session = learned_data["learning_session"]
        convergence = learned_data["learned_parameters"].get("convergence_achieved", False)
        
        if convergence:
            return f"Status: Model optimized for Patient {patient_id} (Learning session {session} - {timestamp})"
        else:
            return f"Status: Learning in progress for Patient {patient_id} (Session {session})"
