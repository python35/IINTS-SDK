#!/usr/bin/env python3
"""
Explainable AI Audit Trail - IINTS-AF
Clinical decision transparency system for medical AI validation
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

class ClinicalAuditTrail:
    """Explainable AI system for clinical decision transparency"""
    
    def __init__(self):
        self.audit_log = []
        self.decision_context = {}
        
    def log_decision(self, timestamp, glucose_current, glucose_trend, insulin_decision, 
                    algorithm_confidence, safety_override=False, context=None):
        """Log AI decision with clinical reasoning"""
        
        # Calculate glucose velocity (mg/dL per minute)
        glucose_velocity = self._calculate_velocity(glucose_trend)
        
        # Generate clinical reasoning
        reasoning = self._generate_clinical_reasoning(
            glucose_current, glucose_velocity, insulin_decision, 
            algorithm_confidence, safety_override, context
        )
        
        # Create audit entry
        audit_entry = {
            'timestamp': timestamp,
            'glucose_mg_dL': glucose_current,
            'glucose_velocity_per_min': glucose_velocity,
            'insulin_decision_U': insulin_decision,
            'algorithm_confidence': algorithm_confidence,
            'safety_override': safety_override,
            'clinical_reasoning': reasoning,
            'risk_assessment': self._assess_immediate_risk(glucose_current, glucose_velocity),
            'decision_category': self._categorize_decision(insulin_decision, glucose_current)
        }
        
        self.audit_log.append(audit_entry)
        return audit_entry
    
    def _calculate_velocity(self, glucose_trend):
        """Calculate glucose rate of change"""
        if not glucose_trend or len(glucose_trend) < 2:
            return 0.0
        
        # Use last 3 readings for velocity calculation (15 minutes)
        recent_readings = glucose_trend[-3:]
        if len(recent_readings) < 2:
            return 0.0
        
        # Calculate slope (mg/dL per 5-minute interval)
        time_intervals = len(recent_readings) - 1
        glucose_change = recent_readings[-1] - recent_readings[0]
        velocity_per_5min = glucose_change / time_intervals
        
        # Convert to per-minute
        return velocity_per_5min / 5.0
    
    def _generate_clinical_reasoning(self, glucose, velocity, insulin, confidence, override, context):
        """Generate human-readable clinical reasoning"""
        
        reasoning_parts = []
        
        # Glucose status assessment
        if glucose < 70:
            reasoning_parts.append(f"Hypoglycemia detected ({glucose:.1f} mg/dL)")
        elif glucose > 180:
            reasoning_parts.append(f"Hyperglycemia detected ({glucose:.1f} mg/dL)")
        else:
            reasoning_parts.append(f"Glucose in target range ({glucose:.1f} mg/dL)")
        
        # Trend analysis
        if abs(velocity) > 2.0:
            direction = "rising" if velocity > 0 else "falling"
            reasoning_parts.append(f"Rapid glucose {direction} at {abs(velocity):.1f} mg/dL/min")
        elif abs(velocity) > 1.0:
            direction = "increasing" if velocity > 0 else "decreasing"
            reasoning_parts.append(f"Moderate glucose {direction} trend")
        else:
            reasoning_parts.append("Stable glucose trend")
        
        # Insulin decision reasoning
        if insulin > 0:
            if glucose > 150 and velocity > 1.0:
                reasoning_parts.append(f"Corrective bolus {insulin:.2f}U for hyperglycemia with rising trend")
            elif glucose > 180:
                reasoning_parts.append(f"Correction bolus {insulin:.2f}U for hyperglycemia")
            else:
                reasoning_parts.append(f"Preventive insulin {insulin:.2f}U based on predictive model")
        elif insulin < 0:
            reasoning_parts.append(f"Basal reduction {abs(insulin):.2f}U to prevent hypoglycemia")
        else:
            reasoning_parts.append("No insulin adjustment - maintaining current therapy")
        
        # Confidence assessment
        if confidence < 0.7:
            reasoning_parts.append(f"Low AI confidence ({confidence:.2f}) - conservative approach")
        elif confidence > 0.9:
            reasoning_parts.append(f"High AI confidence ({confidence:.2f}) - optimal conditions")
        
        # Safety override explanation
        if override:
            reasoning_parts.append("SAFETY OVERRIDE: Decision modified by clinical safety supervisor")
        
        # Context-specific reasoning
        if context:
            if context.get('meal_detected'):
                reasoning_parts.append("Meal bolus component included")
            if context.get('exercise_detected'):
                reasoning_parts.append("Exercise adjustment applied")
            if context.get('sensor_noise'):
                reasoning_parts.append("Sensor reliability considered")
        
        return ". ".join(reasoning_parts) + "."
    
    def _assess_immediate_risk(self, glucose, velocity):
        """Assess immediate clinical risk"""
        
        # Predict glucose in 30 minutes
        predicted_glucose = glucose + (velocity * 30)
        
        if glucose < 54 or predicted_glucose < 54:
            return "CRITICAL - Severe hypoglycemia risk"
        elif glucose < 70 or predicted_glucose < 70:
            return "HIGH - Hypoglycemia risk"
        elif glucose > 300 or predicted_glucose > 300:
            return "HIGH - Severe hyperglycemia risk"
        elif glucose > 250 or predicted_glucose > 250:
            return "MODERATE - Hyperglycemia risk"
        elif 70 <= predicted_glucose <= 180:
            return "LOW - Target range maintained"
        else:
            return "MODERATE - Glucose excursion predicted"
    
    def _categorize_decision(self, insulin, glucose):
        """Categorize the type of clinical decision"""
        
        if insulin > 1.0:
            return "CORRECTIVE_BOLUS"
        elif insulin > 0.1:
            return "MICRO_BOLUS"
        elif insulin < -0.1:
            return "BASAL_REDUCTION"
        elif glucose < 70:
            return "HYPOGLYCEMIA_MANAGEMENT"
        elif glucose > 180:
            return "HYPERGLYCEMIA_MANAGEMENT"
        else:
            return "MAINTENANCE_THERAPY"
    
    def generate_clinical_summary(self, hours=24):
        """Generate clinical summary for specified time period"""
        
        if not self.audit_log:
            return "No clinical decisions recorded"
        
        recent_entries = self.audit_log[-int(hours * 12):]  # 12 entries per hour (5-min intervals)
        
        # Count decision types
        decision_counts = {}
        risk_levels = {}
        total_insulin = 0
        
        for entry in recent_entries:
            decision_type = entry['decision_category']
            risk_level = entry['risk_assessment'].split(' - ')[0]
            
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            total_insulin += entry['insulin_decision_U']
        
        # Generate summary
        summary = f"Clinical Decision Summary ({hours}h period):\n"
        summary += f"Total insulin delivered: {total_insulin:.2f}U\n"
        summary += f"Decision types: {dict(decision_counts)}\n"
        summary += f"Risk distribution: {dict(risk_levels)}\n"
        
        return summary
    
    def export_audit_trail(self, filepath):
        """Export audit trail for clinical review"""
        
        audit_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_decisions': len(self.audit_log),
            'audit_entries': self.audit_log,
            'clinical_summary': self.generate_clinical_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2, default=str)
        
        return filepath

def main():
    """Test explainable AI audit trail"""
    audit = ClinicalAuditTrail()
    
    # Simulate some clinical decisions
    test_scenarios = [
        (120, [115, 118, 120], 0.2, 0.85, False, None),  # Stable
        (185, [170, 178, 185], 1.5, 0.92, False, {'meal_detected': True}),  # Rising
        (65, [75, 70, 65], -0.8, 0.78, True, None),  # Falling with override
    ]
    
    print("Explainable AI Clinical Audit Trail")
    print("=" * 50)
    
    for i, (glucose, trend, insulin, confidence, override, context) in enumerate(test_scenarios):
        timestamp = datetime.now() + timedelta(minutes=i*5)
        
        entry = audit.log_decision(
            timestamp, glucose, trend, insulin, confidence, override, context
        )
        
        print(f"\nDecision {i+1}:")
        print(f"Time: {timestamp.strftime('%H:%M')}")
        print(f"Clinical Reasoning: {entry['clinical_reasoning']}")
        print(f"Risk Assessment: {entry['risk_assessment']}")
        print(f"Decision Category: {entry['decision_category']}")
    
    print(f"\n{audit.generate_clinical_summary(1)}")

if __name__ == "__main__":
    main()