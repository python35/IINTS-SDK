#!/usr/bin/env python3
"""
Algorithm X-Ray - IINTS-AF
Make invisible medical decisions visible through decision replay and what-if analysis
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from iints.data.quality_checker import QualityReport

@dataclass
class DecisionPoint:
    """Single decision point in algorithm timeline"""
    timestamp: datetime
    glucose_mgdl: float
    glucose_velocity: float
    insulin_on_board: float
    carbs_on_board: float
    
    # Decision components
    decision: float
    confidence: float
    reasoning: List[str]
    safety_constraints: List[str]
    risk_level: str
    
    # What-if scenarios
    alternative_decisions: Dict[str, float]
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'glucose_mgdl': self.glucose_mgdl,
            'glucose_velocity': self.glucose_velocity,
            'insulin_on_board': self.insulin_on_board,
            'carbs_on_board': self.carbs_on_board,
            'decision': self.decision,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'safety_constraints': self.safety_constraints,
            'risk_level': self.risk_level,
            'alternative_decisions': self.alternative_decisions
        }

class AlgorithmXRay:
    """X-ray vision into algorithm decision-making process"""
    
    def __init__(self, quality_report: Optional[QualityReport] = None):
        self.decision_timeline: List[DecisionPoint] = []
        self.personality_profile: Dict[str, Any] = {}
        self.quality_report = quality_report

    def get_quality_report_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of the data quality report if available.
        """
        if self.quality_report:
            return {
                "overall_score": self.quality_report.overall_score,
                "summary": self.quality_report.summary,
                "gaps_detected": len(self.quality_report.gaps),
                "anomalies_detected": len(self.quality_report.anomalies),
                "warnings": self.quality_report.warnings
            }
        return {}
        
    def analyze_decision(self, 
                        glucose_mgdl: float,
                        glucose_history: List[float],
                        insulin_history: List[float],
                        time_minutes: int) -> DecisionPoint:
        """
        Deep analysis of single decision point with full reasoning chain
        """
        
        # Calculate glucose velocity
        if len(glucose_history) >= 2:
            glucose_velocity = (glucose_history[-1] - glucose_history[-2]) / 5.0  # mg/dL per minute
        else:
            glucose_velocity = 0.0
        
        # Estimate insulin on board (simplified)
        insulin_on_board = sum(insulin_history[-6:]) * 0.5 if insulin_history else 0.0
        
        # Estimate carbs on board (simplified - would need meal data)
        carbs_on_board = 0.0
        
        # Build reasoning chain
        reasoning = []
        safety_constraints = []
        risk_level = "NORMAL"
        
        # Glucose level reasoning
        if glucose_mgdl < 70:
            reasoning.append(f"Glucose critically low at {glucose_mgdl:.1f} mg/dL")
            risk_level = "HIGH"
        elif glucose_mgdl < 80:
            reasoning.append(f"Glucose approaching low threshold at {glucose_mgdl:.1f} mg/dL")
            risk_level = "MODERATE"
        elif glucose_mgdl > 180:
            reasoning.append(f"Glucose elevated at {glucose_mgdl:.1f} mg/dL")
            risk_level = "MODERATE"
        elif glucose_mgdl > 250:
            reasoning.append(f"Glucose critically high at {glucose_mgdl:.1f} mg/dL")
            risk_level = "HIGH"
        else:
            reasoning.append(f"Glucose in target range at {glucose_mgdl:.1f} mg/dL")
        
        # Velocity reasoning
        if abs(glucose_velocity) > 2.0:
            direction = "rising" if glucose_velocity > 0 else "falling"
            reasoning.append(f"Glucose {direction} rapidly at {abs(glucose_velocity):.2f} mg/dL/min")
            if glucose_velocity < -2.0 and glucose_mgdl < 100:
                safety_constraints.append("Rapid fall near hypoglycemia - insulin blocked")
        elif abs(glucose_velocity) > 1.0:
            direction = "rising" if glucose_velocity > 0 else "falling"
            reasoning.append(f"Glucose {direction} moderately at {abs(glucose_velocity):.2f} mg/dL/min")
        else:
            reasoning.append("Glucose stable")
        
        # Insulin on board reasoning
        if insulin_on_board > 2.0:
            reasoning.append(f"High insulin on board: {insulin_on_board:.2f} units")
            safety_constraints.append("Insulin stacking prevention active")
        elif insulin_on_board > 0.5:
            reasoning.append(f"Moderate insulin on board: {insulin_on_board:.2f} units")
        
        # Calculate base decision
        error = glucose_mgdl - 120  # Target 120 mg/dL
        base_decision = max(0, error * 0.02)
        
        # Apply safety constraints
        final_decision = base_decision
        confidence = 0.8
        
        if glucose_mgdl < 70 or (glucose_velocity < -2.0 and glucose_mgdl < 100):
            final_decision = 0.0
            confidence = 0.95
            safety_constraints.append("Safety supervisor override: insulin delivery blocked")
        elif insulin_on_board > 2.0:
            final_decision = base_decision * 0.5
            confidence = 0.7
            safety_constraints.append("Insulin dose reduced due to stacking risk")
        
        # Generate what-if scenarios
        alternative_decisions = {
            'if_exercised_30min_ago': final_decision * 0.6,
            'if_meal_detected': final_decision * 1.3,
            'if_stress_detected': final_decision * 1.2,
            'if_sensor_noise_high': final_decision * 0.8,
            'aggressive_mode': base_decision * 1.5,
            'conservative_mode': base_decision * 0.5
        }
        
        decision_point = DecisionPoint(
            timestamp=datetime.now(),
            glucose_mgdl=glucose_mgdl,
            glucose_velocity=glucose_velocity,
            insulin_on_board=insulin_on_board,
            carbs_on_board=carbs_on_board,
            decision=final_decision,
            confidence=confidence,
            reasoning=reasoning,
            safety_constraints=safety_constraints,
            risk_level=risk_level,
            alternative_decisions=alternative_decisions
        )
        
        self.decision_timeline.append(decision_point)
        return decision_point
    
    def calculate_personality_profile(self, decision_history: List[DecisionPoint]) -> Dict:
        """
        Calculate algorithm personality traits from decision history
        """
        
        if not decision_history:
            return {}
        
        # Hypo-aversion: how much does it avoid low glucose
        hypo_decisions = [d for d in decision_history if d.glucose_mgdl < 80]
        hypo_aversion = np.mean([1.0 - d.decision for d in hypo_decisions]) if hypo_decisions else 0.5
        
        # Reaction speed: how quickly does it respond to changes
        velocity_responses = [abs(d.decision - 0.5) for d in decision_history if abs(d.glucose_velocity) > 1.0]
        reaction_speed = np.mean(velocity_responses) if velocity_responses else 0.5
        
        # Correction intensity: how aggressive are corrections
        high_glucose = [d for d in decision_history if d.glucose_mgdl > 180]
        correction_intensity = np.mean([d.decision for d in high_glucose]) if high_glucose else 0.5
        
        # Consistency: how stable are decisions in similar conditions
        decision_values = [d.decision for d in decision_history]
        consistency = 1.0 - (np.std(decision_values) if len(decision_values) > 1 else 0.5)
        
        # Safety-first: how often safety constraints are triggered
        safety_triggers = sum(1 for d in decision_history if d.safety_constraints)
        safety_first = safety_triggers / len(decision_history) if decision_history else 0.0
        
        personality = {
            'hypo_aversion': float(hypo_aversion),
            'reaction_speed': float(reaction_speed),
            'correction_intensity': float(correction_intensity),
            'consistency': float(consistency),
            'safety_first': float(safety_first),
            'decision_count': len(decision_history),
            'risk_distribution': {
                'high': sum(1 for d in decision_history if d.risk_level == 'HIGH'),
                'moderate': sum(1 for d in decision_history if d.risk_level == 'MODERATE'),
                'normal': sum(1 for d in decision_history if d.risk_level == 'NORMAL')
            }
        }
        
        self.personality_profile = personality
        return personality
    
    def generate_decision_replay(self, 
                                start_index: int = 0,
                                end_index: Optional[int] = None) -> Dict:
        """
        Generate interactive decision replay with full reasoning
        """
        
        if end_index is None:
            end_index = len(self.decision_timeline)
        
        replay_segment = self.decision_timeline[start_index:end_index]
        
        replay_data = {
            'timeline': [d.to_dict() for d in replay_segment],
            'personality_profile': self.personality_profile,
            'summary': {
                'total_decisions': len(replay_segment),
                'safety_overrides': sum(1 for d in replay_segment if d.safety_constraints),
                'high_risk_moments': sum(1 for d in replay_segment if d.risk_level == 'HIGH'),
                'average_confidence': np.mean([d.confidence for d in replay_segment]),
                'glucose_range': {
                    'min': min(d.glucose_mgdl for d in replay_segment),
                    'max': max(d.glucose_mgdl for d in replay_segment),
                    'mean': np.mean([d.glucose_mgdl for d in replay_segment])
                }
            }
        }
        
        return replay_data
    
    def compare_what_if_scenarios(self, decision_point: DecisionPoint) -> Dict:
        """
        Compare actual decision with what-if alternatives
        """
        
        actual = decision_point.decision
        alternatives = decision_point.alternative_decisions
        
        comparison = {
            'actual_decision': actual,
            'alternatives': alternatives,
            'differences': {
                scenario: {
                    'absolute_diff': alt - actual,
                    'percent_diff': ((alt - actual) / actual * 100) if actual > 0 else 0,
                    'clinical_impact': self._estimate_clinical_impact(actual, alt, decision_point.glucose_mgdl)
                }
                for scenario, alt in alternatives.items()
            }
        }
        
        return comparison
    
    def _estimate_clinical_impact(self, actual: float, alternative: float, glucose: float) -> str:
        """Estimate clinical impact of alternative decision"""
        
        diff = alternative - actual
        
        if abs(diff) < 0.1:
            return "Minimal impact"
        elif glucose < 70:
            if diff < 0:
                return "Safer - reduces hypo risk"
            else:
                return "Riskier - increases hypo risk"
        elif glucose > 180:
            if diff > 0:
                return "More aggressive correction"
            else:
                return "More conservative approach"
        else:
            return "Moderate impact on glucose trajectory"
    
    def export_xray_report(self, filepath: str):
        """Export complete X-ray analysis to JSON"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_decisions': len(self.decision_timeline),
            'personality_profile': self.personality_profile,
            'decision_timeline': [d.to_dict() for d in self.decision_timeline],
            'summary_statistics': self._calculate_summary_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath
    
    def _calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics across all decisions"""
        
        if not self.decision_timeline:
            return {}
        
        return {
            'average_decision': np.mean([d.decision for d in self.decision_timeline]),
            'decision_std': np.std([d.decision for d in self.decision_timeline]),
            'average_confidence': np.mean([d.confidence for d in self.decision_timeline]),
            'safety_override_rate': sum(1 for d in self.decision_timeline if d.safety_constraints) / len(self.decision_timeline),
            'high_risk_rate': sum(1 for d in self.decision_timeline if d.risk_level == 'HIGH') / len(self.decision_timeline),
            'glucose_stats': {
                'mean': np.mean([d.glucose_mgdl for d in self.decision_timeline]),
                'std': np.std([d.glucose_mgdl for d in self.decision_timeline]),
                'min': min(d.glucose_mgdl for d in self.decision_timeline),
                'max': max(d.glucose_mgdl for d in self.decision_timeline)
            }
        }

def main():
    """Demonstration of Algorithm X-Ray system"""
    
    print(" ALGORITHM X-RAY DEMONSTRATION")
    print("=" * 50)
    print("Making invisible medical decisions visible\n")
    
    xray = AlgorithmXRay()
    
    # Simulate decision sequence
    glucose_history = [120, 125, 135, 150, 165, 175, 180, 185, 190, 185]
    insulin_history = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5]
    
    print("Analyzing decision sequence...\n")
    
    for i, glucose in enumerate(glucose_history):
        decision = xray.analyze_decision(
            glucose_mgdl=glucose,
            glucose_history=glucose_history[:i+1],
            insulin_history=insulin_history[:i+1],
            time_minutes=i * 5
        )
        
        print(f"Decision Point {i+1}:")
        print(f"  Glucose: {decision.glucose_mgdl:.1f} mg/dL")
        print(f"  Velocity: {decision.glucose_velocity:.2f} mg/dL/min")
        print(f"  Decision: {decision.decision:.2f} units")
        print(f"  Confidence: {decision.confidence:.1%}")
        print(f"  Risk Level: {decision.risk_level}")
        print(f"  Reasoning:")
        for reason in decision.reasoning:
            print(f"    - {reason}")
        if decision.safety_constraints:
            print(f"  Safety Constraints:")
            for constraint in decision.safety_constraints:
                print(f"      {constraint}")
        print()
    
    # Calculate personality
    print("\n ALGORITHM PERSONALITY PROFILE")
    print("=" * 50)
    personality = xray.calculate_personality_profile(xray.decision_timeline)
    
    print(f"Hypo-Aversion:        {personality['hypo_aversion']:.2f} (0=aggressive, 1=cautious)")
    print(f"Reaction Speed:       {personality['reaction_speed']:.2f} (0=slow, 1=fast)")
    print(f"Correction Intensity: {personality['correction_intensity']:.2f} (0=gentle, 1=aggressive)")
    print(f"Consistency:          {personality['consistency']:.2f} (0=variable, 1=stable)")
    print(f"Safety-First:         {personality['safety_first']:.2f} (0=permissive, 1=strict)")
    
    # Export report
    from pathlib import Path
    results_dir = Path("results/algorithm_xray")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = xray.export_xray_report(str(results_dir / "xray_report.json"))
    print(f"\n X-Ray report exported to: {report_path}")

if __name__ == "__main__":
    main()
