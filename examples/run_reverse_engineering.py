#!/usr/bin/env python3
"""
Commercial Algorithm Reverse Engineering - IINTS-AF
Analyze and reverse engineer commercial insulin pump behaviors.
Part of the #WeAreNotWaiting movement for transparent diabetes tech.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
import json

class CommercialPumpEmulator:
    """Emulates known behaviors of commercial insulin pumps."""
    
    # Known commercial pump characteristics (from public documentation/research)
    PUMP_PROFILES = {
        "medtronic_670g": {
            "name": "Medtronic MiniMed 670G (SmartGuard)",
            "target_glucose": 120,
            "suspend_threshold": 70,
            "max_basal_rate": 5.0,
            "correction_factor_range": (30, 80),
            "carb_ratio_range": (8, 20),
            "behavior_notes": "Conservative, tends to suspend frequently"
        },
        "tandem_control_iq": {
            "name": "Tandem t:slim X2 (Control-IQ)",
            "target_glucose": 112.5,
            "suspend_threshold": 70,
            "max_basal_rate": 6.0,
            "correction_factor_range": (25, 70),
            "carb_ratio_range": (6, 25),
            "behavior_notes": "More aggressive corrections, predictive low glucose suspend"
        },
        "omnipod_5": {
            "name": "Omnipod 5",
            "target_glucose": 110,
            "suspend_threshold": 70,
            "max_basal_rate": 5.0,
            "correction_factor_range": (20, 60),
            "carb_ratio_range": (5, 30),
            "behavior_notes": "Adaptive algorithm, learns from user patterns"
        }
    }
    
    def __init__(self, pump_type="medtronic_670g"):
        self.profile = self.PUMP_PROFILES[pump_type]
        self.pump_type = pump_type
        
    def emulate_decision(self, glucose, time_since_meal=0, carbs=0):
        """Emulate commercial pump decision-making."""
        profile = self.profile
        
        # Basic correction logic (simplified)
        target = profile["target_glucose"]
        correction_factor = np.mean(profile["correction_factor_range"])
        
        if glucose < profile["suspend_threshold"]:
            return {"insulin": 0, "action": "suspend", "reason": "Low glucose suspend"}
        
        correction_needed = max(0, glucose - target) / correction_factor
        
        # Pump-specific behaviors
        if self.pump_type == "medtronic_670g":
            # Conservative approach
            correction_needed *= 0.7
        elif self.pump_type == "tandem_control_iq":
            # More aggressive
            correction_needed *= 1.2
        elif self.pump_type == "omnipod_5":
            # Adaptive (simplified)
            correction_needed *= (0.8 + 0.4 * np.random.random())
        
        return {
            "insulin": min(correction_needed, profile["max_basal_rate"]),
            "action": "deliver",
            "reason": f"Correction for BG {glucose}"
        }

class ReverseEngineeringAnalyzer:
    """Analyze differences between our algorithms and commercial pumps."""
    
    def __init__(self):
        self.commercial_emulators = {
            name: CommercialPumpEmulator(name) 
            for name in CommercialPumpEmulator.PUMP_PROFILES.keys()
        }
        
    def compare_algorithms(self, scenario_data, our_algorithm_results):
        """Compare our algorithms against commercial pump emulations."""
        
        results = {
            "scenario": scenario_data,
            "our_algorithms": our_algorithm_results,
            "commercial_emulations": {},
            "analysis": {}
        }
        
        # Run commercial emulations
        for pump_name, emulator in self.commercial_emulators.items():
            commercial_results = []
            
            for _, row in scenario_data.iterrows():
                decision = emulator.emulate_decision(
                    glucose=row['glucose_actual_mgdl'],
                    time_since_meal=row.get('time_since_meal', 0),
                    carbs=row.get('carbs', 0)
                )
                commercial_results.append(decision)
            
            results["commercial_emulations"][pump_name] = commercial_results
        
        # Analyze differences
        results["analysis"] = self._analyze_differences(
            our_algorithm_results, 
            results["commercial_emulations"]
        )
        
        return results
    
    def _analyze_differences(self, our_results, commercial_results):
        """Analyze key differences between algorithms."""
        
        analysis = {
            "insulin_delivery_comparison": {},
            "safety_behavior_comparison": {},
            "aggressiveness_analysis": {},
            "vulnerability_assessment": {}
        }
        
        # Extract insulin delivery patterns
        our_insulin = [r.get('total_insulin_delivered', 0) for r in our_results]
        
        for pump_name, commercial_data in commercial_results.items():
            commercial_insulin = [d['insulin'] for d in commercial_data]
            
            analysis["insulin_delivery_comparison"][pump_name] = {
                "our_total": sum(our_insulin),
                "commercial_total": sum(commercial_insulin),
                "difference_percent": ((sum(our_insulin) - sum(commercial_insulin)) / sum(commercial_insulin) * 100) if sum(commercial_insulin) > 0 else 0,
                "correlation": np.corrcoef(our_insulin, commercial_insulin)[0,1] if len(our_insulin) == len(commercial_insulin) else 0
            }
        
        return analysis
    
    def generate_vulnerability_report(self, comparison_results):
        """Generate report on potential vulnerabilities found."""
        
        vulnerabilities = []
        
        for pump_name, comparison in comparison_results["analysis"]["insulin_delivery_comparison"].items():
            if abs(comparison["difference_percent"]) > 50:
                vulnerabilities.append({
                    "pump": pump_name,
                    "type": "insulin_delivery_mismatch",
                    "severity": "high" if abs(comparison["difference_percent"]) > 100 else "medium",
                    "description": f"Our algorithm delivers {comparison['difference_percent']:.1f}% different insulin than {pump_name}",
                    "potential_impact": "Could indicate over/under-dosing in real scenarios"
                })
        
        return vulnerabilities
    
    def create_comparison_visualization(self, comparison_results, output_file="reverse_engineering_analysis.png"):
        """Create comprehensive comparison visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Commercial Pump Reverse Engineering Analysis', fontsize=16)
        
        # Insulin delivery comparison
        ax1 = axes[0, 0]
        pump_names = list(comparison_results["analysis"]["insulin_delivery_comparison"].keys())
        our_totals = [comparison_results["analysis"]["insulin_delivery_comparison"][p]["our_total"] for p in pump_names]
        commercial_totals = [comparison_results["analysis"]["insulin_delivery_comparison"][p]["commercial_total"] for p in pump_names]
        
        x = np.arange(len(pump_names))
        width = 0.35
        
        ax1.bar(x - width/2, our_totals, width, label='Our Algorithm', alpha=0.8)
        ax1.bar(x + width/2, commercial_totals, width, label='Commercial Pump', alpha=0.8)
        ax1.set_xlabel('Pump Type')
        ax1.set_ylabel('Total Insulin (Units)')
        ax1.set_title('Insulin Delivery Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([p.replace('_', ' ').title() for p in pump_names], rotation=45)
        ax1.legend()
        
        # Difference percentages
        ax2 = axes[0, 1]
        differences = [comparison_results["analysis"]["insulin_delivery_comparison"][p]["difference_percent"] for p in pump_names]
        colors = ['red' if abs(d) > 50 else 'orange' if abs(d) > 25 else 'green' for d in differences]
        
        bars = ax2.bar(pump_names, differences, color=colors, alpha=0.7)
        ax2.set_xlabel('Pump Type')
        ax2.set_ylabel('Difference (%)')
        ax2.set_title('Insulin Delivery Difference from Commercial')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{diff:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Correlation analysis
        ax3 = axes[1, 0]
        correlations = [comparison_results["analysis"]["insulin_delivery_comparison"][p]["correlation"] for p in pump_names]
        
        bars = ax3.bar(pump_names, correlations, alpha=0.7, color='blue')
        ax3.set_xlabel('Pump Type')
        ax3.set_ylabel('Correlation Coefficient')
        ax3.set_title('Algorithm Behavior Correlation')
        ax3.set_ylim(-1, 1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Vulnerability summary
        ax4 = axes[1, 1]
        vulnerabilities = self.generate_vulnerability_report(comparison_results)
        
        if vulnerabilities:
            severity_counts = {}
            for vuln in vulnerabilities:
                severity = vuln['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            colors = {'high': 'red', 'medium': 'orange', 'low': 'yellow'}
            ax4.pie(severity_counts.values(), 
                   labels=severity_counts.keys(),
                   colors=[colors.get(k, 'gray') for k in severity_counts.keys()],
                   autopct='%1.0f%%')
            ax4.set_title('Vulnerability Severity Distribution')
        else:
            ax4.text(0.5, 0.5, 'No Major\nVulnerabilities\nDetected', 
                    ha='center', va='center', transform=ax4.transAxes, 
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax4.set_title('Vulnerability Assessment')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file

def run_reverse_engineering_analysis():
    """Run complete reverse engineering analysis."""
    print("=== Commercial Pump Reverse Engineering Analysis ===")
    print("Analyzing our algorithms against known commercial pump behaviors")
    
    # This would typically use real scenario data
    # For demo, we'll create synthetic data
    scenario_data = pd.DataFrame({
        'time_minutes': range(0, 480, 5),
        'glucose_actual_mgdl': 120 + 30 * np.sin(np.arange(0, 480, 5) / 60) + np.random.normal(0, 5, 96)
    })
    
    # Simulate our algorithm results
    our_results = [
        {'total_insulin_delivered': max(0, (g - 120) / 50)} 
        for g in scenario_data['glucose_actual_mgdl']
    ]
    
    analyzer = ReverseEngineeringAnalyzer()
    comparison = analyzer.compare_algorithms(scenario_data, our_results)
    
    # Generate vulnerability report
    vulnerabilities = analyzer.generate_vulnerability_report(comparison)
    
    print(f"\n=== Analysis Results ===")
    for pump_name, analysis in comparison["analysis"]["insulin_delivery_comparison"].items():
        print(f"\n{pump_name.replace('_', ' ').title()}:")
        print(f"  Insulin difference: {analysis['difference_percent']:.1f}%")
        print(f"  Behavior correlation: {analysis['correlation']:.3f}")
    
    print(f"\n=== Vulnerabilities Found ===")
    if vulnerabilities:
        for vuln in vulnerabilities:
            print(f"  {vuln['severity'].upper()}: {vuln['description']}")
    else:
        print("  No major vulnerabilities detected")
    
    # Create visualization
    viz_file = analyzer.create_comparison_visualization(comparison, 
        os.path.join(os.path.dirname(__file__), "..", "results", "reports", "reverse_engineering_analysis.png"))
    print(f"\nVisualization saved to: {viz_file}")
    
    # Save detailed report
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "reports", "reverse_engineering_report.json"), 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print("\n=== #WeAreNotWaiting Analysis Complete ===")
    print("This analysis helps the diabetes community understand commercial pump behaviors")
    print("and develop safer, more transparent alternatives.")

if __name__ == '__main__':
    run_reverse_engineering_analysis()