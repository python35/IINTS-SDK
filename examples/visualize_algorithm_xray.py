#!/usr/bin/env python3
"""
Algorithm X-Ray Visualizer - IINTS-AF
Interactive visualization of algorithm decision-making with personality profiling
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iints.analysis.algorithm_xray import AlgorithmXRay
from iints.data.quality_checker import QualityReport, DataGap, DataAnomaly

class XRayVisualizer:
    """Visualize algorithm X-ray analysis"""
    
    def __init__(self, xray: AlgorithmXRay):
        self.xray = xray
        
    def create_decision_replay_visualization(self, save_path: str = None):
        """Create comprehensive decision replay visualization"""
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Algorithm X-Ray: Decision Replay Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Create grid layout (5 rows to accommodate data quality summary)
        gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)
        
        # 1. Glucose timeline with decisions
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_glucose_timeline(ax1)
        
        # 2. Decision confidence over time
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_confidence_timeline(ax2)
        
        # 3. Risk level distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_risk_distribution(ax3)
        
        # 4. Safety override events
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_safety_overrides(ax4)
        
        # 5. Algorithm personality radar
        ax5 = fig.add_subplot(gs[2, :2], projection='polar')
        self._plot_personality_radar(ax5)
        
        # 6. What-if scenario comparison
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_whatif_scenarios(ax6)
        
        # 7. Decision reasoning panel
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_decision_reasoning(ax7)
        
        # 8. Data Quality Summary (new subplot)
        ax8 = fig.add_subplot(gs[4, :])
        self._plot_data_quality_summary(ax8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
    def _plot_glucose_timeline(self, ax):
        """Plot glucose levels with decision markers"""
        
        timeline = self.xray.decision_timeline
        if not timeline:
            return
        
        time_points = range(len(timeline))
        glucose_values = [d.glucose_mgdl for d in timeline]
        decisions = [d.decision for d in timeline]
        
        # Plot glucose
        ax.plot(time_points, glucose_values, 'b-', linewidth=2, label='Glucose')
        
        # Add zone boundaries
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Hypo Threshold')
        ax.axhline(y=180, color='orange', linestyle='--', alpha=0.7, label='Hyper Threshold')
        ax.fill_between(time_points, 70, 180, alpha=0.1, color='green', label='Target Range')
        
        # Mark decisions with size proportional to insulin dose
        decision_sizes = [d * 100 + 20 for d in decisions]
        colors = ['red' if d.risk_level == 'HIGH' else 'orange' if d.risk_level == 'MODERATE' else 'green' 
                 for d in timeline]
        
        ax.scatter(time_points, glucose_values, s=decision_sizes, c=colors, alpha=0.6, 
                  edgecolors='black', linewidth=1, zorder=5)
        
        # Mark safety overrides
        override_points = [i for i, d in enumerate(timeline) if d.safety_constraints]
        if override_points:
            override_glucose = [timeline[i].glucose_mgdl for i in override_points]
            ax.scatter(override_points, override_glucose, marker='X', s=200, 
                      c='red', edgecolors='black', linewidth=2, zorder=10, label='Safety Override')
        
        ax.set_title('Glucose Timeline with Decision Points')
        ax.set_xlabel('Time Points (5-min intervals)')
        ax.set_ylabel('Glucose (mg/dL)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(40, 250)
        
    def _plot_confidence_timeline(self, ax):
        """Plot decision confidence over time"""
        
        timeline = self.xray.decision_timeline
        if not timeline:
            return
        
        time_points = range(len(timeline))
        confidence_values = [d.confidence for d in timeline]
        
        ax.plot(time_points, confidence_values, 'g-', linewidth=2)
        ax.fill_between(time_points, confidence_values, alpha=0.3, color='green')
        
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='High Confidence')
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Medium Confidence')
        
        ax.set_title('AI Decision Confidence')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_risk_distribution(self, ax):
        """Plot risk level distribution"""
        
        timeline = self.xray.decision_timeline
        if not timeline:
            return
        
        risk_counts = {
            'HIGH': sum(1 for d in timeline if d.risk_level == 'HIGH'),
            'MODERATE': sum(1 for d in timeline if d.risk_level == 'MODERATE'),
            'NORMAL': sum(1 for d in timeline if d.risk_level == 'NORMAL')
        }
        
        colors = ['red', 'orange', 'green']
        ax.bar(risk_counts.keys(), risk_counts.values(), color=colors, alpha=0.7)
        
        # Add percentages
        total = len(timeline)
        for i, (risk, count) in enumerate(risk_counts.items()):
            percentage = (count / total * 100) if total > 0 else 0
            ax.text(i, count + 0.5, f'{percentage:.1f}%', ha='center', fontweight='bold')
        
        ax.set_title('Risk Level Distribution')
        ax.set_ylabel('Number of Decisions')
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_safety_overrides(self, ax):
        """Plot safety override statistics"""
        
        timeline = self.xray.decision_timeline
        if not timeline:
            return
        
        total_decisions = len(timeline)
        overrides = sum(1 for d in timeline if d.safety_constraints)
        normal = total_decisions - overrides
        
        sizes = [overrides, normal]
        labels = [f'Safety Override\n({overrides})', f'Normal\n({normal})']
        colors = ['red', 'lightgreen']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 10})
        
        ax.set_title('Safety Supervisor Activity')
        
    def _plot_personality_radar(self, ax):
        """Plot algorithm personality as radar chart"""
        
        personality = self.xray.personality_profile
        if not personality:
            return
        
        categories = ['Hypo-Aversion', 'Reaction\nSpeed', 'Correction\nIntensity', 
                     'Consistency', 'Safety-First']
        values = [
            personality.get('hypo_aversion', 0.5),
            personality.get('reaction_speed', 0.5),
            personality.get('correction_intensity', 0.5),
            personality.get('consistency', 0.5),
            personality.get('safety_first', 0.5)
        ]
        
        # Close the plot
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', label='Algorithm Profile')
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # Add reference circles
        ax.set_ylim(0, 1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=8)
        ax.grid(True)
        
        ax.set_title('Algorithm Personality Profile', pad=20, fontweight='bold')
        
    def _plot_whatif_scenarios(self, ax):
        """Plot what-if scenario comparison"""
        
        timeline = self.xray.decision_timeline
        if not timeline:
            return
        
        # Use last decision point for what-if comparison
        last_decision = timeline[-1]
        actual = last_decision.decision
        alternatives = last_decision.alternative_decisions
        
        scenarios = list(alternatives.keys())
        values = list(alternatives.values())
        
        # Shorten scenario names
        short_names = [s.replace('if_', '').replace('_', ' ').title()[:15] for s in scenarios]
        
        colors = ['red' if v < actual else 'green' if v > actual else 'gray' for v in values]
        bars = ax.barh(short_names, values, color=colors, alpha=0.7)
        
        # Mark actual decision
        ax.axvline(x=actual, color='blue', linestyle='--', linewidth=2, label='Actual Decision')
        
        ax.set_title('What-If Scenarios\n(Last Decision Point)')
        ax.set_xlabel('Insulin Dose (units)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
    def _plot_decision_reasoning(self, ax):
        """Display decision reasoning for key moments"""
        
        timeline = self.xray.decision_timeline
        if not timeline:
            return
        
        ax.axis('off')
        
        # Find most interesting decision (highest risk or safety override)
        interesting_decisions = [d for d in timeline if d.risk_level == 'HIGH' or d.safety_constraints]
        if not interesting_decisions:
            interesting_decisions = timeline[-3:]  # Last 3 decisions
        
        y_position = 0.9
        
        ax.text(0.5, 0.95, 'Key Decision Reasoning', ha='center', fontsize=12, 
               fontweight='bold', transform=ax.transAxes)
        
        for i, decision in enumerate(interesting_decisions[:3]):  # Show max 3
            # Decision box
            box_y = y_position - (i * 0.3)
            
            decision_text = f"Time Point {timeline.index(decision) + 1}: {decision.glucose_mgdl:.1f} mg/dL"
            ax.text(0.05, box_y, decision_text, fontsize=10, fontweight='bold', 
                   transform=ax.transAxes)
            
            # Reasoning
            reasoning_y = box_y - 0.05
            for j, reason in enumerate(decision.reasoning[:3]):  # Max 3 reasons
                ax.text(0.08, reasoning_y - (j * 0.04), f"• {reason}", fontsize=8, 
                       transform=ax.transAxes)
            
            # Safety constraints
            if decision.safety_constraints:
                constraint_y = reasoning_y - (len(decision.reasoning[:3]) * 0.04) - 0.02
                ax.text(0.08, constraint_y, f"{decision.safety_constraints[0]}", 
                       fontsize=8, color='red', fontweight='bold', transform=ax.transAxes)
    
    def _plot_data_quality_summary(self, ax):
        """Display summary of the data quality report"""
        ax.axis('off')
        ax.set_title('Dataset Reliability', fontsize=12, fontweight='bold', pad=10)

        quality_summary = self.xray.get_quality_report_summary()
        
        if quality_summary:
            overall_score = quality_summary.get("overall_score", 0.0)
            summary_text = quality_summary.get("summary", "No summary available")
            gaps_detected = quality_summary.get("gaps_detected", 0)
            anomalies_detected = quality_summary.get("anomalies_detected", 0)
            warnings = quality_summary.get("warnings", [])

            # Color code the overall score
            score_color = 'green'
            if overall_score < 0.75:
                score_color = 'orange'
            if overall_score < 0.5:
                score_color = 'red'

            score_color = 'green'
            if overall_score < 0.75:
                score_color = 'orange'
            if overall_score < 0.5:
                score_color = 'red'

            ax.text(0.05, 0.8, f"Overall Score: {overall_score:.1%}", 
                   fontsize=10, transform=ax.transAxes, color=score_color)
            ax.text(0.05, 0.65, f"Summary: {summary_text}", fontsize=9, transform=ax.transAxes)
            ax.text(0.05, 0.5, f"Gaps Detected: {gaps_detected}", fontsize=9, transform=ax.transAxes)
            ax.text(0.05, 0.35, f"Anomalies Detected: {anomalies_detected}", fontsize=9, transform=ax.transAxes)

            if warnings:
                ax.text(0.05, 0.2, "Warnings:", fontsize=9, fontweight='bold', transform=ax.transAxes)
                for i, warning in enumerate(warnings[:2]): # Show up to 2 warnings
                    ax.text(0.08, 0.1 - i * 0.05, f"- {warning}", fontsize=8, color='red', transform=ax.transAxes)
        else:
            ax.text(0.05, 0.5, "No data quality report available.", fontsize=10, transform=ax.transAxes)


def main():
    """Demonstration of Algorithm X-Ray system"""
    
    print("ALGORITHM X-RAY VISUALIZATION")
    print("=" * 50)
    print("Making invisible medical decisions visible\n")
    
    # --- Create Mock Quality Report for demonstration ---
    mock_gaps = [
        DataGap(start_time=100, end_time=150, duration_minutes=50, data_points_missing=10, percentage_of_total=5, time_range_description="100-150min")
    ]
    mock_anomalies = [
        DataAnomaly(index=5, timestamp=25, value=700, anomaly_type='impossible_value', severity='high', description="Glucose 700 mg/dL above physiological maximum")
    ]
    mock_quality_report = QualityReport(
        overall_score=0.82,
        completeness_score=0.95,
        consistency_score=0.90,
        validity_score=0.80,
        gaps=mock_gaps,
        anomalies=mock_anomalies,
        warnings=["[WARN] DATA GAP DETECTED: 5.0% of data missing (10 points)", "[WARN] CRITICAL ANOMALY: Glucose 700 mg/dL above physiological maximum"],
        summary="Good data quality with minor issues"
    )
    # --- End Mock Quality Report ---

    # Create X-Ray analysis, passing the mock quality report
    xray = AlgorithmXRay(quality_report=mock_quality_report)
    
    # Simulate realistic glucose scenario
    glucose_history = [120, 125, 135, 150, 165, 175, 180, 185, 190, 185, 180, 170, 160, 150, 140, 130, 120, 110, 100, 95]
    insulin_history = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0]
    
    print("Analyzing decision sequence...")
    for i, glucose in enumerate(glucose_history):
        xray.analyze_decision(
            glucose_mgdl=glucose,
            glucose_history=glucose_history[:i+1],
            insulin_history=insulin_history[:i+1],
            time_minutes=i * 5
        )
    
    # Calculate personality
    xray.calculate_personality_profile(xray.decision_timeline)
    
    # Create visualization
    visualizer = XRayVisualizer(xray)
    
    results_dir = Path("results/algorithm_xray")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = results_dir / "xray_visualization.png"
    visualizer.create_decision_replay_visualization(str(save_path))
    
    print(f"\n Visualization complete!")
    print(f"Total decisions analyzed: {len(xray.decision_timeline)}")
    print(f"Safety overrides: {sum(1 for d in xray.decision_timeline if d.safety_constraints)}")

if __name__ == "__main__":
    main()
