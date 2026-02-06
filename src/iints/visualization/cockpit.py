#!/usr/bin/env python3
"""
Clinical Control Center - IINTS-AF
Professional medical dashboard for diabetes algorithm research.

Features:
- Real-time glucose visualization with uncertainty cloud
- Algorithm reasoning log ("Why" panel)
- Multi-algorithm comparison (Battle Mode)
- Legacy pump comparison
- Clinical metrics dashboard
- Alert and safety monitoring

This is not a hobby app - it's a professional medical cockpit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class CockpitConfig:
    """Configuration for the clinical cockpit"""
    # Layout
    figure_size: Tuple[int, int] = (20, 12)
    dpi: int = 150
    
    # Colors
    primary_color: str = '#2196F3'
    secondary_color: str = '#4CAF50'
    alert_color: str = '#F44336'
    warning_color: str = '#FF9800'
    success_color: str = '#4CAF50'
    
    # Target zones
    target_low: float = 70.0
    target_high: float = 180.0
    critical_low: float = 54.0
    critical_high: float = 250.0
    
    # Update interval (for real-time mode)
    update_interval: float = 0.5  # seconds


@dataclass
class DashboardState:
    """Current state of the dashboard"""
    current_time: int = 0
    current_glucose: float = 0.0
    glucose_velocity: float = 0.0
    insulin_delivered: float = 0.0
    iob: float = 0.0
    cob: float = 0.0
    
    # Algorithm state
    algorithm_name: str = ""
    algorithm_confidence: float = 0.0
    prediction: float = 0.0
    uncertainty: float = 0.0
    
    # Safety state
    safety_alerts: List[str] = field(default_factory=list)
    hypo_risk: str = "normal"
    hyper_risk: str = "normal"
    
    # Metrics
    tir: float = 0.0
    cv: float = 0.0
    gmi: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'current_time': self.current_time,
            'current_glucose': self.current_glucose,
            'glucose_velocity': self.glucose_velocity,
            'insulin_delivered': self.insulin_delivered,
            'iob': self.iob,
            'cob': self.cob,
            'algorithm_name': self.algorithm_name,
            'algorithm_confidence': self.algorithm_confidence,
            'prediction': self.prediction,
            'uncertainty': self.uncertainty,
            'safety_alerts': self.safety_alerts,
            'tir': self.tir,
            'cv': self.cv,
            'gmi': self.gmi
        }


class ClinicalCockpit:
    """
    Professional clinical dashboard for diabetes algorithm research.
    
    This dashboard provides:
    1. Main glucose display with target zones
    2. Uncertainty cloud around predictions
    3. Reasoning log ("Why" panel)
    4. Algorithm personality metrics
    5. Safety alerts
    6. Battle mode comparison
    
    Usage:
        cockpit = ClinicalCockpit()
        
        # For real-time updates
        cockpit.update(state)
        
        # For end-to-end visualization
        cockpit.visualize_results(simulation_data)
        
        # For battle mode comparison
        cockpit.compare_battle(battle_report)
    """
    
    def __init__(self, config: Optional[CockpitConfig] = None):
        """
        Initialize clinical cockpit.
        
        Args:
            config: Dashboard configuration (default used if None)
        """
        self.config = config or CockpitConfig()
        self.state = DashboardState()
        self.history: List[Dict[str, Any]] = []
        
    def _create_glucose_panel(self, 
                              gs: GridSpec,
                              simulation_data: pd.DataFrame,
                              predictions: Optional[np.ndarray] = None,
                              ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Create main glucose visualization panel"""
        if ax is None:
            ax = plt.subplot(gs[0, :])
        
        timestamps = simulation_data['time_minutes']
        glucose = simulation_data['glucose_actual_mgdl']
        
        # Target zones
        ax.axhspan(self.config.target_low, self.config.target_high,
                   alpha=0.15, color='green', label='Target (70-180)')
        ax.axhspan(self.config.critical_low, self.config.target_low,
                   alpha=0.2, color='red', label='Low Zone')
        ax.axhspan(self.config.target_high, self.config.critical_high,
                   alpha=0.2, color='orange', label='High Zone')
        
        # Glucose line
        ax.plot(timestamps, glucose, 
                color=self.config.primary_color, 
                linewidth=2, 
                label='Glucose')

        # Safety overrides timeline markers
        if 'safety_triggered' in simulation_data.columns:
            safety_mask = simulation_data['safety_triggered'].astype(bool)
            if safety_mask.any():
                ax.scatter(
                    timestamps[safety_mask],
                    glucose[safety_mask],
                    color='red',
                    s=20,
                    label='Safety Override',
                    zorder=3
                )
        
        # Add predictions if provided
        if predictions is not None:
            ax.plot(timestamps, predictions,
                    color=self.config.secondary_color,
                    linewidth=1.5,
                    linestyle='--',
                    label='Prediction',
                    alpha=0.7)
            
            # Uncertainty band
            if 'uncertainty' in simulation_data.columns:
                uncertainty = np.asarray(simulation_data['uncertainty'], dtype=float)
                lower = predictions - 30 * (1.0 + uncertainty)
                upper = predictions + 30 * (1.0 + uncertainty)
                ax.fill_between(timestamps, lower, upper,
                               alpha=0.2, color=self.config.secondary_color,
                               label='Uncertainty')
        
        # Reference lines
        ax.axhline(y=120, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=self.config.target_low, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=self.config.target_high, color='orange', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_xlim(timestamps.min(), timestamps.max())
        ax.set_ylim(40, 350)
        ax.set_xlabel('Time (minutes)', fontsize=10)
        ax.set_ylabel('Glucose (mg/dL)', fontsize=10)
        ax.set_title('Glucose Monitor', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def _create_reasoning_panel(self,
                                gs: GridSpec,
                                reasoning_logs: List[Dict],
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Create reasoning log ("Why") panel"""
        if ax is None:
            ax = plt.subplot(gs[1, :2])
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.set_title('Reasoning Log', fontsize=12, fontweight='bold', pad=10)
        
        # Display most recent reasoning
        if reasoning_logs:
            latest = reasoning_logs[-1]
            algorithm = latest.get('algorithm', 'Unknown')
            decision = latest.get('decision', 0)
            reasons = latest.get('reasons', ['No reasoning available'])
            
            # Algorithm name
            ax.text(0.5, 9.5, f"Algorithm: {algorithm}", fontsize=11, fontweight='bold')
            
            # Decision
            ax.text(0.5, 8.5, f"Decision: {decision:.2f} units", fontsize=10)
            
            # Primary reason
            ax.text(0.5, 7.5, "Why?", fontsize=10, fontweight='bold', color='blue')
            
            for i, reason in enumerate(reasons[:5]):
                y_pos = 6.5 - i * 0.8
                ax.text(0.7, y_pos, f"• {reason}", fontsize=9)
            
            # Confidence indicator
            confidence = latest.get('confidence', 0.5)
            conf_color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red'
            ax.text(0.5, 2.5, f"AI Confidence: {confidence:.0%}", 
                   fontsize=10, color=conf_color, fontweight='bold')
        else:
            ax.text(0.5, 5, "No decisions recorded", fontsize=10, ha='center')
        
        return ax
    
    def _create_metrics_panel(self,
                              gs: GridSpec,
                              metrics: Dict,
                              ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Create clinical metrics panel"""
        if ax is None:
            ax = plt.subplot(gs[1, 2:4])
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.set_title('Clinical Metrics', fontsize=12, fontweight='bold', pad=10)
        
        # Key metrics
        metrics_to_show = [
            ('TIR (70-180)', f"{metrics.get('tir_70_180', 0):.1f}%"),
            ('TIR (70-140)', f"{metrics.get('tir_70_140', 0):.1f}%"),
            ('Time <70', f"{metrics.get('tir_below_70', 0):.1f}%"),
            ('Time >180', f"{metrics.get('tir_above_180', 0):.1f}%"),
            ('CV', f"{metrics.get('cv', 0):.1f}%"),
            ('GMI', f"{metrics.get('gmi', 0):.1f}%"),
            ('LBGI', f"{metrics.get('lbgi', 0):.2f}"),
            ('Total Insulin', f"{metrics.get('total_insulin', 0):.1f} U"),
        ]
        
        for i, (name, value) in enumerate(metrics_to_show):
            ax.text(0.5, 8.5 - i * 0.9, f"{name}:", fontsize=9, fontweight='bold')
            
            # Color code TIR
            if 'TIR' in name and '70-180' in name:
                val = float(value.replace('%', ''))
                color = 'green' if val > 70 else 'orange' if val > 50 else 'red'
            elif 'CV' in name:
                val = float(value.replace('%', ''))
                color = 'green' if val < 36 else 'orange' if val < 50 else 'red'
            else:
                color = 'black'
            
            ax.text(4.5, 8.5 - i * 0.9, value, fontsize=9, color=color)
        
        return ax
    
    def _create_safety_panel(self,
                             gs: GridSpec,
                             safety_timeline: List[Dict[str, Any]],
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Create safety alerts panel with timeline"""
        if ax is None:
            ax = plt.subplot(gs[1, 4:6])
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.set_title('Safety Monitor', fontsize=12, fontweight='bold', pad=10)

        if safety_timeline:
            recent = safety_timeline[-6:]
            for i, entry in enumerate(recent):
                y_pos = 8.5 - i * 1.2
                reason = entry.get('reason', 'UNKNOWN')
                time_min = entry.get('time_min', 0)
                marker = '🔴' if 'HYPO' in reason or 'EMERGENCY' in reason else '🟡'
                ax.text(0.5, y_pos, f"{marker} t={time_min:.0f}m {reason}", fontsize=8)
            ax.text(0.5, 1.0, f"Overrides: {len(safety_timeline)}", fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 5, "✅ No active alerts", fontsize=10, ha='center', color='green')
        
        return ax
    
    def _create_algorithm_personality_panel(self,
                                            gs: GridSpec,
                                            personality: Dict,
                                            ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Create algorithm personality panel"""
        if ax is None:
            ax = plt.subplot(gs[2, :2])
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.set_title('Algorithm Personality', fontsize=12, fontweight='bold', pad=10)
        
        name = personality.get('name', 'Unknown')
        p = personality.get('personality', {})
        
        ax.text(0.5, 9.0, name, fontsize=11, fontweight='bold')
        
        traits = [
            ('Aggressiveness', p.get('aggressiveness', 'Unknown')),
            ('Hypo Aversion', p.get('hypo_aversion', 'Unknown')),
            ('Response Speed', p.get('response_speed', 'Unknown')),
            ('Correction Style', p.get('correction_aggressiveness', 'Unknown')),
        ]
        
        for i, (trait, value) in enumerate(traits):
            ax.text(0.5, 7.5 - i * 1.0, f"{trait}:", fontsize=9, fontweight='bold')
            ax.text(4.0, 7.5 - i * 1.0, value, fontsize=9)
        
        return ax
    
    def _create_insulin_panel(self,
                              gs: GridSpec,
                              simulation_data: pd.DataFrame,
                              ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Create insulin delivery panel"""
        if ax is None:
            ax = plt.subplot(gs[2, 2:4])
        
        timestamps = simulation_data['time_minutes']
        insulin = simulation_data['delivered_insulin_units']
        if 'patient_iob_units' in simulation_data.columns:
            iob = simulation_data['patient_iob_units']
        else:
            iob = pd.Series(np.zeros(len(timestamps)))
        
        ax.bar(timestamps, insulin, width=4, alpha=0.7, 
               color=self.config.secondary_color, label='Insulin Delivered')
        ax.plot(timestamps, iob, color='purple', linewidth=2, 
                label='Insulin on Board')
        
        ax.set_xlabel('Time (minutes)', fontsize=10)
        ax.set_ylabel('Units', fontsize=10)
        ax.set_title('Insulin Delivery', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def _create_summary_panel(self,
                              gs: GridSpec,
                              summary: Dict,
                              ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Create summary statistics panel"""
        if ax is None:
            ax = plt.subplot(gs[2, 4:6])
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.set_title('Summary', fontsize=12, fontweight='bold', pad=10)
        
        summary_text = [
            f"Duration: {summary.get('duration_hours', 0):.1f} hours",
            f"Data Points: {summary.get('data_points', 0)}",
            f"Decisions: {summary.get('decisions', 0)}",
            f"Hypo Events: {summary.get('hypo_events', 0)}",
            f"Hyper Events: {summary.get('hyper_events', 0)}",
            f"Avg Glucose: {summary.get('mean_glucose', 0):.0f} mg/dL",
            f"Glucose Range: {summary.get('min_glucose', 0):.0f}-{summary.get('max_glucose', 0):.0f}",
        ]
        
        for i, line in enumerate(summary_text):
            ax.text(0.5, 8.5 - i * 1.0, line, fontsize=9)
        
        return ax
    
    def visualize_results(self,
                          simulation_data: pd.DataFrame,
                          predictions: Optional[np.ndarray] = None,
                          reasoning_logs: Optional[List[Dict]] = None,
                          metrics: Optional[Dict] = None,
                          personality: Optional[Dict] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create complete clinical dashboard visualization.
        
        Args:
            simulation_data: DataFrame with simulation results
            predictions: Optional array of glucose predictions
            reasoning_logs: Optional list of reasoning logs
            metrics: Optional clinical metrics dictionary
            personality: Optional algorithm personality
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        # Create figure
        fig = plt.figure(
            figsize=self.config.figure_size,
            dpi=self.config.dpi
        )
        
        # Create grid layout
        gs = GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.3)
        
        # Create panels
        self._create_glucose_panel(gs, simulation_data, predictions)
        
        reasoning_logs = reasoning_logs or []
        self._create_reasoning_panel(gs, reasoning_logs)
        
        metrics = metrics or {}
        self._create_metrics_panel(gs, metrics)
        
        safety_timeline: List[Dict[str, Any]] = []
        if 'safety_triggered' in simulation_data.columns and 'safety_reason' in simulation_data.columns:
            for _, row in simulation_data.iterrows():
                if bool(row.get('safety_triggered', False)):
                    safety_timeline.append({
                        'time_min': row.get('time_minutes', 0),
                        'reason': row.get('safety_reason', 'UNKNOWN')
                    })
        self._create_safety_panel(gs, safety_timeline)
        
        personality = personality or {}
        self._create_algorithm_personality_panel(gs, personality)
        
        self._create_insulin_panel(gs, simulation_data)
        
        # Calculate summary
        glucose_col = pd.to_numeric(simulation_data['glucose_actual_mgdl'])
        summary = {
            'duration_hours': (simulation_data['time_minutes'].max() - 
                              simulation_data['time_minutes'].min()) / 60,
            'data_points': len(simulation_data),
            'decisions': (simulation_data['delivered_insulin_units'].gt(0)).sum(),
            'hypo_events': ((glucose_col.lt(70.0)) & (glucose_col.diff().lt(0))).sum(),
            'hyper_events': ((glucose_col.gt(250.0)) & (glucose_col.diff().gt(0))).sum(),
            'mean_glucose': glucose_col.mean(),
            'min_glucose': glucose_col.min(),
            'max_glucose': glucose_col.max(),
        }
        self._create_summary_panel(gs, summary)
        
        # Add title
        fig.suptitle('IINTS-AF Clinical Control Center',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f" Dashboard saved to: {save_path}")
        
        return fig
    
    def compare_battle(self,
                       battle_report,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create battle mode comparison dashboard.
        
        Args:
            battle_report: BattleReport object
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(
            figsize=(20, 14),
            dpi=self.config.dpi
        )
        
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # Title
        fig.suptitle(f' Battle Mode: {battle_report.battle_name}\n Winner: {battle_report.winner}',
                    fontsize=14, fontweight='bold')
        
        # Rankings summary
        ax_rankings = plt.subplot(gs[0, :])
        ax_rankings.axis('off')
        
        ranking_text = "RANKINGS\n" + "="*50 + "\n"
        for i, rank in enumerate(battle_report.rankings, 1):
            medal = "1st" if i == 1 else "2nd" if i == 2 else "3rd"
            ranking_text += (
                f"{medal} {i}. {rank['participant']}: "
                f"Score={rank['overall_score']:.3f}, "
                f"TIR={rank['tir']:.1f}%, "
                f"CV={rank['cv']:.1f}%\n"
            )
        
        ax_rankings.text(0.5, 0.5, ranking_text, fontsize=11, 
                        ha='center', va='center', transform=ax_rankings.transAxes,
                        family='monospace')
        
        # Comparison bars
        ax_comparison = plt.subplot(gs[1, :])
        
        algorithms = [r['participant'] for r in battle_report.rankings]
        tir_scores = [r['tir'] for r in battle_report.rankings]
        cv_scores = [r['cv'] for r in battle_report.rankings]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = ax_comparison.bar(x - width/2, tir_scores, width, 
                                  label='TIR (%)', color='#2196F3')
        bars2 = ax_comparison.bar(x + width/2, cv_scores, width,
                                  label='CV (%)', color='#4CAF50')
        
        ax_comparison.set_ylabel('Percentage')
        ax_comparison.set_title('Algorithm Comparison: TIR vs CV')
        ax_comparison.set_xticks(x)
        ax_comparison.set_xticklabels(algorithms)
        ax_comparison.legend()
        ax_comparison.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax_comparison.text(bar.get_x() + bar.get_width()/2., height + 1,
                             f'{height:.1f}%', ha='center', fontsize=9)
        
        # Detailed metrics table
        ax_table = plt.subplot(gs[2, :])
        ax_table.axis('off')
        
        table_data = []
        headers = ['Algorithm', 'TIR', 'TIR Tight', '<70', '>180', 'CV', 'GMI', 'LBGI']
        
        for rank in battle_report.rankings:
            table_data.append([
                rank['participant'],
                f"{rank['tir']:.1f}%",
                f"{rank['tir_tight']:.1f}%",
                f"{rank['time_below_70']:.1f}%",
                f"{rank['time_above_180']:.1f}%",
                f"{rank['cv']:.1f}%",
                f"{rank['gmi']:.1f}%",
                f"{rank['lbgi']:.2f}"
            ])
        
        table = ax_table.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def update(self, state: DashboardState):
        """Update dashboard with new state (for real-time mode)"""
        self.state = state
        self.history.append(state.to_dict())
    
    def export_state(self) -> Dict:
        """Export current dashboard state"""
        return {
            'state': self.state.to_dict(),
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }


def demo_clinical_cockpit():
    """Demonstrate clinical cockpit functionality"""
    print("=" * 70)
    print("CLINICAL CONTROL CENTER DEMONSTRATION")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n_points = 97
    
    timestamps = np.arange(0, n_points * 5, 5)
    glucose = 120 + 30 * np.sin(timestamps / (24 * 12 / (2 * np.pi))) + np.random.normal(0, 15, n_points)
    glucose = np.clip(glucose, 40, 350)
    
    simulation_data = pd.DataFrame({
        'time_minutes': timestamps,
        'glucose_actual_mgdl': glucose,
        'delivered_insulin_units': np.random.uniform(0, 1, n_points),
        'patient_iob_units': np.cumsum(np.random.uniform(0, 0.1, n_points)),
        'uncertainty': np.random.uniform(0.1, 0.4, n_points)
    })
    
    # Create cockpit
    cockpit = ClinicalCockpit()
    
    # Create visualization
    print("\n Generating clinical dashboard...")
    predictions = np.roll(glucose, 1)
    predictions[0] = glucose[0]
    
    reasoning_logs = [
        {
            'algorithm': 'PID Controller',
            'decision': 0.5,
            'reasons': [
                'Glucose elevated at 145 mg/dL',
                'Rising at 1.5 mg/dL/min',
                'No significant IOB'
            ],
            'confidence': 0.85
        }
    ]
    
    metrics = {
        'tir_70_180': 72.5,
        'tir_70_140': 45.2,
        'tir_below_70': 5.1,
        'tir_above_180': 22.4,
        'cv': 32.5,
        'gmi': 6.8,
        'lbgi': 2.1,
        'total_insulin': 45.2
    }
    
    personality = {
        'name': 'PID Controller',
        'personality': {
            'aggressiveness': 'Moderate',
            'hypo_aversion': 'Moderate',
            'response_speed': 'Fast',
            'correction_aggressiveness': 'Moderate'
        }
    }
    
    fig = cockpit.visualize_results(
        simulation_data=simulation_data,
        predictions=predictions,
        reasoning_logs=reasoning_logs,
        metrics=metrics,
        personality=personality,
        save_path="results/visualization/clinical_cockpit.png"
    )
    
    print(" Dashboard saved to: results/visualization/clinical_cockpit.png")
    
    print("\n" + "=" * 70)
    print("CLINICAL CONTROL CENTER DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_clinical_cockpit()
