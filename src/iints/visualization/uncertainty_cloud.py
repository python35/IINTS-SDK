#!/usr/bin/env python3
"""
Uncertainty Cloud Visualizer - IINTS-AF
Creates visualization of AI confidence as shadow around glucose predictions.

The Uncertainty Cloud provides:
- Visual representation of AI confidence
- Confidence intervals around glucose predictions
- Color-coded uncertainty levels
- "Why" annotations for key decisions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UncertaintyData:
    """Data structure for uncertainty visualization"""
    timestamps: np.ndarray
    glucose_values: np.ndarray
    predictions: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    confidence_scores: np.ndarray  # 0.0 to 1.0
    decision_points: Optional[List[Dict]] = None
    
    def __post_init__(self):
        if self.decision_points is None:
            self.decision_points = []


@dataclass
class VisualizationConfig:
    """Configuration for uncertainty cloud visualization"""
    # Figure settings
    figure_size: Tuple[int, int] = (14, 8)
    dpi: int = 150
    style: str = 'default'
    
    # Glucose target zones
    target_low: float = 70.0
    target_high: float = 180.0
    tight_low: float = 70.0
    tight_high: float = 140.0
    
    # Colors
    glucose_color: str = '#2196F3'
    prediction_color: str = '#4CAF50'
    uncertainty_color: str = '#81C784'  # Light green
    cloud_alpha: float = 0.3
    
    # Thresholds for coloring
    critical_low: float = 54.0
    critical_high: float = 250.0
    
    # Annotation settings
    show_annotations: bool = True
    annotation_fontsize: int = 8


class UncertaintyCloud:
    """
    Creates uncertainty cloud visualizations for glucose predictions.
    
    The cloud represents AI confidence through:
    1. Semi-transparent shaded regions showing prediction bounds
    2. Color intensity based on confidence score
    3. Annotation markers for key decision points
    
    Usage:
        cloud = UncertaintyCloud()
        
        # Create data
        data = UncertaintyData(
            timestamps=np.arange(0, 480, 5),
            glucose_values=glucose,
            predictions=predictions,
            lower_bounds=lower,
            upper_bounds=upper,
            confidence_scores=confidence
        )
        
        # Generate plot
        fig = cloud.plot(data)
        plt.savefig("uncertainty_cloud.png")
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize uncertainty cloud visualizer.
        
        Args:
            config: Visualization configuration (uses default if None)
        """
        self.config = config or VisualizationConfig()
        self.style_applied = False
        
    def _apply_style(self):
        """Apply matplotlib style settings"""
        if not self.style_applied:
            plt.style.use(self.config.style)
            self.style_applied = True
    
    def _create_target_zones(self, ax: plt.Axes, time_range: Tuple[float, float]):
        """Create shaded regions for target glucose zones"""
        # Very tight range (70-140) - green
        ax.axhspan(
            self.config.tight_low, self.config.tight_high,
            xmin=0, xmax=1,
            alpha=0.15, color='green', 
            label='Tight Range (70-140)'
        )
        
        # Standard target range (70-180) - yellow
        ax.axhspan(
            self.config.target_low, self.config.target_high,
            xmin=0, xmax=1,
            alpha=0.1, color='orange',
            label='Target Range (70-180)'
        )
        
        # Critical zones
        ax.axhspan(
            0, self.config.critical_low,
            xmin=0, xmax=1,
            alpha=0.3, color='red',
            label='Critical Low (<54)'
        )
        
        ax.axhspan(
            self.config.critical_high, 400,
            xmin=0, xmax=1,
            alpha=0.3, color='red',
            label='Critical High (>250)'
        )
    
    def _plot_uncertainty_cloud(self, 
                                ax: plt.Axes, 
                                data: UncertaintyData) -> plt.Axes:
        """Plot the uncertainty cloud as filled region"""
        # Calculate cloud properties based on confidence
        # Higher confidence = narrower cloud, darker color
        
        alpha_values = self.config.cloud_alpha * (1 - data.confidence_scores * 0.5)
        
        # Plot multiple layers for visual depth
        for i in range(len(data.timestamps) - 1):
            # Main cloud (prediction bounds)
            polygon = plt.Polygon(
                [
                    (data.timestamps[i], data.lower_bounds[i]),
                    (data.timestamps[i+1], data.lower_bounds[i+1]),
                    (data.timestamps[i+1], data.upper_bounds[i+1]),
                    (data.timestamps[i], data.upper_bounds[i])
                ],
                alpha=alpha_values[i] if i < len(alpha_values) else self.config.cloud_alpha,
                facecolor=self.config.uncertainty_color,
                edgecolor='none'
            )
            ax.add_patch(polygon)
        
        # Center line (prediction)
        ax.plot(
            data.timestamps, data.predictions,
            color=self.config.prediction_color,
            linewidth=2,
            linestyle='--',
            label='AI Prediction',
            zorder=3
        )
        
        return ax
    
    def _plot_glucose_line(self, 
                           ax: plt.Axes, 
                           data: UncertaintyData,
                           time_range: Tuple[float, float]) -> plt.Axes:
        """Plot actual glucose values"""
        # Color glucose line based on values
        colors = []
        for g in data.glucose_values:
            if g < self.config.critical_low:
                colors.append('darkred')
            elif g < self.config.target_low:
                colors.append('red')
            elif g > self.config.critical_high:
                colors.append('darkred')
            elif g > self.config.target_high:
                colors.append('orange')
            else:
                colors.append(self.config.glucose_color)
        
        ax.plot(
            data.timestamps, data.glucose_values,
            color=self.config.glucose_color,
            linewidth=2,
            label='Actual Glucose',
            zorder=4
        )
        
        # Fill below line with gradient
        ax.fill_between(
            data.timestamps,
            0,
            data.glucose_values,
            alpha=0.1,
            color=self.config.glucose_color
        )
        
        return ax
    
    def _add_annotations(self, 
                         ax: plt.Axes, 
                         data: UncertaintyData,
                         time_range: Tuple[float, float]):
        """Add decision point annotations"""
        if not self.config.show_annotations or not data.decision_points:
            return
        
        # Find annotation-worthy points
        for point in data.decision_points:
            timestamp = point.get('timestamp', 0)
            glucose = point.get('glucose', 0)
            reason = point.get('reason', '')
            uncertainty = point.get('uncertainty', 0.5)
            
            # Skip if outside time range
            if timestamp < time_range[0] or timestamp > time_range[1]:
                continue
            
            # Only annotate significant events
            if abs(uncertainty) > 0.3 or glucose < 70 or glucose > 200:
                # Create annotation text
                if glucose < 70:
                    text = f" Low: {glucose:.0f}"
                elif glucose > 200:
                    text = f" High: {glucose:.0f}"
                else:
                    text = f"Conf: {1-uncertainty:.0%}"
                
                ax.annotate(
                    text,
                    xy=(timestamp, glucose),
                    xytext=(timestamp, glucose + 30),
                    fontsize=self.config.annotation_fontsize,
                    ha='center',
                    arrowprops=dict(
                        arrowstyle='->',
                        color='gray',
                        lw=0.5
                    ),
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        alpha=0.8
                    ),
                    zorder=5
                )
    
    def _format_axes(self, 
                     ax: plt.Axes, 
                     data: UncertaintyData,
                     time_range: Tuple[float, float]):
        """Format plot axes"""
        # Set limits
        ax.set_xlim(time_range)
        ax.set_ylim(40, 350)
        
        # Labels
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Glucose (mg/dL)', fontsize=12)
        
        # Title
        ax.set_title(
            'Glucose Prediction with Uncertainty Cloud',
            fontsize=14,
            fontweight='bold'
        )
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Y-axis reference lines
        ax.axhline(y=120, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=180, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=70, color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        # X-axis ticks (every 60 minutes)
        ax.set_xticks(np.arange(0, data.timestamps.max() + 60, 60))
        
        # Legend
        ax.legend(loc='upper right', fontsize=9)
    
    def _add_confidence_legend(self, 
                               fig: plt.Figure,
                               ax: plt.Axes):
        """Add confidence level legend"""
        # Create legend for uncertainty
        legend_elements = [
            mpatches.Patch(
                color=self.config.uncertainty_color,
                alpha=0.5,
                label='High Confidence (>80%)'
            ),
            mpatches.Patch(
                color=self.config.uncertainty_color,
                alpha=0.2,
                label='Low Confidence (<50%)'
            )
        ]
        
        ax.legend(
            handles=legend_elements,
            loc='lower right',
            fontsize=9,
            title='Uncertainty Level'
        )
    
    def plot(self, 
             data: UncertaintyData,
             time_range: Optional[Tuple[float, float]] = None,
             save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate uncertainty cloud visualization.
        
        Args:
            data: UncertaintyData with all required arrays
            time_range: Optional (min_time, max_time) to display
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        self._apply_style()
        
        # Determine time range
        if time_range is None:
            time_range = (data.timestamps.min(), data.timestamps.max())
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=self.config.figure_size,
            dpi=self.config.dpi
        )
        
        # Create target zones
        self._create_target_zones(ax, time_range)
        
        # Plot uncertainty cloud
        self._plot_uncertainty_cloud(ax, data)
        
        # Plot actual glucose
        self._plot_glucose_line(ax, data, time_range)
        
        # Add annotations
        self._add_annotations(ax, data, time_range)
        
        # Format axes
        self._format_axes(ax, data, time_range)
        
        # Add confidence legend
        self._add_confidence_legend(fig, ax)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f" Saved uncertainty cloud to: {save_path}")
        
        return fig
    
    def plot_comparison(self,
                        data_list: List[Tuple[str, UncertaintyData]],
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple uncertainty clouds for comparison.
        
        Args:
            data_list: List of (name, data) tuples
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        self._apply_style()
        
        # Create figure with subplots
        n_plots = len(data_list)
        fig, axes = plt.subplots(
            n_plots, 1,
            figsize=(self.config.figure_size[0], 
                     self.config.figure_size[1] * n_plots / 2),
            dpi=self.config.dpi
        )
        
        if n_plots == 1:
            axes = [axes]
        
        time_range = (data_list[0][1].timestamps.min(), 
                      data_list[0][1].timestamps.max())
        
        for idx, (name, data) in enumerate(data_list):
            ax = axes[idx]
            
            # Target zones
            self._create_target_zones(ax, time_range)
            
            # Uncertainty cloud
            self._plot_uncertainty_cloud(ax, data)
            
            # Glucose line
            self._plot_glucose_line(ax, data, time_range)
            
            # Format
            ax.set_ylabel(f'{name}\nGlucose (mg/dL)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(time_range)
            ax.set_ylim(40, 350)
            
            # Calculate average confidence
            avg_conf = data.confidence_scores.mean()
            ax.set_title(f'{name} - Avg Confidence: {avg_conf:.0%}', fontsize=11)
        
        axes[-1].set_xlabel('Time (minutes)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def create_dashboard_widget(self,
                                data: UncertaintyData,
                                width: int = 400,
                                height: int = 200) -> plt.Figure:
        """
        Create a compact widget for dashboard display.
        
        Args:
            data: UncertaintyData
            width: Widget width in pixels
            height: Widget height in pixels
            
        Returns:
            matplotlib Figure
        """
        self._apply_style()
        
        # Create compact figure
        fig, ax = plt.subplots(
            figsize=(width / 100, height / 100),
            dpi=100
        )
        
        # Simplified visualization
        ax.fill_between(
            data.timestamps,
            data.lower_bounds,
            data.upper_bounds,
            alpha=0.3,
            color=self.config.uncertainty_color,
            label='Uncertainty'
        )
        
        ax.plot(
            data.timestamps, data.glucose_values,
            color=self.config.glucose_color,
            linewidth=1.5
        )
        
        ax.plot(
            data.timestamps, data.predictions,
            color=self.config.prediction_color,
            linewidth=1,
            linestyle='--',
            alpha=0.7
        )
        
        # Target zone
        ax.axhspan(70, 180, alpha=0.1, color='green')
        
        # Formatting
        ax.set_xlim(data.timestamps.min(), data.timestamps.max())
        ax.set_ylim(40, 300)
        ax.set_xticks([])
        ax.set_yticks([70, 120, 180, 250])
        ax.set_yticklabels(['70', '120', '180', '250'], fontsize=8)
        ax.grid(True, alpha=0.2)
        
        # Confidence indicator
        avg_conf = data.confidence_scores.mean()
        conf_color = 'green' if avg_conf > 0.7 else 'orange' if avg_conf > 0.5 else 'red'
        ax.text(
            0.98, 0.95,
            f'Conf: {avg_conf:.0%}',
            transform=ax.transAxes,
            fontsize=8,
            ha='right',
            color=conf_color,
            fontweight='bold'
        )
        
        return fig


def generate_sample_data() -> UncertaintyData:
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_points = 97  # 8 hours at 5-min intervals
    
    timestamps = np.arange(0, n_points * 5, 5)
    
    # Simulate glucose with realistic patterns
    base_glucose = 120 + 30 * np.sin(timestamps / (24 * 12 / (2 * np.pi)))
    glucose = base_glucose + np.random.normal(0, 10, n_points)
    glucose = np.clip(glucose, 40, 350)
    
    # Predictions (slightly ahead of actual)
    predictions = np.roll(glucose, 1)
    predictions[0] = glucose[0]
    predictions = predictions + np.random.normal(0, 5, n_points)
    
    # Confidence scores (lower during meals/changes)
    confidence = 0.9 - 0.3 * np.abs(np.gradient(glucose)) / 10
    confidence = np.clip(confidence, 0.4, 0.95)
    
    # Uncertainty bounds
    uncertainty = 1 - confidence
    lower_bounds = predictions - 20 * (1 + uncertainty)
    upper_bounds = predictions + 20 * (1 + uncertainty)
    lower_bounds = np.clip(lower_bounds, 20, 350)
    upper_bounds = np.clip(upper_bounds, 20, 350)
    
    # Decision points
    decision_points = [
        {'timestamp': 100, 'glucose': 170, 'reason': 'High glucose correction', 'uncertainty': 0.2},
        {'timestamp': 250, 'glucose': 65, 'reason': 'Low glucose detected', 'uncertainty': 0.1},
    ]
    
    return UncertaintyData(
        timestamps=timestamps,
        glucose_values=glucose,
        predictions=predictions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        confidence_scores=confidence,
        decision_points=decision_points
    )


def demo_uncertainty_cloud():
    """Demonstrate uncertainty cloud visualization"""
    print("=" * 70)
    print("UNCERTAINTY CLOUD VISUALIZATION DEMONSTRATION")
    print("=" * 70)
    
    # Generate sample data
    print("\n Generating sample data...")
    data = generate_sample_data()
    
    print(f"  Data points: {len(data.timestamps)}")
    print(f"  Time range: {data.timestamps.min()}-{data.timestamps.max()} minutes")
    print(f"  Avg confidence: {data.confidence_scores.mean():.1%}")
    
    # Create visualizer
    cloud = UncertaintyCloud()
    
    # Generate main plot
    print("\n Generating main uncertainty cloud...")
    fig = cloud.plot(data, save_path="results/visualization/uncertainty_cloud_main.png")
    
    # Generate comparison plot
    print("\n Generating comparison visualization...")
    
    # Create second dataset (improved algorithm)
    data2 = generate_sample_data()
    data2.predictions = data2.predictions * 0.95  # Better predictions
    data2.confidence_scores = np.clip(data2.confidence_scores + 0.05, 0, 1)
    
    comparison_fig = cloud.plot_comparison(
        [("Original Algorithm", data), ("Improved Algorithm", data2)],
        save_path="results/visualization/uncertainty_cloud_comparison.png"
    )
    
    # Generate dashboard widget
    print("\n Generating dashboard widget...")
    widget = cloud.create_dashboard_widget(data)
    widget.savefig("results/visualization/uncertainty_widget.png", 
                   dpi=100, bbox_inches='tight')
    
    print("\n Visualization files saved:")
    print("  - results/visualization/uncertainty_cloud_main.png")
    print("  - results/visualization/uncertainty_cloud_comparison.png")
    print("  - results/visualization/uncertainty_widget.png")
    
    print("\n" + "=" * 70)
    print("UNCERTAINTY CLOUD DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_uncertainty_cloud()

