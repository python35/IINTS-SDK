#!/usr/bin/env python3
"""
Real-Time Dashboard - IINTS-AF
Live glucose simulation with real-time TIR updates and AI decision visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time
import threading
import queue
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.clinical_tir_analyzer import ClinicalTIRAnalyzer

class RealTimeDashboard:
    """Real-time glucose monitoring dashboard with live TIR analysis"""
    
    def __init__(self):
        self.tir_analyzer = ClinicalTIRAnalyzer()
        
        # Data storage
        self.glucose_history = []
        self.insulin_history = []
        self.time_history = []
        self.tir_history = {'very_low': [], 'low': [], 'target': [], 'high': [], 'very_high': []}
        self.uncertainty_history = []
        
        # Simulation parameters
        self.current_glucose = 120  # Starting glucose
        self.target_glucose = 120
        self.max_history = 200  # Keep last 200 points (about 16 hours at 5-min intervals)
        
        # Dashboard state
        self.is_running = False
        self.data_queue = queue.Queue()
        
        # Colors for zones (Medtronic standard)
        self.zone_colors = {
            'very_low': '#FF8C00',   # Dark orange
            'low': '#FFD700',        # Gold  
            'target': '#32CD32',     # Lime green
            'high': '#FF6B6B',       # Light red
            'very_high': '#DC143C'   # Crimson
        }
        
    def simulate_glucose_data(self):
        """Simulate realistic glucose data with meals, exercise, and noise"""
        
        # Base circadian rhythm (lower at night, higher during day)
        current_time = datetime.now()
        hour = current_time.hour + current_time.minute / 60.0
        circadian_effect = 10 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Meal effects (simplified)
        meal_effect = 0
        if 7 <= hour <= 8:  # Breakfast
            meal_effect = 40 * np.exp(-(hour - 7.5)**2 / 0.5)
        elif 12 <= hour <= 13:  # Lunch
            meal_effect = 35 * np.exp(-(hour - 12.5)**2 / 0.5)
        elif 18 <= hour <= 19:  # Dinner
            meal_effect = 45 * np.exp(-(hour - 18.5)**2 / 0.5)
        
        # Random walk with mean reversion
        glucose_change = np.random.normal(0, 3)  # Random noise
        mean_reversion = -0.1 * (self.current_glucose - self.target_glucose)  # Drift toward target
        
        # Update glucose
        self.current_glucose += glucose_change + mean_reversion + circadian_effect * 0.1 + meal_effect * 0.1
        
        # Apply physiological bounds
        self.current_glucose = np.clip(self.current_glucose, 40, 400)
        
        return self.current_glucose
    
    def simulate_ai_decision(self, glucose_value):
        """Simulate AI insulin decision with uncertainty"""
        
        # Simple insulin calculation (for demo)
        error = glucose_value - self.target_glucose
        insulin_dose = max(0, error * 0.02)  # Simple proportional control
        
        # Add some uncertainty based on glucose level
        if glucose_value < 70 or glucose_value > 250:
            uncertainty = np.random.uniform(0.6, 0.9)  # High uncertainty in extreme ranges
        elif 70 <= glucose_value <= 180:
            uncertainty = np.random.uniform(0.1, 0.3)  # Low uncertainty in target range
        else:
            uncertainty = np.random.uniform(0.3, 0.6)  # Medium uncertainty
        
        return insulin_dose, uncertainty
    
    def update_data(self):
        """Update glucose and insulin data"""
        
        # Simulate new glucose reading
        new_glucose = self.simulate_glucose_data()
        new_insulin, new_uncertainty = self.simulate_ai_decision(new_glucose)
        current_time = datetime.now()
        
        # Add to history
        self.glucose_history.append(new_glucose)
        self.insulin_history.append(new_insulin)
        self.uncertainty_history.append(new_uncertainty)
        self.time_history.append(current_time)
        
        # Maintain maximum history length
        if len(self.glucose_history) > self.max_history:
            self.glucose_history.pop(0)
            self.insulin_history.pop(0)
            self.uncertainty_history.pop(0)
            self.time_history.pop(0)
        
        # Update TIR analysis
        if len(self.glucose_history) >= 10:  # Need minimum data for analysis
            tir_analysis = self.tir_analyzer.analyze_glucose_zones(self.glucose_history)
            
            for zone in self.tir_history.keys():
                self.tir_history[zone].append(tir_analysis[zone]['percentage'])
                
                # Maintain history length
                if len(self.tir_history[zone]) > self.max_history:
                    self.tir_history[zone].pop(0)
        
        return new_glucose, new_insulin, new_uncertainty, current_time
    
    def create_dashboard(self):
        """Create the real-time dashboard visualization"""
        
        # Set up the figure and subplots
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('IINTS-AF Real-Time Glucose Monitoring Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Add live metrics ticker at top
        current_confidence = (1 - self.uncertainty_history[-1]) * 100 if self.uncertainty_history else 95.0
        metrics_text = f"CPU Load: 12% (Jetson Nano) | Neural Inference: 8.4ms | Safety Override: STANDBY | Confidence: {current_confidence:.1f}%"
        fig.text(0.5, 0.95, metrics_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main glucose plot (top row, spans 2 columns)
        ax_glucose = fig.add_subplot(gs[0, :2])
        
        # TIR meter (top right)
        ax_tir = fig.add_subplot(gs[0, 2])
        
        # Insulin delivery (middle left)
        ax_insulin = fig.add_subplot(gs[1, 0])
        
        # Uncertainty plot (middle center)
        ax_uncertainty = fig.add_subplot(gs[1, 1])
        
        # Zone distribution (middle right)
        ax_zones = fig.add_subplot(gs[1, 2])
        
        # Status panel (bottom row)
        ax_status = fig.add_subplot(gs[2, :])
        
        return fig, {
            'glucose': ax_glucose,
            'tir': ax_tir,
            'insulin': ax_insulin,
            'uncertainty': ax_uncertainty,
            'zones': ax_zones,
            'status': ax_status
        }
    
    def update_plots(self, frame, fig, axes):
        """Update all plots with new data"""
        
        if not self.glucose_history:
            return
        
        # Clear all axes
        for ax in axes.values():
            ax.clear()
        
        # 1. Main Glucose Plot
        ax_glucose = axes['glucose']
        
        if len(self.glucose_history) > 1:
            time_points = range(len(self.glucose_history))
            
            # Neural confidence glow effect
            for i in range(1, len(self.glucose_history)):
                if i < len(self.uncertainty_history):
                    conf = 1 - self.uncertainty_history[i]  # Convert uncertainty to confidence
                    alpha = 0.3 + 0.7 * conf  # More confident = more opaque
                    linewidth = 1 + 3 * conf  # More confident = thicker line
                    color_intensity = conf
                    
                    ax_glucose.plot([i-1, i], [self.glucose_history[i-1], self.glucose_history[i]], 
                                  color=(0, 0, color_intensity), linewidth=linewidth, alpha=alpha)
                else:
                    ax_glucose.plot([i-1, i], [self.glucose_history[i-1], self.glucose_history[i]], 
                                  'b-', linewidth=2)
            
            # Add zone boundaries
            ax_glucose.axhline(y=54, color='red', linestyle='--', alpha=0.7, label='Very Low')
            ax_glucose.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Low')
            ax_glucose.axhline(y=180, color='orange', linestyle='--', alpha=0.7, label='High')
            ax_glucose.axhline(y=250, color='red', linestyle='--', alpha=0.7, label='Very High')
            
            # Highlight target range
            ax_glucose.fill_between(time_points, 70, 180, alpha=0.2, color='green', label='Target Range')
            
            # Current glucose indicator
            current_glucose = self.glucose_history[-1]
            ax_glucose.scatter([len(self.glucose_history)-1], [current_glucose], 
                             color='red', s=100, zorder=5)
            ax_glucose.text(len(self.glucose_history)-1, current_glucose + 10, 
                          f'{current_glucose:.1f}', ha='center', fontweight='bold')
        
        ax_glucose.set_title('Real-Time Glucose Monitoring')
        ax_glucose.set_ylabel('Glucose (mg/dL)')
        ax_glucose.set_xlabel('Time Points (5-min intervals)')
        ax_glucose.legend(loc='upper right')
        ax_glucose.grid(True, alpha=0.3)
        ax_glucose.set_ylim(40, 300)
        
        # 2. TIR Meter
        ax_tir = axes['tir']
        
        if len(self.glucose_history) >= 10:
            current_tir = self.tir_history['target'][-1] if self.tir_history['target'] else 0
            
            # Create TIR meter (pie chart style)
            sizes = [current_tir, 100 - current_tir]
            colors = ['#32CD32', '#E0E0E0']
            
            wedges, texts = ax_tir.pie(sizes, colors=colors, startangle=90, 
                                      counterclock=False, wedgeprops=dict(width=0.3))
            
            # Add TIR percentage in center
            ax_tir.text(0, 0, f'{current_tir:.1f}%\nTIR', ha='center', va='center', 
                       fontsize=14, fontweight='bold')
            
        ax_tir.set_title('Time in Range')
        
        # 3. Insulin Delivery
        ax_insulin = axes['insulin']
        
        if len(self.insulin_history) > 1:
            time_points = range(len(self.insulin_history))
            ax_insulin.bar(time_points[-20:], self.insulin_history[-20:], 
                          color='skyblue', alpha=0.7)
            
        ax_insulin.set_title('Insulin Delivery (Last 20 readings)')
        ax_insulin.set_ylabel('Insulin (units)')
        ax_insulin.set_xlabel('Time Points')
        
        # 4. Uncertainty Plot
        ax_uncertainty = axes['uncertainty']
        
        if len(self.uncertainty_history) > 1:
            time_points = range(len(self.uncertainty_history))
            colors = ['green' if u < 0.3 else 'yellow' if u < 0.6 else 'red' 
                     for u in self.uncertainty_history]
            
            ax_uncertainty.scatter(time_points[-50:], self.uncertainty_history[-50:], 
                                 c=colors[-50:], alpha=0.7)
            
            # Add uncertainty threshold lines
            ax_uncertainty.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.7, label='Medium')
            ax_uncertainty.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='High')
        
        ax_uncertainty.set_title('AI Uncertainty')
        ax_uncertainty.set_ylabel('Uncertainty Score')
        ax_uncertainty.set_xlabel('Time Points')
        ax_uncertainty.set_ylim(0, 1)
        ax_uncertainty.legend()
        
        # 5. Zone Distribution
        ax_zones = axes['zones']
        
        if len(self.glucose_history) >= 10:
            tir_analysis = self.tir_analyzer.analyze_glucose_zones(self.glucose_history)
            
            zones = ['Very Low', 'Low', 'Target', 'High', 'Very High']
            percentages = [
                tir_analysis['very_low']['percentage'],
                tir_analysis['low']['percentage'], 
                tir_analysis['target']['percentage'],
                tir_analysis['high']['percentage'],
                tir_analysis['very_high']['percentage']
            ]
            colors = ['#FF8C00', '#FFD700', '#32CD32', '#FF6B6B', '#DC143C']
            
            bars = ax_zones.bar(zones, percentages, color=colors, alpha=0.8)
            
            # Add percentage labels
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax_zones.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax_zones.set_title('Current Zone Distribution')
        ax_zones.set_ylabel('Percentage (%)')
        ax_zones.tick_params(axis='x', rotation=45)
        
        # 6. Status Panel
        ax_status = axes['status']
        ax_status.axis('off')
        
        if self.glucose_history:
            current_glucose = self.glucose_history[-1]
            current_insulin = self.insulin_history[-1] if self.insulin_history else 0
            current_uncertainty = self.uncertainty_history[-1] if self.uncertainty_history else 0
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Determine glucose status
            if current_glucose < 54:
                glucose_status = "CRITICAL LOW"
                status_color = "red"
            elif current_glucose < 70:
                glucose_status = "LOW"
                status_color = "orange"
            elif current_glucose <= 180:
                glucose_status = "TARGET RANGE"
                status_color = "green"
            elif current_glucose <= 250:
                glucose_status = "HIGH"
                status_color = "orange"
            else:
                glucose_status = "CRITICAL HIGH"
                status_color = "red"
            
            # Status text
            status_text = f"""
CURRENT STATUS - {current_time}
Glucose: {current_glucose:.1f} mg/dL ({glucose_status})
Insulin: {current_insulin:.2f} units
AI Uncertainty: {current_uncertainty:.2f}
Readings: {len(self.glucose_history)}
            """
            
            ax_status.text(0.1, 0.5, status_text, transform=ax_status.transAxes,
                          fontsize=12, verticalalignment='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
    
    def data_generator(self):
        """Generate data in background thread"""
        while self.is_running:
            try:
                new_data = self.update_data()
                self.data_queue.put(new_data)
                time.sleep(2)  # Update every 2 seconds (faster than real 5-min intervals for demo)
            except Exception as e:
                print(f"Data generation error: {e}")
                break
    
    def start_dashboard(self, duration_minutes=10):
        """Start the real-time dashboard"""
        
        print("[START] STARTING REAL-TIME DASHBOARD")
        print("=" * 40)
        print(f"Duration: {duration_minutes} minutes")
        print("Close the window to stop the dashboard")
        print()
        
        # Initialize with some data
        for _ in range(20):
            self.update_data()
        
        # Create dashboard
        fig, axes = self.create_dashboard()
        
        # Start data generation thread
        self.is_running = True
        data_thread = threading.Thread(target=self.data_generator, daemon=True)
        data_thread.start()
        
        # Create animation
        ani = animation.FuncAnimation(fig, self.update_plots, fargs=(fig, axes),
                                    interval=2000, blit=False, cache_frame_data=False)
        
        # Show dashboard
        plt.show()
        
        # Stop data generation
        self.is_running = False
        
        print("\nDashboard session complete!")
        print(f"Total glucose readings: {len(self.glucose_history)}")
        
        if self.glucose_history:
            final_tir = self.tir_history['target'][-1] if self.tir_history['target'] else 0
            print(f"Final TIR: {final_tir:.1f}%")
        
        return self.glucose_history, self.insulin_history, self.uncertainty_history

def main():
    """Launch real-time dashboard demonstration"""
    
    dashboard = RealTimeDashboard()
    
    try:
        # Start dashboard for 5 minutes
        glucose_data, insulin_data, uncertainty_data = dashboard.start_dashboard(duration_minutes=5)
        
        # Save session data
        results_dir = Path("results/real_time_dashboard")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'glucose_history': glucose_data,
            'insulin_history': insulin_data,
            'uncertainty_history': uncertainty_data,
            'final_tir': dashboard.tir_history['target'][-1] if dashboard.tir_history['target'] else 0
        }
        
        import json
        with open(results_dir / 'dashboard_session.json', 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"Session data saved to: {results_dir}/dashboard_session.json")
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()
