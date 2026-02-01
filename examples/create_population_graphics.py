#!/usr/bin/env python3
"""
Population Study Visualizer - IINTS-AF
Converts Excel summary data into professional Science Expo graphics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob

class PopulationStudyVisualizer:
    def __init__(self):
        self.results_dir = Path("results")
        self.plots_dir = Path("results/population_plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Professional medical colors
        self.colors = {
            'lstm_learned': '#2E8B57',    # Sea Green (AI success)
            'hybrid': '#4682B4',          # Steel Blue (hybrid approach)  
            'rule_based': '#CD853F',      # Peru (traditional)
            'improvement': '#228B22',     # Forest Green (positive)
            'decline': '#DC143C'          # Crimson (negative)
        }
    
    def load_latest_excel(self):
        """Load the most recent Excel summary"""
        excel_files = list(self.results_dir.glob("IINTS_Summary_*.xlsx"))
        if not excel_files:
            raise FileNotFoundError("No Excel summary found. Generate one first.")
        
        latest_excel = max(excel_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading: {latest_excel.name}")
        
        # Load all sheets
        data = {}
        data['population'] = pd.read_excel(latest_excel, sheet_name='Population_Study')
        data['statistics'] = pd.read_excel(latest_excel, sheet_name='Statistical_Analysis')
        data['comparison'] = pd.read_excel(latest_excel, sheet_name='Algorithm_Comparison')
        
        return data
    
    def create_population_overview(self, data):
        """Create main population study overview chart"""
        pop_df = data['population']
        
        # Calculate algorithm performance
        alg_summary = pop_df.groupby('Algorithm').agg({
            'Improvement_%': ['mean', 'std', 'count'],
            'TIR_AI_%': 'mean',
            'Safety_Overrides': 'mean'
        }).round(2)
        
        # Flatten column names
        alg_summary.columns = ['Mean_Improvement', 'Std_Improvement', 'Patient_Count', 'Mean_TIR', 'Mean_Overrides']
        alg_summary = alg_summary.reset_index()
        
        # Create professional figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('IINTS-AF Population Study Results\nOhio T1DM Dataset (N=7 patients, 21 scenarios)', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # 1. TIR Improvement by Algorithm
        bars1 = ax1.bar(alg_summary['Algorithm'], alg_summary['Mean_Improvement'], 
                       color=[self.colors.get(alg, '#666666') for alg in alg_summary['Algorithm']],
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add error bars
        ax1.errorbar(alg_summary['Algorithm'], alg_summary['Mean_Improvement'], 
                    yerr=alg_summary['Std_Improvement'], fmt='none', color='black', capsize=5)
        
        ax1.set_title('Mean TIR Improvement by Algorithm', fontweight='bold')
        ax1.set_ylabel('TIR Improvement (%)')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, alg_summary['Mean_Improvement']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Success Rate (% of positive improvements)
        success_rates = []
        for alg in alg_summary['Algorithm']:
            alg_data = pop_df[pop_df['Algorithm'] == alg]
            success_rate = (alg_data['Improvement_%'] > 0).mean() * 100
            success_rates.append(success_rate)
        
        bars2 = ax2.bar(alg_summary['Algorithm'], success_rates,
                       color=[self.colors.get(alg, '#666666') for alg in alg_summary['Algorithm']],
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_title('Success Rate (% Positive Outcomes)', fontweight='bold')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. TIR Distribution by Algorithm
        for alg in pop_df['Algorithm'].unique():
            alg_data = pop_df[pop_df['Algorithm'] == alg]['TIR_AI_%']
            ax3.hist(alg_data, alpha=0.6, label=alg, bins=8, 
                    color=self.colors.get(alg, '#666666'))
        
        ax3.set_title('TIR Distribution by Algorithm', fontweight='bold')
        ax3.set_xlabel('Time in Range (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Safety vs Performance Scatter
        for alg in pop_df['Algorithm'].unique():
            alg_data = pop_df[pop_df['Algorithm'] == alg]
            ax4.scatter(alg_data['Safety_Overrides'], alg_data['TIR_AI_%'], 
                       label=alg, alpha=0.7, s=60,
                       color=self.colors.get(alg, '#666666'))
        
        ax4.set_title('Safety vs Performance Trade-off', fontweight='bold')
        ax4.set_xlabel('Safety Overrides (count)')
        ax4.set_ylabel('TIR Achievement (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / "population_study_overview.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Population overview saved: {plot_file}")
        
        return plot_file
    
    def create_scenario_heatmap(self, data):
        """Create scenario performance heatmap"""
        pop_df = data['population']
        
        # Create pivot table for heatmap
        heatmap_data = pop_df.pivot_table(
            values='Improvement_%', 
            index='Algorithm', 
            columns='Scenario', 
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'TIR Improvement (%)'})
        
        plt.title('Algorithm Performance by Clinical Scenario\n(Mean TIR Improvement %)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Clinical Scenario', fontweight='bold')
        plt.ylabel('Algorithm Type', fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_file = self.plots_dir / "scenario_performance_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Scenario heatmap saved: {heatmap_file}")
        
        return heatmap_file
    
    def create_statistical_summary(self, data):
        """Create statistical significance summary"""
        stats_df = data['statistics']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. Mean improvement with confidence intervals
        algorithms = stats_df['Algorithm']
        means = stats_df['Mean_TIR_Improvement_%']
        stds = stats_df['Std_TIR_Improvement']
        
        bars = ax1.bar(algorithms, means, 
                      color=[self.colors.get(alg, '#666666') for alg in algorithms],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax1.errorbar(algorithms, means, yerr=stds, fmt='none', color='black', capsize=5)
        ax1.set_title('Mean TIR Improvement ± SD', fontweight='bold')
        ax1.set_ylabel('TIR Improvement (%)')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, (bar, mean, p_val) in enumerate(zip(bars, means, stats_df['P_Value_Estimate'])):
            significance = '***' if p_val == '< 0.05' else 'ns'
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds.iloc[i] + 0.5, 
                    significance, ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # 2. Success rate comparison
        success_rates = stats_df['Success_Rate_%']
        bars2 = ax2.bar(algorithms, success_rates,
                       color=[self.colors.get(alg, '#666666') for alg in algorithms],
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_title('Clinical Success Rate', fontweight='bold')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save statistical summary
        stats_file = self.plots_dir / "statistical_summary.png"
        plt.savefig(stats_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Statistical summary saved: {stats_file}")
        
        return stats_file

def main():
    visualizer = PopulationStudyVisualizer()
    
    try:
        # Load Excel data
        data = visualizer.load_latest_excel()
        
        print("Creating population study visualizations...")
        
        # Generate all plots
        overview_plot = visualizer.create_population_overview(data)
        heatmap_plot = visualizer.create_scenario_heatmap(data)
        stats_plot = visualizer.create_statistical_summary(data)
        
        print(f"\nScience Expo graphics generated!")
        print(f"Location: results/population_plots/")
        print(f"\nFiles created:")
        print(f"- population_study_overview.png (Main poster graphic)")
        print(f"- scenario_performance_heatmap.png (Clinical scenarios)")
        print(f"- statistical_summary.png (P-values & significance)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()