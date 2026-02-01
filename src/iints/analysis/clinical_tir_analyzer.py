#!/usr/bin/env python3
"""
Professional 5-Zone TIR Analysis - IINTS-AF
Implements Medtronic clinical standard for glucose zone classification
"""

import numpy as np
import pandas as pd
from pathlib import Path

class ClinicalTIRAnalyzer:
    """Professional 5-zone Time in Range analysis following clinical standards"""
    
    def __init__(self):
        # Medtronic 5-Zone Clinical Standard
        self.zones = {
            'very_low': {'range': (0, 54), 'color': '#FF8C00', 'name': 'Very Low', 'clinical': 'Severe Hypoglycemia'},
            'low': {'range': (54, 70), 'color': '#FFD700', 'name': 'Low', 'clinical': 'Hypoglycemia'},
            'target': {'range': (70, 180), 'color': '#32CD32', 'name': 'Target', 'clinical': 'Time in Range'},
            'high': {'range': (180, 250), 'color': '#FF6B6B', 'name': 'High', 'clinical': 'Hyperglycemia'},
            'very_high': {'range': (250, 400), 'color': '#DC143C', 'name': 'Very High', 'clinical': 'Severe Hyperglycemia'}
        }
    
    def analyze_glucose_zones(self, glucose_values):
        """Analyze glucose data using professional 5-zone classification"""
        if glucose_values is None or len(glucose_values) == 0:
            return self._empty_analysis()
        
        glucose_array = np.array(glucose_values)
        total_readings = len(glucose_array)
        
        analysis = {}
        for zone_name, zone_info in self.zones.items():
            min_val, max_val = zone_info['range']
            
            if zone_name == 'very_high':
                in_zone = np.sum(glucose_array >= min_val)
            else:
                in_zone = np.sum((glucose_array >= min_val) & (glucose_array < max_val))
            
            percentage = (in_zone / total_readings) * 100
            
            analysis[zone_name] = {
                'count': int(in_zone),
                'percentage': round(percentage, 1),
                'color': zone_info['color'],
                'clinical_name': zone_info['clinical'],
                'range_mg_dL': f"{min_val}-{max_val if zone_name != 'very_high' else '400+'}"
            }
        
        # Clinical risk assessment
        analysis['clinical_assessment'] = self._assess_clinical_risk(analysis)
        analysis['total_readings'] = total_readings
        
        return analysis
    
    def _assess_clinical_risk(self, analysis):
        """Assess clinical risk based on zone percentages"""
        very_low_pct = analysis['very_low']['percentage']
        low_pct = analysis['low']['percentage']
        target_pct = analysis['target']['percentage']
        high_pct = analysis['high']['percentage']
        very_high_pct = analysis['very_high']['percentage']
        
        # Clinical risk criteria
        if very_low_pct > 1.0:
            risk_level = "HIGH RISK"
            primary_concern = "Severe hypoglycemia events exceed 1% threshold"
        elif low_pct > 4.0:
            risk_level = "MODERATE RISK"
            primary_concern = "Hypoglycemia events exceed 4% threshold"
        elif target_pct < 70.0:
            risk_level = "SUBOPTIMAL"
            primary_concern = f"Time in Range {target_pct:.1f}% below 70% target"
        elif very_high_pct > 5.0:
            risk_level = "MODERATE RISK"
            primary_concern = "Severe hyperglycemia events exceed 5% threshold"
        else:
            risk_level = "OPTIMAL"
            primary_concern = "All glucose zones within clinical targets"
        
        return {
            'risk_level': risk_level,
            'primary_concern': primary_concern,
            'tir_quality': 'Excellent' if target_pct >= 80 else 'Good' if target_pct >= 70 else 'Needs Improvement'
        }
    
    def _empty_analysis(self):
        """Return empty analysis structure"""
        analysis = {}
        for zone_name, zone_info in self.zones.items():
            analysis[zone_name] = {
                'count': 0,
                'percentage': 0.0,
                'color': zone_info['color'],
                'clinical_name': zone_info['clinical'],
                'range_mg_dL': f"{zone_info['range'][0]}-{zone_info['range'][1] if zone_name != 'very_high' else '400+'}"
            }
        
        analysis['clinical_assessment'] = {
            'risk_level': 'NO DATA',
            'primary_concern': 'Insufficient glucose readings for analysis',
            'tir_quality': 'Cannot assess'
        }
        analysis['total_readings'] = 0
        
        return analysis

def main():
    """Test professional TIR analysis"""
    analyzer = ClinicalTIRAnalyzer()
    
    # Test with sample glucose data
    sample_glucose = [
        45, 65, 85, 120, 140, 165, 180, 195, 220, 260,  # Various zones
        110, 125, 135, 150, 160, 170, 145, 130, 115, 105  # Mostly target
    ]
    
    analysis = analyzer.analyze_glucose_zones(sample_glucose)
    
    print("Professional 5-Zone TIR Analysis")
    print("=" * 40)
    
    for zone_name, data in analysis.items():
        if zone_name in ['clinical_assessment', 'total_readings']:
            continue
            
        print(f"{data['clinical_name']:20} ({data['range_mg_dL']:>8}): {data['percentage']:>5.1f}% ({data['count']:>2} readings)")
    
    print(f"\nTotal Readings: {analysis['total_readings']}")
    print(f"Clinical Assessment: {analysis['clinical_assessment']['risk_level']}")
    print(f"Primary Concern: {analysis['clinical_assessment']['primary_concern']}")
    print(f"TIR Quality: {analysis['clinical_assessment']['tir_quality']}")

if __name__ == "__main__":
    main()