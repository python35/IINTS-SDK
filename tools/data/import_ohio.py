#!/usr/bin/env python3
"""
Ohio T1DM Dataset Importer
Converts Ohio University T1DM dataset to IINTS-AF format
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime

class OhioDataImporter:
    def __init__(self, ohio_path, output_path):
        self.ohio_path = Path(ohio_path)
        self.output_path = Path(output_path)
        
    def sanitize_check(self, df):
        """Check for PII and sanitize"""
        pii_columns = ['name', 'email', 'phone', 'address', 'ssn']
        found_pii = [col for col in pii_columns if col in df.columns.str.lower()]
        
        if found_pii:
            print(f"[WARNING] PII detected: {found_pii}")
            return False
        
        print("[INFO] Dataset sanitized: No PII detected")
        return True
    
    def convert_patient(self, patient_id):
        """Convert single Ohio patient to IINTS format"""
        patient_dir = self.ohio_path / f"{patient_id}-ws-training"
        
        if not patient_dir.exists():
            print(f"[ERROR] Patient {patient_id} not found")
            return False
            
        # Load Ohio data files
        cgm_file = patient_dir / f"{patient_id}-ws-training_cgm.csv"
        insulin_file = patient_dir / f"{patient_id}-ws-training_insulin.csv"
        meal_file = patient_dir / f"{patient_id}-ws-training_meal.csv"
        
        if not all(f.exists() for f in [cgm_file, insulin_file, meal_file]):
            print(f"[ERROR] Missing data files for patient {patient_id}")
            return False
            
        # Read and sanitize
        cgm_df = pd.read_csv(cgm_file)
        insulin_df = pd.read_csv(insulin_file)
        meal_df = pd.read_csv(meal_file)
        
        for df in [cgm_df, insulin_df, meal_df]:
            if not self.sanitize_check(df):
                return False
        
        # Convert to IINTS format
        output_data = []
        
        # Merge on timestamp
        cgm_df['ts'] = pd.to_datetime(cgm_df['ts'])
        insulin_df['ts'] = pd.to_datetime(insulin_df['ts'])
        meal_df['ts'] = pd.to_datetime(meal_df['ts'])
        
        # Create unified timeline (5-minute intervals)
        start_time = min(cgm_df['ts'].min(), insulin_df['ts'].min(), meal_df['ts'].min())
        end_time = max(cgm_df['ts'].max(), insulin_df['ts'].max(), meal_df['ts'].max())
        
        timeline = pd.date_range(start_time, end_time, freq='5min')
        
        for t in timeline:
            # Get closest CGM reading
            cgm_closest = cgm_df.iloc[(cgm_df['ts'] - t).abs().argsort()[:1]]
            glucose = cgm_closest['glucose'].iloc[0] if not cgm_closest.empty else None
            
            # Get insulin in last 5 minutes
            insulin_recent = insulin_df[(insulin_df['ts'] >= t - pd.Timedelta(minutes=5)) & 
                                     (insulin_df['ts'] <= t)]
            insulin = insulin_recent['dose'].sum() if not insulin_recent.empty else 0.0
            
            # Get meals in last 5 minutes
            meal_recent = meal_df[(meal_df['ts'] >= t - pd.Timedelta(minutes=5)) & 
                                (meal_df['ts'] <= t)]
            carbs = meal_recent['carbs'].sum() if not meal_recent.empty else 0.0
            
            if glucose is not None:
                output_data.append({
                    'timestamp': t.isoformat(),
                    'glucose_mg_dl': float(glucose),
                    'insulin_units': float(insulin),
                    'carbs_grams': float(carbs),
                    'patient_id': f"ohio_{patient_id}"
                })
        
        # Save to IINTS format
        output_dir = self.output_path / "public" / "ohio_t1dm" / f"patient_{patient_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        df_output = pd.DataFrame(output_data)
        df_output.to_csv(output_dir / "timeseries.csv", index=False)
        
        # Create metadata
        metadata = {
            "name": f"Ohio T1DM Patient {patient_id}",
            "source": "Ohio University T1DM Dataset",
            "patient_profile": {
                "type": "real_world",
                "duration_days": len(timeline) / (24 * 12),  # 5-min intervals
                "data_points": len(output_data)
            },
            "variables": ["glucose_mg_dl", "insulin_units", "carbs_grams"],
            "sampling_rate": "5_minutes",
            "privacy": "anonymized",
            "citation": "Marling, C., & Bunescu, R. (2018). The OhioT1DM Dataset for Blood Glucose Level Prediction"
        }
        
        with open(output_dir / "metadata.yaml", 'w') as f:
            import yaml
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"[SUCCESS] Converted patient {patient_id}: {len(output_data)} data points")
        return True

def main():
    parser = argparse.ArgumentParser(description='Import Ohio T1DM Dataset')
    parser.add_argument('ohio_path', help='Path to Ohio dataset directory')
    parser.add_argument('--output', default='data_packs', help='Output directory')
    parser.add_argument('--patients', nargs='+', default=['559', '563', '570', '575', '588', '591'], 
                       help='Patient IDs to convert')
    
    args = parser.parse_args()
    
    importer = OhioDataImporter(args.ohio_path, args.output)
    
    success_count = 0
    for patient_id in args.patients:
        if importer.convert_patient(patient_id):
            success_count += 1
    
    print(f"\n[SUMMARY] Successfully converted {success_count}/{len(args.patients)} patients")
    print(f"Real-world data now available in: {args.output}/public/ohio_t1dm/")

if __name__ == "__main__":
    main()