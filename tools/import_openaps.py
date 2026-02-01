#!/usr/bin/env python3
"""
OpenAPS Data Commons Importer
Converts OpenHumans/OpenAPS data to IINTS-AF format
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime

class OpenAPSImporter:
    def __init__(self, openaps_path, output_path):
        self.openaps_path = Path(openaps_path)
        self.output_path = Path(output_path)
        
    def sanitize_check(self, data):
        """Enhanced PII detection for OpenAPS data"""
        if isinstance(data, dict):
            pii_keys = ['name', 'email', 'phone', 'address', 'device_id', 'serial']
            found_pii = [key for key in pii_keys if key in str(data).lower()]
        else:
            pii_columns = ['name', 'email', 'phone', 'address', 'device_id']
            found_pii = [col for col in pii_columns if col in data.columns.str.lower()]
        
        if found_pii:
            print(f"[WARNING] Potential PII detected: {found_pii}")
            return False
            
        print("[INFO] Dataset sanitized: No PII detected")
        return True
    
    def convert_nightscout_export(self, export_file, patient_alias):
        """Convert Nightscout JSON export to IINTS format"""
        
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        if not self.sanitize_check(data):
            return False
            
        output_data = []
        
        # Process entries (CGM data)
        entries = data.get('entries', [])
        treatments = data.get('treatments', [])
        
        # Create timeline from entries
        for entry in entries:
            timestamp = datetime.fromtimestamp(entry['date'] / 1000)
            glucose = entry.get('sgv', None)
            
            if glucose and glucose > 0:
                # Find treatments around this time (±15 minutes)
                entry_time = entry['date']
                relevant_treatments = [
                    t for t in treatments 
                    if abs(t.get('created_at', entry_time) - entry_time) < 15 * 60 * 1000
                ]
                
                # Sum insulin and carbs
                insulin = sum(float(t.get('insulin', 0)) for t in relevant_treatments)
                carbs = sum(float(t.get('carbs', 0)) for t in relevant_treatments)
                
                output_data.append({
                    'timestamp': timestamp.isoformat(),
                    'glucose_mg_dl': float(glucose),
                    'insulin_units': float(insulin),
                    'carbs_grams': float(carbs),
                    'patient_id': f"openaps_{patient_alias}"
                })
        
        # Sort by timestamp
        output_data.sort(key=lambda x: x['timestamp'])
        
        # Save to IINTS format
        output_dir = self.output_path / "public" / "openaps" / f"patient_{patient_alias}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_output = pd.DataFrame(output_data)
        df_output.to_csv(output_dir / "timeseries.csv", index=False)
        
        # Create metadata
        metadata = {
            "name": f"OpenAPS Patient {patient_alias}",
            "source": "OpenAPS Data Commons / Nightscout",
            "patient_profile": {
                "type": "real_world_diy",
                "duration_days": len(set(d['timestamp'][:10] for d in output_data)),
                "data_points": len(output_data),
                "algorithm_used": "OpenAPS/AndroidAPS"
            },
            "variables": ["glucose_mg_dl", "insulin_units", "carbs_grams"],
            "sampling_rate": "variable",
            "privacy": "anonymized_community",
            "citation": "OpenAPS Data Commons, https://openhumans.org/"
        }
        
        import yaml
        with open(output_dir / "metadata.yaml", 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"[SUCCESS] Converted OpenAPS patient {patient_alias}: {len(output_data)} data points")
        return True

def main():
    parser = argparse.ArgumentParser(description='Import OpenAPS/Nightscout Data')
    parser.add_argument('data_path', help='Path to OpenAPS data directory or Nightscout export')
    parser.add_argument('--output', default='data_packs', help='Output directory')
    parser.add_argument('--alias', default='anonymous', help='Patient alias for anonymization')
    
    args = parser.parse_args()
    
    importer = OpenAPSImporter(args.data_path, args.output)
    
    data_path = Path(args.data_path)
    
    if data_path.is_file() and data_path.suffix == '.json':
        # Single Nightscout export
        success = importer.convert_nightscout_export(data_path, args.alias)
        if success:
            print(f"\n[SUCCESS] Real-world DIY data imported to: {args.output}/public/openaps/")
        else:
            print("[ERROR] Import failed")
    else:
        print("[ERROR] Please provide a Nightscout JSON export file")

if __name__ == "__main__":
    main()