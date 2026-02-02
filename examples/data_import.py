#!/usr/bin/env python3
"""
IINTS-AF Data Import Tool
Professional CLI for importing custom datasets
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iints.data.adapter import DataAdapter

def import_dataset(file_path: str, pack_name: str, sanitize: bool = True):
    """Import custom dataset with validation"""
    
    print(f"Importing dataset: {file_path}")
    print(f"Pack name: {pack_name}")
    print(f"Sanitize: {sanitize}")
    
    try:
        # Initialize adapter
        adapter = DataAdapter()
        
        # Create metadata
        metadata = {
            "name": pack_name,
            "version": "1.0",
            "description": f"Custom imported dataset from {Path(file_path).name}",
            "source": "custom",
            "created": "2024-01-08"
        }
        
        # Import dataset
        data_pack = adapter.import_custom_dataset(file_path, metadata, sanitize)
        
        # Normalize to standard format
        normalized_df = adapter.normalize_to_timeseries(data_pack)
        
        # Create output directory
        output_dir = project_root / "data_packs" / "custom" / pack_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save normalized data
        normalized_df.to_csv(output_dir / "data.csv", index=False)
        
        # Save metadata
        with open(output_dir / "metadata.yaml", "w") as f:
            import yaml
            yaml.dump(metadata, f)
        
        # Create README
        readme_content = f"""# {pack_name}

## Dataset Information
- **Source**: Custom import from {Path(file_path).name}
- **Records**: {len(normalized_df)}
- **Duration**: {normalized_df['timestamp'].max():.1f} minutes
- **Glucose Range**: {normalized_df['glucose'].min():.1f} - {normalized_df['glucose'].max():.1f} mg/dL

## Usage
```python
from iints.data.adapter import DataAdapter

adapter = DataAdapter()
data_pack = adapter.load_data_pack("custom/{pack_name}")
df = adapter.normalize_to_timeseries(data_pack)
```

## Validation
- [OK] Schema compliance checked
- [OK] Physiological ranges validated
- [OK] Temporal consistency verified
{"- [OK] Privacy sanitization applied" if sanitize else "- [WARN] No sanitization applied"}
"""
        
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print(f"[OK] Dataset imported successfully!")
        print(f"Location: {output_dir}")
        print(f"Records: {len(normalized_df)}")
        print(f"Duration: {normalized_df['timestamp'].max():.1f} minutes")
        print(f"Glucose range: {normalized_df['glucose'].min():.1f} - {normalized_df['glucose'].max():.1f} mg/dL")
        
        return str(output_dir)
        
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return None

def list_data_packs():
    """List all available data packs"""
    
    adapter = DataAdapter()
    packs = adapter.list_available_packs()
    
    print("Available Data Packs:")
    print("=" * 50)
    
    if not packs:
        print("No data packs found.")
        return
    
    for pack in packs:
        print(f"{pack['name']}")
        print(f"   Description: {pack['description']}")
        print(f"   Patients: {pack['patient_count']}")
        print(f"   Scenarios: {', '.join(pack['scenarios'])}")
        print()

def validate_dataset(file_path: str):
    """Validate dataset without importing"""
    
    print(f"Validating dataset: {file_path}")
    
    try:
        # Load file
        if Path(file_path).suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif Path(file_path).suffix.lower() == '.json':
            df = pd.read_json(file_path)
        else:
            print(f"[FAIL] Unsupported file format: {Path(file_path).suffix}")
            return False
        
        # Basic validation
        adapter = DataAdapter()
        adapter._validate_dataframe(df)
        
        print("[OK] Dataset validation passed!")
        print(f"Records: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        
        if 'timestamp' in df.columns:
            print(f"Duration: {df['timestamp'].max() - df['timestamp'].min():.1f} minutes")
        
        if 'glucose' in df.columns:
            print(f"Glucose range: {df['glucose'].min():.1f} - {df['glucose'].max():.1f} mg/dL")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="IINTS-AF Data Import Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import custom dataset
  python data_import.py --import mydata.csv --name my_study --sanitize
  
  # Validate dataset
  python data_import.py --validate mydata.csv
  
  # List available data packs
  python data_import.py --list
        """
    )
    
    parser.add_argument("--import", dest="import_file", help="Import dataset file")
    parser.add_argument("--name", help="Data pack name for imported dataset")
    parser.add_argument("--sanitize", action="store_true", help="Apply privacy sanitization")
    parser.add_argument("--validate", help="Validate dataset file without importing")
    parser.add_argument("--list", action="store_true", help="List available data packs")
    
    args = parser.parse_args()
    
    if args.list:
        list_data_packs()
    elif args.validate:
        validate_dataset(args.validate)
    elif args.import_file:
        if not args.name:
            print("[FAIL] Error: --name required when importing dataset")
            sys.exit(1)
        import_dataset(args.import_file, args.name, args.sanitize)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
