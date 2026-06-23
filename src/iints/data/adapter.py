#!/usr/bin/env python3
"""
IINTS-AF Universal Data Adapter
Professional data import layer with schema validation
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from iints.core.safety.config import SENSOR_GLUCOSE_MAX_MGDL, SENSOR_GLUCOSE_MIN_MGDL

class DataAdapter:
    """Universal data adapter for IINTS-AF framework"""
    
    def __init__(self):
        self.data_packs_dir = Path(__file__).parent.parent.parent / "data_packs"
        
    def load_data_pack(self, pack_name: str) -> Dict:
        """Load and validate a data pack"""
        pack_dir = self.data_packs_dir / pack_name
        
        if not pack_dir.exists():
            raise FileNotFoundError(f"Data pack not found: {pack_name}")
        
        # Load data
        data_file = pack_dir / "data.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
        else:
            data_file = pack_dir / "data.json"
            if data_file.exists():
                df = pd.read_json(data_file)
            else:
                raise FileNotFoundError(f"No data file found in {pack_name}")
        
        # Basic validation
        self._validate_dataframe(df)
        
        return {
            "data": df,
            "pack_name": pack_name,
            "source_file": str(data_file)
        }
    
    def _validate_dataframe(self, df: pd.DataFrame):
        """Basic dataframe validation"""
        required_cols = ["timestamp", "glucose"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate glucose range
        if df['glucose'].min() < SENSOR_GLUCOSE_MIN_MGDL or df['glucose'].max() > SENSOR_GLUCOSE_MAX_MGDL:
            raise ValueError(
                "Glucose values outside broad CGM/sensor-valid range "
                f"({int(SENSOR_GLUCOSE_MIN_MGDL)}-{int(SENSOR_GLUCOSE_MAX_MGDL)} mg/dL)"
            )
    
    def load_ohio_dataset(self, patient_id: str) -> pd.DataFrame:
        """Load Ohio T1DM dataset with clinical benchmarks"""
        ohio_path = Path(f"data_packs/public/ohio_t1dm/patient_{patient_id}")
        
        if not ohio_path.exists():
            raise FileNotFoundError(
                f"Ohio patient {patient_id} not found. Run: python tools/data/import_ohio.py"
            )
        
        # Load timeseries data
        timeseries_file = ohio_path / "timeseries.csv"
        if not timeseries_file.exists():
            raise FileNotFoundError(f"No timeseries.csv found for patient {patient_id}")
        
        df = pd.read_csv(timeseries_file)
        
        # Calculate clinical benchmarks from original data
        glucose_values = df['glucose_mg_dl']
        benchmarks = {
            'original_tir': ((glucose_values >= 70) & (glucose_values <= 180)).mean() * 100,
            'original_gmi': (3.31 + 0.02392 * glucose_values.mean()),  # GMI formula
            'original_cv': glucose_values.std() / glucose_values.mean() * 100,
            'data_quality': len(df) / (8 * 7 * 24 * 12),  # Expected vs actual data points
            'patient_id': patient_id
        }
        
        # Add benchmarks as metadata
        df.attrs['clinical_benchmarks'] = benchmarks
        
        return df
    
    def get_available_ohio_patients(self) -> List[str]:
        """Get list of available Ohio T1DM patients"""
        ohio_dir = Path("data_packs/public/ohio_t1dm")
        if not ohio_dir.exists():
            return []
        
        patients = []
        for patient_dir in ohio_dir.glob("patient_*"):
            if (patient_dir / "timeseries.csv").exists():
                patient_id = patient_dir.name.replace("patient_", "")
                patients.append(patient_id)
        
        return sorted(patients)
    
    def clinical_benchmark_comparison(
        self,
        patient_id: str,
        algorithms: List[str],
        evaluated_outputs: Optional[Dict[str, Union[pd.DataFrame, str, Path]]] = None,
    ) -> Dict[str, Any]:
        """Compare measured algorithm traces against Ohio T1DM benchmarks.

        No synthetic improvement is generated. Algorithms without an evaluated
        output trace are returned as ``not_evaluated``.
        """
        df = self.load_ohio_dataset(patient_id)
        benchmarks = df.attrs.get('clinical_benchmarks', {})
        
        results: Dict[str, Any] = {
            'patient_id': patient_id,
            'original_performance': {
                'tir_70_180': benchmarks.get('original_tir', 0),
                'gmi': benchmarks.get('original_gmi', 0),
                'cv_percent': benchmarks.get('original_cv', 0)
            },
            'algorithm_results': {},
            'methodology': 'measured_trace_only',
            'synthetic_improvements_used': False,
        }

        baseline_tir = float(benchmarks.get('original_tir', 0.0))
        evaluated_outputs = evaluated_outputs or {}
        for algorithm in algorithms:
            source = evaluated_outputs.get(algorithm)
            if source is None:
                results['algorithm_results'][algorithm] = {
                    'status': 'not_evaluated',
                    'tir_70_180': None,
                    'improvement_percent': None,
                    'relative_improvement': None,
                }
                continue

            output_df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(Path(source))
            glucose_column = next(
                (
                    candidate
                    for candidate in ('glucose_actual_mgdl', 'glucose', 'glucose_mg_dl', 'cgm')
                    if candidate in output_df.columns
                ),
                None,
            )
            if glucose_column is None:
                raise ValueError(
                    f"Evaluated output for '{algorithm}' has no supported glucose column"
                )
            glucose = pd.to_numeric(output_df[glucose_column], errors='coerce').dropna()
            if glucose.empty:
                raise ValueError(f"Evaluated output for '{algorithm}' contains no valid glucose values")
            measured_tir = float(((glucose >= 70.0) & (glucose <= 180.0)).mean() * 100.0)
            improvement_points = measured_tir - baseline_tir
            relative_improvement = (
                improvement_points / baseline_tir * 100.0 if baseline_tir > 0.0 else None
            )
            results['algorithm_results'][algorithm] = {
                'status': 'measured',
                'tir_70_180': measured_tir,
                'improvement_percent': improvement_points,
                'relative_improvement': relative_improvement,
                'rows_evaluated': int(len(glucose)),
            }

        return results

def main():
    """Demo usage of DataAdapter"""
    adapter = DataAdapter()
    print("IINTS-AF Data Adapter initialized")

if __name__ == "__main__":
    main()
