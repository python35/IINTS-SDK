import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional
import yaml

class DataIngestor:
    """
    Standardized Data Bridge for ingesting various diabetes datasets into a
    universal IINTS-AF format.
    """
    UNIVERSAL_SCHEMA = {
        "timestamp": float,
        "glucose": float,
        "carbs": float, # Can be null, but pandas will infer float if mixed
        "insulin": float, # Can be null, but pandas will infer float if mixed
        "source": str,
    }

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or "./data"
        # Zorg dat de rest van je code zelf dit pad gebruikt om bestanden te vinden

    def _load_ohio_t1dm_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Loads and transforms Ohio T1DM dataset CSV into universal schema.
        Expected columns in Ohio T1DM: 'timestamp', 'glucose', 'carbs', 'insulin'
        """
        df = pd.read_csv(file_path)

        # Assuming 'timestamp' is already in minutes from start or can be converted
        # For simplicity, assuming it's already a float representing minutes
        # If it's a datetime, conversion would be needed:
        # df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60.0

        # Rename columns to match universal schema if necessary
        # Example: if original columns were different, map them here.
        # For Ohio T1DM, let's assume they are already lowercase 'glucose', 'carbs', 'insulin'
        # if 'BG' in df.columns: df = df.rename(columns={'BG': 'glucose'})
        # if 'Carbs' in df.columns: df = df.rename(columns={'Carbs': 'carbs'})
        # if 'Insulin' in df.columns: df = df.rename(columns={'Insulin': 'insulin'})

        # Add 'source' column
        df['source'] = 'public_ohio_t1dm'
        
        # Ensure only universal schema columns are present and in order
        required_cols = list(self.UNIVERSAL_SCHEMA.keys())
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA # Add missing columns as NA
        
        return df[required_cols]

    def _validate_schema(self, df: pd.DataFrame, schema: Dict[str, type]) -> None:
        """
        Validates DataFrame against the expected schema.
        Raises ValueError if validation fails.
        """
        for col, expected_type in schema.items():
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            # Basic type check (pandas dtypes are more complex, this is a simplified check)
            # if not pd.api.types.is_dtype_equal(df[col].dtype, pd.Series(dtype=expected_type).dtype):
            #     print(f"Warning: Column '{col}' type mismatch. Expected {expected_type}, got {df[col].dtype}")
        
        # Basic quality checks (from DATA_SCHEMA.md)
        if 'glucose' in df.columns:
            if not ((df['glucose'] >= 20) & (df['glucose'] <= 600)).all():
                raise ValueError("Glucose values outside acceptable range (20-600 mg/dL)")
        if 'insulin' in df.columns and not df['insulin'].isna().all():
            if not ((df['insulin'] >= 0) & (df['insulin'] <= 50)).all():
                raise ValueError("Insulin values outside acceptable range (0-50 units)")
        
        # Check for missing timestamps (assuming 'timestamp' exists)
        if df['timestamp'].isnull().any():
            raise ValueError("Missing values in 'timestamp' column.")

    def get_patient_model(self, file_path: Union[str, Path], data_type: str) -> pd.DataFrame:
        """
        Loads patient data from a file and returns it as a standardized DataFrame.

        Args:
            file_path (Union[str, Path]): Path to the data file. Can be extension-less for 'model' type.
            data_type (str): Type of the data source (e.g., 'ohio_t1dm', 'iints_standard_csv', 'model').

        Returns:
            pd.DataFrame: A DataFrame conforming to the universal IINTS-AF schema.
        
        Raises:
            ValueError: If the data_type is not supported or validation fails.
            FileNotFoundError: If the data file cannot be found.
        """
        file_path = Path(file_path)

        # If data type is 'model' (JSON), ensure the .json extension is present.
        if data_type == 'model' or data_type == 'iints_standard_json':
            if not file_path.suffix:
                file_path = file_path.with_suffix('.json')

        if not file_path.is_file():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.DataFrame()
        if data_type == 'ohio_t1dm':
            df = self._load_ohio_t1dm_csv(file_path)
        elif data_type == 'iints_standard_csv':
            # Assume this is already in the universal schema format
            df = pd.read_csv(file_path)
        elif data_type == 'model' or data_type == 'iints_standard_json':
            # Assumes a records-oriented JSON file.
            df = pd.read_json(file_path, orient='records')
            # Add source column if not present
            if 'source' not in df.columns:
                df['source'] = 'iints_standard_json'

            # Ensure only universal schema columns are present and in order
            required_cols = list(self.UNIVERSAL_SCHEMA.keys())
            for col in required_cols:
                if col not in df.columns:
                    df[col] = pd.NA # Add missing columns as NA
            df = df[required_cols]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Validate the loaded data against the universal schema
        self._validate_schema(df, self.UNIVERSAL_SCHEMA)

        return df

if __name__ == "__main__":
    # Example usage:
    # This assumes you have a timeseries.csv in data_packs/public/ohio_t1dm/patient_XXX/
    # For testing, we'll try to find one.
    
    ohio_data_path = Path("data_packs/public/ohio_t1dm")
    
    patient_dirs = [d for d in ohio_data_path.iterdir() if d.is_dir() and d.name.startswith("patient_")]
    
    if patient_dirs:
        sample_timeseries_file = None
        for patient_dir in patient_dirs:
            if (patient_dir / "timeseries.csv").is_file():
                sample_timeseries_file = patient_dir / "timeseries.csv"
                break
        
        if sample_timeseries_file:
            print(f"Loading sample Ohio T1DM data from: {sample_timeseries_file}")
            ingestor = DataIngestor()
            try:
                df = ingestor.get_patient_model(sample_timeseries_file, 'ohio_t1dm')
                print("Data loaded successfully and validated:")
                print(df.head())
                df.info()
            except FileNotFoundError as e:
                print(f"Error loading data: {e}")
            except ValueError as e:
                print(f"Data quality issue detected during validation for {sample_timeseries_file.name}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        else:
            print(f"No 'timeseries.csv' found in any patient directory within {ohio_data_path}. Cannot run example.")
    else:
        print(f"No patient directories found in {ohio_data_path}. Cannot run example.")
