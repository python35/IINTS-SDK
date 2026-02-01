import pandas as pd
from typing import Dict, Any, Tuple

def calculate_tir(df: pd.DataFrame, lower_bound: float = 70.0, upper_bound: float = 180.0) -> float:
    """
    Calculates Time In Range (TIR) for glucose values.

    Args:
        df (pd.DataFrame): DataFrame containing simulation results, must have a 'glucose_actual_mgdl' column.
        lower_bound (float): Lower bound for Time In Range (mg/dL).
        upper_bound (float): Upper bound for Time In Range (mg/dL).

    Returns:
        float: Percentage of time in range (0-100).
    """
    if 'glucose_actual_mgdl' not in df.columns:
        return float('nan') # Or raise error, returning nan for robustness in benchmark
        # raise ValueError("DataFrame must contain 'glucose_actual_mgdl' column for TIR calculation.")

    in_range = (df['glucose_actual_mgdl'] >= lower_bound) & (df['glucose_actual_mgdl'] <= upper_bound)
    tir_percentage = in_range.mean() * 100
    return tir_percentage

def calculate_hypoglycemia(df: pd.DataFrame, threshold: float = 70.0) -> float:
    """
    Calculates percentage of time in hypoglycemia.

    Args:
        df (pd.DataFrame): DataFrame containing simulation results, must have a 'glucose_actual_mgdl' column.
        threshold (float): Glucose threshold for hypoglycemia (mg/dL).

    Returns:
        float: Percentage of time in hypoglycemia (0-100).
    """
    if 'glucose_actual_mgdl' not in df.columns:
        return float('nan')
        # raise ValueError("DataFrame must contain 'glucose_actual_mgdl' column for hypoglycemia calculation.")
    
    hypo = (df['glucose_actual_mgdl'] < threshold)
    hypo_percentage = hypo.mean() * 100
    return hypo_percentage

def calculate_hyperglycemia(df: pd.DataFrame, threshold: float = 180.0) -> float:
    """
    Calculates percentage of time in hyperglycemia.

    Args:
        df (pd.DataFrame): DataFrame containing simulation results, must have a 'glucose_actual_mgdl' column.
        threshold (float): Glucose threshold for hyperglycemia (mg/dL).

    Returns:
        float: Percentage of time in hyperglycemia (0-100).
    """
    if 'glucose_actual_mgdl' not in df.columns:
        return float('nan')
        # raise ValueError("DataFrame must contain 'glucose_actual_mgdl' column for hyperglycemia calculation.")
    
    hyper = (df['glucose_actual_mgdl'] > threshold)
    hyper_percentage = hyper.mean() * 100
    return hyper_percentage

def calculate_average_glucose(df: pd.DataFrame) -> float:
    """
    Calculates the average glucose value.

    Args:
        df (pd.DataFrame): DataFrame containing simulation results, must have a 'glucose_actual_mgdl' column.

    Returns:
        float: Average glucose value (mg/dL).
    """
    if 'glucose_actual_mgdl' not in df.columns:
        return float('nan')
        # raise ValueError("DataFrame must contain 'glucose_actual_mgdl' column for average glucose calculation.")
    
    return df['glucose_actual_mgdl'].mean()

def generate_benchmark_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Generates a dictionary of key benchmark metrics from simulation results.

    Args:
        df (pd.DataFrame): DataFrame containing simulation results.

    Returns:
        Dict[str, float]: A dictionary of calculated metrics.
    """
    metrics = {
        "TIR (%)": calculate_tir(df),
        "Hypoglycemia (<70 mg/dL) (%)": calculate_hypoglycemia(df),
        "Hyperglycemia (>180 mg/dL) (%)": calculate_hyperglycemia(df),
        "Avg Glucose (mg/dL)": calculate_average_glucose(df),
    }
    return metrics

if __name__ == "__main__":
    # Example usage:
    print("Running example for metrics.py")
    # Create a dummy DataFrame for testing
    data = {
        'time_minutes': range(0, 60, 5),
        'glucose_actual_mgdl': [100, 110, 150, 200, 170, 80, 60, 50, 90, 120, 140, 160]
    }
    dummy_df = pd.DataFrame(data)

    print("\nDummy DataFrame:")
    print(dummy_df)

    metrics_results = generate_benchmark_metrics(dummy_df)
    print("\nCalculated Metrics:")
    for key, value in metrics_results.items():
        print(f"- {key}: {value:.2f}")

    # Test with custom ranges
    print("\nCalculated Metrics (Custom TIR 80-140):")
    custom_tir = calculate_tir(dummy_df, lower_bound=80, upper_bound=140)
    print(f"- TIR (80-140 mg/dL): {custom_tir:.2f}%")
