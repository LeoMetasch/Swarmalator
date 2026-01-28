from swarm import Swarm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_transient_times(
    csv_path: str, 
    window_size: int = 50, 
    threshold: float = 0.01,
    require_all: bool = True
) -> dict:
    """
    Calculate transient time (convergence time) for order parameters.
    
    Args:
        csv_path: Path to CSV with logged order parameters
        window_size: Size of rolling window for std calculation
        threshold: Std threshold below which we consider convergence
        require_all: If True, convergence only when ALL parameters converge
        
    Returns:
        dict with:
            - 'transient_time': Single convergence step (when all converge)
            - 'individual_times': Dict of convergence time per parameter
            - 'converged': Boolean if system converged
    """
    df = pd.read_csv(csv_path)
    params = ['S', 'V', 'omega', 'R']
    
    individual_times = {}
    
    for param in params:
        rolling_std = df[param].rolling(window=window_size).std()
        
        converged_idx = None
        for i in range(window_size, len(rolling_std)):
            if rolling_std.iloc[i] < threshold:
                check_range = min(i + 10, len(rolling_std))
                if all(rolling_std.iloc[i:check_range] < threshold):
                    converged_idx = i
                    break
        
        individual_times[param] = df.loc[converged_idx, 'step'] if converged_idx is not None else None
    
    # Determine overall transient time
    if require_all:
        # Convergence = when the LAST parameter converges
        valid_times = [t for t in individual_times.values() if t is not None]
        transient_time = max(valid_times) if len(valid_times) == len(params) else None
        converged = len(valid_times) == len(params)
    else:
        # Convergence = when the FIRST parameter converges
        valid_times = [t for t in individual_times.values() if t is not None]
        transient_time = min(valid_times) if valid_times else None
        converged = len(valid_times) > 0
    
    return {
        'transient_time': transient_time,
        'converged': converged,
        'window_size': window_size,
        'threshold': threshold
    }
    
def combine_logs_to_transient_times(
    log_dir: str,
    output_csv: str,
    window_size: int = 50,
    threshold: float = 0.01,
    require_all: bool = True
):
    """
    Process all log CSVs in a directory to compute transient times and save summary.
    
    Args:
        log_dir: Directory with log CSV files
        output_csv: Path to save summary CSV
        window_size: Rolling window size for std calculation
        threshold: Std threshold for convergence
        require_all: If True, convergence only when ALL parameters converge
    """
    log_path = Path(log_dir)
    summary_records = []
    
    for csv_file in log_path.glob("*.csv"):
        params = {}
        parts = csv_file.stem.split('_')
        for part in parts:
            if part.startswith('N'):
                params['N'] = int(part[1:])
            elif part.startswith('J'):
                params['J'] = float(part[1:])
            elif part.startswith('K'):
                params['K'] = float(part[1:])
            elif part.startswith('seed'):
                params['seed'] = int(part[4:])
        
        results = calculate_transient_times(
            str(csv_file), 
            window_size=window_size, 
            threshold=threshold,
            require_all=require_all
        )
        
        record = {**params, **results}
        summary_records.append(record)
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved transient time summary to {output_csv}")

if __name__ == "__main__":
    # csv_path = "./temp_N200_J1.0_K0.0_seed0.csv"
    
    # # Calculate transient time only
    # results = calculate_transient_times(csv_path, window_size=50, threshold=0.01)
    # print(f"Transient time: {results['transient_time']}")
    
    combine_logs_to_transient_times(
        log_dir="./logs",
        output_csv="./results_data/transient_times_summary.csv",
        window_size=50,
        threshold=0.01,
        require_all=True
    )