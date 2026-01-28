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

def calculate_transient_time_mser(series: pd.Series) -> int:
    """
    Calculate transient time using the Marginal Standard Error Rule (MSER)
    combined with the Kneedle (Elbow) algorithm.
    
    1. Calculate MSER curve for all potential truncation points d.
    2. Find the 'Elbow' of the curve to balance error reduction vs data loss.
       This avoids the common failure mode where MSER monotonically decreases
       for converging systems.
    
    Args:
        series: Time series data.
        
    Returns:
        d: The optimal truncation point.
    """
    values = series.values
    n = len(values)
    if n < 10:
        return 0
        
    # Vectorized MSER Calculation
    # MSER(d) = Var(X_{d+1}..X_n) / (n-d)
    # Var = Mean(x^2) - Mean(x)^2
    
    # We compute stats for all suffixes.
    # Reverse array to use cumsum for suffixes
    rev_vals = values[::-1]
    cum_sum = np.cumsum(rev_vals)
    cum_sq = np.cumsum(rev_vals**2)
    
    # k goes from 5 to n (length of suffix)
    ks = np.arange(5, n + 1)
    
    # corresponding indices in cum arrays (0-based)
    idxs = ks - 1
    
    sums = cum_sum[idxs]
    sqs = cum_sq[idxs]
    
    means = sums / ks
    # Population variance
    variances = (sqs / ks) - (means**2)
    variances[variances < 0] = 0 # Numerical noise floor
    
    msers = variances / ks
    
    # msers array corresponds to suffix lengths k=5..n
    # d = n - k
    # So index 0 of msers is k=5 -> d = n-5 (end of series)
    # index last is k=n -> d=0 (start)
    
    # We want msers sorted by d (0 to n-5)
    msers_by_d = msers[::-1]
    ds = np.arange(len(msers_by_d))
    
    # Elbow Detection (Kneedle)
    d_min, d_max = ds[0], ds[-1]
    m_min, m_max = msers_by_d.min(), msers_by_d.max()
    
    if m_max <= m_min + 1e-15:
        # Flat curve, no transient or pure noise
        return 0
        
    # Normalize
    ds_norm = (ds - d_min) / (d_max - d_min)
    msers_norm = (msers_by_d - m_min) / (m_max - m_min)
    
    # Distance from line connecting start (0, y_start) to end (1, y_end)
    x1, y1 = ds_norm[0], msers_norm[0]
    x2, y2 = ds_norm[-1], msers_norm[-1]
    
    # Vectorized distance to chord
    # numerator = |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1|
    # denominator = sqrt((y2-y1)^2 + (x2-x1)^2)
    
    num = np.abs((y2 - y1) * ds_norm - (x2 - x1) * msers_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    distances = num / den
    
    best_idx = np.argmax(distances)
    best_d = ds[best_idx]
    
    return int(best_d)

def apply_mser_to_csv(csv_path: str) -> dict:
    """
    Apply MSER to all standard order parameters in the CSV and return the max transient time.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {'transient_time': None, 'converged': False}
        
    params = ['S', 'V', 'omega', 'R']
    transient_times = {}
    
    for param in params:
        if param in df.columns:
            d_index = calculate_transient_time_mser(df[param])
            transient_times[param] = df.loc[d_index, 'step'] if d_index < len(df) else df['step'].iloc[-1]
    
    valid_times = list(transient_times.values())
    overall_transient = max(valid_times) if valid_times else 0
    
    return {
        'transient_time': overall_transient,
        'individual_times': transient_times
    }

def combine_logs_mser(
    log_dir: str,
    output_csv: str
):
    """
    Process all logs using MSER and save summary.
    """
    log_path = Path(log_dir)
    summary_records = []
    
    csv_files = list(log_path.glob("*.csv"))
    total_files = len(csv_files)
    print(f"Processing {total_files} files with MSER...")
    
    for i, csv_file in enumerate(csv_files):
        if i % 100 == 0:
            print(f"Processed {i}/{total_files}")
            
        params = {}
        try:
            parts = csv_file.stem.split('_')
            for part in parts:
                if part.startswith('N'): params['N'] = int(part[1:])
                elif part.startswith('J'): params['J'] = float(part[1:])
                elif part.startswith('K'): params['K'] = float(part[1:])
                elif part.startswith('seed'): params['seed'] = int(part[4:])
        except ValueError:
            pass
            
        param_data = apply_mser_to_csv(str(csv_file))
        
        record = {**params, "transient_time": param_data['transient_time']}
        summary_records.append(record)
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved MSER transient time summary to {output_csv}")

if __name__ == "__main__":
    # csv_path = "./temp_N200_J1.0_K0.0_seed0.csv"
    
    # # Calculate transient time only
    # results = calculate_transient_times(csv_path, window_size=50, threshold=0.01)
    # print(f"Transient time: {results['transient_time']}")
    
    # combine_logs_to_transient_times(
    #     log_dir="./logs",
    #     output_csv="./results_data/transient_times_static_sync.csv",
    #     window_size=50,
    #     threshold=0.01,
    #     require_all=True
    # )
    
    combine_logs_mser(
        log_dir="./logs",
        output_csv="./results_data/transient_times_static_sync_mser.csv"
    )