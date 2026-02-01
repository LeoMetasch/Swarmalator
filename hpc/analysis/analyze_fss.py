"""
Finite-Size Scaling Analysis for Swarmalator Phase Transitions.

Analyzes K-sweep results at multiple N values to:
1. Find critical K where susceptibility peaks
2. Estimate critical exponents (β, γ, ν)
3. Attempt data collapse onto universal scaling function
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


from typing import Optional, Tuple


def load_ksweep_data(filepath: Path) -> pd.DataFrame:
    """Load K-sweep CSV and compute per-K statistics.
    
    Args:
        filepath: Path to the K-sweep CSV file.
    
    Returns:
        DataFrame with per-K aggregated statistics.
    """
    df = pd.read_csv(filepath)
    
    # Group by K and compute statistics
    stats = df.groupby('K').agg({
        'S': ['mean', 'std', 'var'],
        'R': ['mean', 'std', 'var'],
        'V': 'mean',
        'omega': 'mean',
    }).reset_index()
    stats.columns = ['K', 'S_mean', 'S_std', 'S_var', 'R_mean', 'R_std', 'R_var', 'V_mean', 'omega_mean']
    stats['N'] = df['K'].count() // len(stats)
    return stats


def find_critical_K(
    df: pd.DataFrame,
    column: str = 'chi_S',
    method: str = 'peak',
    threshold: float = 0.1,
    **kwargs
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Find the critical K value using specified method.
    
    Args:
        df: DataFrame with K and order parameter columns.
        column: Column name to analyze for criticality.
        method: Detection method ('peak', 'threshold', or 'rolling_threshold').
        threshold: Threshold value for threshold-based methods.
        **kwargs: Additional arguments (e.g., window for rolling methods).
    
    Returns:
        Tuple of (K_c, value_at_Kc, index) or (None, None, None) if not found.
    """
    if method == 'peak':
        if len(df) > 5:
            smoothed = savgol_filter(df[column].values, min(11, len(df) // 2 * 2 + 1), 3)
        else:
            smoothed = df[column].values
        
        idx_max = np.argmax(smoothed)
        K_c = df['K'].iloc[idx_max]
        val_at_Kc = df[column].iloc[idx_max]
        return K_c, val_at_Kc, idx_max
        
    elif method == 'threshold':
        above = df[df[column] > threshold]
        if not above.empty:
            return above['K'].iloc[0], above[column].iloc[0], above.index[0]
        return None, None, None

    elif method == 'rolling_threshold':
        window = kwargs.get('window', 5)
        rolling = df[column].rolling(window=window, center=True).mean()
        above = df[rolling > threshold]
        if not above.empty:
            return above['K'].iloc[0], above[column].iloc[0], above.index[0]
        return None, None, None
            
    return None, None, None


def analyze_fss(data_dir: Path, N_values: list, output_dir: Path) -> None:
    """Perform finite-size scaling analysis on multiple N datasets.
    
    Args:
        data_dir: Directory containing K-sweep CSV files.
        N_values: List of N values to analyze.
        output_dir: Output directory for results and plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    all_data = {}
    
    for N in N_values:
        # Glob all seed files for this N
        pattern = f"ksweep_N{N}_seed*.csv"
        files = sorted(list(data_dir.glob(pattern)))
        
        print(f"N = {N} ({len(files)} files)")
        
        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        
        stats = combined.groupby('K').agg({
            'S': ['mean', 'std', 'var'],
            'R': ['mean', 'std', 'var'],
        }).reset_index()
        stats.columns = ['K', 'S_mean', 'S_std', 'S_var', 'R_mean', 'R_std', 'R_var']
        
        stats['chi_S'] = N * stats['S_var']
        stats['chi_R'] = N * stats['R_var']
        stats['N'] = N
        
        all_data[N] = stats
        
        K_c, _, _ = find_critical_K(stats, 'chi_S', method='rolling_threshold', threshold=0.25, window=5)
        
        if K_c is None:
            K_c, _, _ = find_critical_K(stats, 'chi_S', method='peak')

        K_c_S = K_c
        K_c_R = K_c
        
        _, chi_max_S, _ = find_critical_K(stats, 'chi_S', method='peak')
        _, chi_max_R, _ = find_critical_K(stats, 'chi_R', method='peak')
        
        S_at_Kc = stats.loc[stats['K'] == K_c_S, 'S_mean'].values[0]
        R_at_Kc = stats.loc[stats['K'] == K_c_R, 'R_mean'].values[0]
        
        results.append({
            'N': N,
            'K_c_S': K_c_S,
            'K_c_R': K_c_R,
            'chi_max_S': chi_max_S,
            'chi_max_R': chi_max_R,
            'S_at_Kc': S_at_Kc,
            'R_at_Kc': R_at_Kc,
        })
        
        print(f"  K_c = {K_c_R:.4f}, χ_max(R) = {chi_max_R:.4f}")
    
    if len(results) < 2:
        print("Need at least 2 N values")
        return
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "fss_summary.csv", index=False)
    
    N_arr = results_df['N'].values
    K_c_arr = results_df['K_c_S'].values
    chi_max_arr = results_df['chi_max_S'].values
    S_at_Kc_arr = results_df['S_at_Kc'].values
    
    # Estimate K_c(∞)
    try:
        popt, _ = curve_fit(lambda x, a, b: a + b*x, 1/N_arr, K_c_arr)
        K_c_inf = popt[0]
    except:
        K_c_inf = np.mean(K_c_arr)
    
    # Estimate γ/ν
    try:
        log_N = np.log(N_arr)
        log_chi = np.log(chi_max_arr)
        popt_chi, _ = curve_fit(lambda x, a, b: a + b*x, log_N, log_chi)
        gamma_over_nu = popt_chi[1]
    except:
        gamma_over_nu = None
    
    # Estimate β/ν
    try:
        log_order = np.log(S_at_Kc_arr)
        popt_beta, _ = curve_fit(lambda x, a, b: a + b*x, log_N, log_order)
        neg_beta_over_nu = popt_beta[1]
    except:
        neg_beta_over_nu = None
    
    print(f"\nK_c(∞) ≈ {K_c_inf:.4f}")
    if gamma_over_nu:
        print(f"γ/ν ≈ {gamma_over_nu:.3f}")
    if neg_beta_over_nu:
        print(f"β/ν ≈ {-neg_beta_over_nu:.3f}")
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    for N, stats in all_data.items():
        ax.plot(stats['K'], stats['S_mean'], label=f'N={N}')
    ax.axvline(K_c_inf, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('K')
    ax.set_ylabel('S')
    ax.set_title('Order Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for N, stats in all_data.items():
        ax.plot(stats['K'], stats['chi_S'], label=f'N={N}')
    ax.axvline(K_c_inf, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('K')
    ax.set_ylabel('χ_S')
    ax.set_title('Susceptibility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fss_analysis.png', dpi=150)
    plt.close()

def main() -> None:
    """Parse CLI arguments and run finite-size scaling analysis."""
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="results/fss")
    p.add_argument("--N_values", type=int, nargs='+', default=[100, 200, 400])
    p.add_argument("--output_dir", type=str, default="results/fss_analysis")
    
    args = p.parse_args()
    
    analyze_fss(
        data_dir=Path(args.data_dir),
        N_values=args.N_values,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
