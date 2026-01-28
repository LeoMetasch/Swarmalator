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
import warnings

def load_ksweep_data(filepath: Path) -> pd.DataFrame:
    """Load K-sweep CSV and compute per-K statistics."""
    df = pd.read_csv(filepath)
    
    # Group by K and compute statistics
    stats = df.groupby('K').agg({
        'S': ['mean', 'std', 'var'],
        'R': ['mean', 'std', 'var'],
        'V': 'mean',
        'omega': 'mean',
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['K', 'S_mean', 'S_std', 'S_var', 'R_mean', 'R_std', 'R_var', 'V_mean', 'omega_mean']
    
    # Extract N from the data
    stats['N'] = df['K'].count() // len(stats)  # Approximate
    
    return stats


def compute_susceptibility(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """Compute susceptibility χ = N * Var(S)."""
    df = df.copy()
    df['chi_S'] = N * df['S_var']
    df['chi_R'] = N * df['R_var']
    return df


def find_critical_K(df: pd.DataFrame, column: str = 'chi_S') -> tuple:
    """Find K where susceptibility peaks."""
    # Smooth to reduce noise
    if len(df) > 5:
        smoothed = savgol_filter(df[column].values, min(11, len(df) // 2 * 2 + 1), 3)
    else:
        smoothed = df[column].values
    
    idx_max = np.argmax(smoothed)
    K_c = df['K'].iloc[idx_max]
    chi_max = df[column].iloc[idx_max]
    
    return K_c, chi_max, idx_max


def analyze_fss(data_dir: Path, N_values: list, output_dir: Path):
    """
    Perform finite-size scaling analysis on multiple N datasets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    all_data = {}
    
    print("=" * 60)
    print("FINITE-SIZE SCALING ANALYSIS")
    print("=" * 60)
    
    # Load and analyze each N
    for N in N_values:
        filepath = data_dir / f"ksweep_N{N}.csv"
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping N={N}")
            continue
        
        print(f"\nAnalyzing N = {N}...")
        df = pd.read_csv(filepath)
        
        # Compute per-K statistics
        stats = df.groupby('K').agg({
            'S': ['mean', 'std', 'var'],
            'R': ['mean', 'std', 'var'],
        }).reset_index()
        stats.columns = ['K', 'S_mean', 'S_std', 'S_var', 'R_mean', 'R_std', 'R_var']
        
        # Compute susceptibility
        stats['chi_S'] = N * stats['S_var']
        stats['chi_R'] = N * stats['R_var']
        stats['N'] = N
        
        all_data[N] = stats
        
        # Find critical K
        K_c_S, chi_max_S, _ = find_critical_K(stats, 'chi_S')
        K_c_R, chi_max_R, _ = find_critical_K(stats, 'chi_R')
        
        # Get S at the critical point
        S_at_Kc = stats.loc[stats['K'] == K_c_S, 'S_mean'].values[0]
        
        results.append({
            'N': N,
            'K_c_S': K_c_S,
            'K_c_R': K_c_R,
            'chi_max_S': chi_max_S,
            'chi_max_R': chi_max_R,
            'S_at_Kc': S_at_Kc,
        })
        
        print(f"  K_c (from χ_S peak) = {K_c_S:.4f}")
        print(f"  K_c (from χ_R peak) = {K_c_R:.4f}")
        print(f"  χ_max(S) = {chi_max_S:.4f}")
        print(f"  S at K_c = {S_at_Kc:.4f}")
    
    if len(results) < 2:
        print("\nNeed at least 2 N values for scaling analysis!")
        return
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "fss_summary.csv", index=False)
    
    # =========================================================
    # SCALING ANALYSIS
    # =========================================================
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    
    N_arr = results_df['N'].values
    K_c_arr = results_df['K_c_S'].values
    chi_max_arr = results_df['chi_max_S'].values
    S_at_Kc_arr = results_df['S_at_Kc'].values
    
    # 1. Estimate K_c(∞) from K_c(N) vs 1/N
    print("\n1. Estimating K_c(∞) from finite-size shift...")
    try:
        # Fit: K_c(N) = K_c(∞) + a/N^(1/ν)
        # Try ν = 1 first (mean-field-like)
        popt, _ = curve_fit(lambda x, a, b: a + b*x, 1/N_arr, K_c_arr)
        K_c_inf = popt[0]
        print(f"  K_c(∞) ≈ {K_c_inf:.4f} (linear extrapolation in 1/N)")
    except Exception as e:
        K_c_inf = np.mean(K_c_arr)
        print(f"  Fit failed, using mean: K_c ≈ {K_c_inf:.4f}")
    
    # 2. Estimate γ/ν from χ_max ~ N^(γ/ν)
    print("\n2. Estimating γ/ν from χ_max scaling...")
    try:
        log_N = np.log(N_arr)
        log_chi = np.log(chi_max_arr)
        popt, _ = curve_fit(lambda x, a, b: a + b*x, log_N, log_chi)
        gamma_over_nu = popt[1]
        print(f"  χ_max ~ N^({gamma_over_nu:.3f})")
        print(f"  → γ/ν ≈ {gamma_over_nu:.3f}")
    except Exception as e:
        gamma_over_nu = None
        print(f"  Fit failed: {e}")
    
    # 3. Estimate β/ν from S(K_c) ~ N^(-β/ν)
    print("\n3. Estimating β/ν from order parameter scaling...")
    try:
        log_S = np.log(S_at_Kc_arr)
        popt, _ = curve_fit(lambda x, a, b: a + b*x, log_N, log_S)
        neg_beta_over_nu = popt[1]
        print(f"  S(K_c) ~ N^({neg_beta_over_nu:.3f})")
        print(f"  → β/ν ≈ {-neg_beta_over_nu:.3f}")
    except Exception as e:
        neg_beta_over_nu = None
        print(f"  Fit failed: {e}")
    
    # =========================================================
    # PLOTS
    # =========================================================
    print(f"\n4. Generating plots in {output_dir}...")
    
    # Plot 1: S(K) for all N
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    for N, stats in all_data.items():
        ax.plot(stats['K'], stats['S_mean'], label=f'N={N}')
    ax.axvline(K_c_inf, color='k', linestyle='--', alpha=0.5, label=f'K_c≈{K_c_inf:.3f}')
    ax.set_xlabel('K')
    ax.set_ylabel('S (order parameter)')
    ax.set_title('Order Parameter vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: χ_S(K) for all N
    ax = axes[0, 1]
    for N, stats in all_data.items():
        ax.plot(stats['K'], stats['chi_S'], label=f'N={N}')
    ax.axvline(K_c_inf, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('K')
    ax.set_ylabel('χ_S (susceptibility)')
    ax.set_title('Susceptibility vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: K_c vs 1/N
    ax = axes[1, 0]
    ax.plot(1/N_arr, K_c_arr, 'o-', markersize=8)
    ax.axhline(K_c_inf, color='r', linestyle='--', label=f'K_c(∞)≈{K_c_inf:.3f}')
    ax.set_xlabel('1/N')
    ax.set_ylabel('K_c (from χ peak)')
    ax.set_title('Critical K Finite-Size Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: χ_max vs N (log-log)
    ax = axes[1, 1]
    ax.loglog(N_arr, chi_max_arr, 'o-', markersize=8)
    if gamma_over_nu is not None:
        fit_line = np.exp(popt[0]) * N_arr**gamma_over_nu
        ax.loglog(N_arr, fit_line, 'r--', label=f'N^{gamma_over_nu:.2f}')
    ax.set_xlabel('N')
    ax.set_ylabel('χ_max')
    ax.set_title('Susceptibility Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fss_analysis.png', dpi=150)
    plt.close()
    
    # =========================================================
    # DATA COLLAPSE ATTEMPT
    # =========================================================
    if gamma_over_nu is not None and neg_beta_over_nu is not None:
        print("\n5. Attempting data collapse...")
        
        nu_guess = 1.0  # Start with mean-field
        beta_over_nu = -neg_beta_over_nu
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for N, stats in all_data.items():
            # Rescaled variables
            x_scaled = (stats['K'] - K_c_inf) * (N ** (1/nu_guess))
            y_scaled = stats['S_mean'] * (N ** beta_over_nu)
            ax.plot(x_scaled, y_scaled, '.', label=f'N={N}', alpha=0.7)
        
        ax.set_xlabel(f'(K - K_c) × N^(1/ν), ν={nu_guess}')
        ax.set_ylabel(f'S × N^(β/ν), β/ν={beta_over_nu:.2f}')
        ax.set_title('Data Collapse Attempt')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'data_collapse.png', dpi=150)
        plt.close()
        print(f"  Saved data collapse plot")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  K_c(∞) ≈ {K_c_inf:.4f}")
    if gamma_over_nu:
        print(f"  γ/ν ≈ {gamma_over_nu:.3f}")
    if neg_beta_over_nu:
        print(f"  β/ν ≈ {-neg_beta_over_nu:.3f}")
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="Finite-size scaling analysis")
    p.add_argument("--data_dir", type=str, default="results/fss", 
                   help="Directory containing ksweep_N*.csv files")
    p.add_argument("--N_values", type=int, nargs='+', default=[50, 100, 200, 400],
                   help="N values to analyze")
    p.add_argument("--output_dir", type=str, default="results/fss_analysis",
                   help="Output directory for analysis results")
    
    args = p.parse_args()
    
    analyze_fss(
        data_dir=Path(args.data_dir),
        N_values=args.N_values,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
