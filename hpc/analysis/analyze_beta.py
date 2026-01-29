"""
Extract critical exponent β from ordered-phase scaling.

Analyzes S ~ |K - Kc|^β using log-log linear regression.
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from glob import glob


def analyze_beta(data_dir: Path, Kc_file: Path, output_dir: Path):
    """
    Extract β exponent from ordered-phase data via log-log fit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Kc
    with open(Kc_file) as f:
        Kc_data = json.load(f)
    # Kc = Kc_data["Kc_threshold"]
    Kc = -0.528
    print(f"Using Kc = {Kc:.4f}")
    
    # Load data files
    files = sorted(glob(str(data_dir / "*.csv")))
    if not files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(files)} data files")
    
    # Combine all data
    all_dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Compute per-K mean S
    stats_df = combined.groupby('K').agg({'S': ['mean', 'std']}).reset_index()
    stats_df.columns = ['K', 'S_mean', 'S_std']
    
    # Filter to ordered phase (K < Kc) and near critical region
    # |K - Kc| in [0.01, 0.15] for good scaling regime
    delta_K = np.abs(stats_df['K'] - Kc)
    mask = (stats_df['K'] < Kc) & (delta_K > 0.01) & (delta_K < 0.15)
    
    filtered = stats_df[mask].copy()
    
    if len(filtered) < 3:
        print(f"Not enough data points in scaling regime (found {len(filtered)})")
        print("Try adjusting K range or Kc value")
        return
    
    print(f"Using {len(filtered)} points in scaling regime")
    
    # Log-log transformation
    filtered['log_delta_K'] = np.log10(Kc - filtered['K'])  # K < Kc
    filtered['log_S'] = np.log10(filtered['S_mean'])
    
    # Linear regression: log(S) = β * log|K - Kc| + C
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered['log_delta_K'], filtered['log_S']
    )
    
    beta = slope
    beta_err = std_err
    r_squared = r_value**2
    
    print(f"\n{'='*50}")
    print(f"CRITICAL EXPONENT β ANALYSIS")
    print(f"{'='*50}")
    print(f"β = {beta:.4f} ± {beta_err:.4f}")
    print(f"R² = {r_squared:.4f}")
    print(f"Mean-field prediction: β = 0.5")
    print(f"Deviation from mean-field: {abs(beta - 0.5):.4f}")
    
    # Save results
    results = {
        "beta": float(beta),
        "beta_err": float(beta_err),
        "R_squared": float(r_squared),
        "Kc_used": float(Kc),
        "n_points": len(filtered),
        "K_range": [float(filtered['K'].min()), float(filtered['K'].max())],
    }
    
    with open(output_dir / "beta_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_dir / 'beta_results.json'}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw data with fit
    ax = axes[0]
    ax.errorbar(stats_df['K'], stats_df['S_mean'], yerr=stats_df['S_std'],
                fmt='o', alpha=0.5, label='All data')
    ax.errorbar(filtered['K'], filtered['S_mean'], yerr=filtered['S_std'],
                fmt='s', color='red', label='Fit region')
    ax.axvline(Kc, color='k', linestyle='--', label=f'Kc={Kc:.3f}')
    ax.set_xlabel('K')
    ax.set_ylabel('S')
    ax.set_title('Order Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log-log plot
    ax = axes[1]
    ax.scatter(filtered['log_delta_K'], filtered['log_S'], c='red', s=50, 
               label='Data')
    
    x_fit = np.linspace(filtered['log_delta_K'].min(), 
                        filtered['log_delta_K'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k-', linewidth=2, 
            label=f'Fit: β = {beta:.3f} ± {beta_err:.3f}\nR² = {r_squared:.3f}')
    
    ax.set_xlabel('log₁₀|K - Kc|')
    ax.set_ylabel('log₁₀(S)')
    ax.set_title(f'Log-Log Scaling: S ~ |K - Kc|^β')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'beta_analysis.png', dpi=150)
    plt.close()
    print(f"Saved plot: {output_dir / 'beta_analysis.png'}")


def main():
    p = argparse.ArgumentParser(description="Extract critical exponent β")
    p.add_argument("--data_dir", type=str, default="results/exp1_beta",
                   help="Directory containing K-sweep CSV files")
    p.add_argument("--Kc_file", type=str, default="results/phase1/Kc.json",
                   help="JSON file with Kc value")
    p.add_argument("--output_dir", type=str, default="results/exp1_beta",
                   help="Output directory")
    
    args = p.parse_args()
    analyze_beta(Path(args.data_dir), Path(args.Kc_file), Path(args.output_dir))


if __name__ == "__main__":
    main()
