"""
Analyze a single K-sweep CSV file to identify phase transitions and critical points.

Generates:
1. Order parameter plots (S, R, V, omega vs K)
2. Susceptibility/variance plots (peaks indicate critical K)
3. dS/dK derivative plot (steep changes at transitions)
4. State transition timeline
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter


def analyze_ksweep(filepath: Path, output_dir: Path):
    """Analyze a K-sweep CSV and generate diagnostic plots."""
    
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} rows")
    
    # Get unique K values
    K_values = df['K'].unique()
    print(f"  {len(K_values)} unique K values: [{K_values.min():.3f}, {K_values.max():.3f}]")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================
    # Per-K statistics
    # =========================================================
    print("Computing per-K statistics...")
    stats = df.groupby('K').agg({
        'S': ['mean', 'std', 'var', 'min', 'max'],
        'R': ['mean', 'std', 'var', 'min', 'max'],
        'V': ['mean', 'std'],
        'omega': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    stats.columns = [
        'K', 
        'S_mean', 'S_std', 'S_var', 'S_min', 'S_max',
        'R_mean', 'R_std', 'R_var', 'R_min', 'R_max',
        'V_mean', 'V_std',
        'omega_mean', 'omega_std',
    ]
    
    # Estimate N from pattern (steps_per_K)
    steps_per_K = len(df) // len(K_values)
    print(f"  Estimated steps_per_K = {steps_per_K}")
    
    # Susceptibility
    N_estimated = 50  # rough estimate, adjust if needed
    stats['chi_S'] = N_estimated * stats['S_var']
    stats['chi_R'] = N_estimated * stats['R_var']
    
    # Numerical derivatives
    dK = K_values[1] - K_values[0] if len(K_values) > 1 else 0.01
    stats['dS_dK'] = np.gradient(stats['S_mean'], dK)
    stats['dR_dK'] = np.gradient(stats['R_mean'], dK)
    
    # Save stats
    stats.to_csv(output_dir / 'ksweep_stats.csv', index=False)
    print(f"  Saved stats to {output_dir / 'ksweep_stats.csv'}")
    
    # =========================================================
    # Find critical K candidates
    # =========================================================
    print("\nSearching for critical points...")
    
    # Where chi_S peaks
    chi_S_smooth = savgol_filter(stats['chi_S'].values, min(21, len(stats) // 2 * 2 + 1), 3)
    idx_chi_max = np.argmax(chi_S_smooth)
    K_c_chi = stats['K'].iloc[idx_chi_max]
    print(f"  χ_S peak at K ≈ {K_c_chi:.4f}")
    
    # Where |dS/dK| is largest
    dS_smooth = savgol_filter(np.abs(stats['dS_dK'].values), min(21, len(stats) // 2 * 2 + 1), 3)
    idx_dS_max = np.argmax(dS_smooth)
    K_c_dS = stats['K'].iloc[idx_dS_max]
    print(f"  |dS/dK| peak at K ≈ {K_c_dS:.4f}")
    
    # Where S_std is largest (fluctuations)
    idx_std_max = stats['S_std'].argmax()
    K_c_std = stats['K'].iloc[idx_std_max]
    print(f"  S_std peak at K ≈ {K_c_std:.4f}")
    
    # =========================================================
    # Plots
    # =========================================================
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # 1. S vs K (with mean ± std band)
    ax = axes[0, 0]
    ax.fill_between(stats['K'], stats['S_min'], stats['S_max'], alpha=0.2, color='blue', label='min-max')
    ax.fill_between(stats['K'], stats['S_mean'] - stats['S_std'], stats['S_mean'] + stats['S_std'], alpha=0.4, color='blue', label='±1σ')
    ax.plot(stats['K'], stats['S_mean'], 'b-', linewidth=1.5, label='mean')
    ax.axvline(K_c_chi, color='r', linestyle='--', alpha=0.7, label=f'K_c≈{K_c_chi:.3f}')
    ax.set_xlabel('K')
    ax.set_ylabel('S (correlation order parameter)')
    ax.set_title('Order Parameter S vs K')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. R vs K
    ax = axes[0, 1]
    ax.fill_between(stats['K'], stats['R_min'], stats['R_max'], alpha=0.2, color='green')
    ax.fill_between(stats['K'], stats['R_mean'] - stats['R_std'], stats['R_mean'] + stats['R_std'], alpha=0.4, color='green')
    ax.plot(stats['K'], stats['R_mean'], 'g-', linewidth=1.5)
    ax.axvline(K_c_chi, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('K')
    ax.set_ylabel('R (synchrony order parameter)')
    ax.set_title('Order Parameter R vs K')
    ax.grid(True, alpha=0.3)
    
    # 3. Susceptibility χ_S vs K
    ax = axes[1, 0]
    ax.plot(stats['K'], stats['chi_S'], 'b-', linewidth=1, alpha=0.5)
    ax.plot(stats['K'], chi_S_smooth, 'b-', linewidth=2, label='smoothed')
    ax.axvline(K_c_chi, color='r', linestyle='--', alpha=0.7, label=f'peak at {K_c_chi:.3f}')
    ax.set_xlabel('K')
    ax.set_ylabel('χ_S (susceptibility)')
    ax.set_title('Susceptibility χ_S = N × Var(S)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. |dS/dK| vs K  
    ax = axes[1, 1]
    ax.plot(stats['K'], np.abs(stats['dS_dK']), 'purple', linewidth=1, alpha=0.5)
    ax.plot(stats['K'], dS_smooth, 'purple', linewidth=2, label='smoothed')
    ax.axvline(K_c_dS, color='r', linestyle='--', alpha=0.7, label=f'peak at {K_c_dS:.3f}')
    ax.set_xlabel('K')
    ax.set_ylabel('|dS/dK|')
    ax.set_title('Rate of Change of S')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. V and omega vs K
    ax = axes[2, 0]
    ax.plot(stats['K'], stats['V_mean'], 'orange', linewidth=1.5, label='V (spatial velocity)')
    ax.plot(stats['K'], stats['omega_mean'], 'red', linewidth=1.5, label='ω (phase velocity)')
    ax.axvline(K_c_chi, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('K')
    ax.set_ylabel('Velocity')
    ax.set_title('Mean Velocities vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. S time series for selected K values
    ax = axes[2, 1]
    K_samples = [K_values[0], K_c_chi, K_values[-1]]
    colors = ['blue', 'red', 'green']
    for K_sample, color in zip(K_samples, colors):
        subset = df[np.isclose(df['K'], K_sample, atol=dK/2)]
        if len(subset) > 0:
            ax.plot(subset['t_at_K'].values[:min(500, len(subset))], 
                   subset['S'].values[:min(500, len(subset))], 
                   color=color, alpha=0.7, linewidth=0.5, label=f'K={K_sample:.2f}')
    ax.set_xlabel('t (within K window)')
    ax.set_ylabel('S')
    ax.set_title('S Time Series at Selected K Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ksweep_analysis.png', dpi=150)
    plt.close()
    print(f"  Saved {output_dir / 'ksweep_analysis.png'}")
    
    # =========================================================
    # State transitions
    # =========================================================
    print("\nState transition analysis...")
    state_counts = df.groupby(['K', 'state']).size().unstack(fill_value=0)
    dominant_states = state_counts.idxmax(axis=1)
    
    # Find K where state changes
    transitions = []
    prev_state = None
    for K, state in dominant_states.items():
        if state != prev_state and prev_state is not None:
            transitions.append((K, prev_state, state))
        prev_state = state
    
    print("  State transitions detected:")
    for K, from_state, to_state in transitions:
        print(f"    K ≈ {K:.4f}: {from_state} → {to_state}")
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Critical K candidates:")
    print(f"  From χ_S peak:      K_c ≈ {K_c_chi:.4f}")
    print(f"  From |dS/dK| peak:  K_c ≈ {K_c_dS:.4f}")
    print(f"  From σ(S) peak:     K_c ≈ {K_c_std:.4f}")
    print(f"\nOutput saved to: {output_dir}")
    print("=" * 60)
    
    return stats


def main():
    p = argparse.ArgumentParser(description="Analyze a K-sweep CSV file")
    p.add_argument("filepath", type=str, help="Path to ksweep CSV file")
    p.add_argument("--output", type=str, default=None, help="Output directory (default: same as input)")
    
    args = p.parse_args()
    
    filepath = Path(args.filepath)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = filepath.parent / f"{filepath.stem}_analysis"
    
    analyze_ksweep(filepath, output_dir)


if __name__ == "__main__":
    main()
