"""
Analyze hysteresis data: Compare forward vs backward K-sweeps.

Generates:
1. Overlay plot of S vs K for forward/backward with confidence bands
2. Hysteresis loop visualization
3. Critical K estimates for each direction
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter


def load_sweep_data(data_dir: Path) -> pd.DataFrame:
    """Load all CSV files from a directory and combine them."""
    all_dfs = []
    csv_files = sorted(data_dir.glob("*.csv"))
    
    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        # Extract seed from filename
        seed = int(f.stem.split("seed")[-1])
        df['seed'] = seed
        df['run'] = i
        all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Loaded {len(csv_files)} files, {len(combined):,} total rows")
    return combined


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-K statistics across all runs."""
    stats = df.groupby('K').agg({
        'S': ['mean', 'std', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'R': ['mean', 'std'],
        'V': ['mean'],
        'omega': ['mean'],
    }).reset_index()
    
    # Flatten columns
    stats.columns = ['K', 'S_mean', 'S_std', 'S_median', 'S_q25', 'S_q75', 
                     'R_mean', 'R_std', 'V_mean', 'omega_mean']
    
    # Susceptibility
    stats['chi_S'] = 50 * df.groupby('K')['S'].var().values  # N=50
    
    return stats


def find_critical_K(stats: pd.DataFrame) -> float:
    """Find K where susceptibility peaks."""
    window = min(21, len(stats) // 2 * 2 + 1)
    if window < 5:
        window = 5
    chi_smooth = savgol_filter(stats['chi_S'].values, window, 3)
    idx = np.argmax(chi_smooth)
    return stats['K'].iloc[idx]


def analyze_hysteresis(forward_dir: Path, backward_dir: Path, output_dir: Path):
    """Main analysis function."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading forward sweep data...")
    df_fwd = load_sweep_data(forward_dir)
    
    print("Loading backward sweep data...")
    df_bwd = load_sweep_data(backward_dir)
    
    # Compute stats
    print("Computing statistics...")
    stats_fwd = compute_stats(df_fwd)
    stats_bwd = compute_stats(df_bwd)
    
    # Find critical K
    K_c_fwd = find_critical_K(stats_fwd)
    K_c_bwd = find_critical_K(stats_bwd)
    
    print(f"\nCritical K estimates:")
    print(f"  Forward (K↑):  K_c ≈ {K_c_fwd:.4f}")
    print(f"  Backward (K↓): K_c ≈ {K_c_bwd:.4f}")
    print(f"  Hysteresis width: ΔK ≈ {abs(K_c_fwd - K_c_bwd):.4f}")
    
    # Save stats
    stats_fwd.to_csv(output_dir / 'stats_forward.csv', index=False)
    stats_bwd.to_csv(output_dir / 'stats_backward.csv', index=False)
    
    # =========================================================
    # Plots
    # =========================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. S vs K with confidence bands
    ax = axes[0, 0]
    # Forward
    ax.fill_between(stats_fwd['K'], stats_fwd['S_q25'], stats_fwd['S_q75'], 
                    alpha=0.3, color='blue', label='Forward IQR')
    ax.plot(stats_fwd['K'], stats_fwd['S_mean'], 'b-', linewidth=2, label='Forward (K↑)')
    # Backward
    ax.fill_between(stats_bwd['K'], stats_bwd['S_q25'], stats_bwd['S_q75'], 
                    alpha=0.3, color='red')
    ax.plot(stats_bwd['K'], stats_bwd['S_mean'], 'r-', linewidth=2, label='Backward (K↓)')
    # Critical K markers
    ax.axvline(K_c_fwd, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(K_c_bwd, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('K')
    ax.set_ylabel('S (order parameter)')
    ax.set_title('Order Parameter S vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Hysteresis loop (S_fwd vs S_bwd at same K)
    ax = axes[0, 1]
    # Create common K grid and interpolate both onto it
    K_common = np.linspace(
        max(stats_fwd['K'].min(), stats_bwd['K'].min()),
        min(stats_fwd['K'].max(), stats_bwd['K'].max()),
        200
    )
    S_fwd_interp = np.interp(K_common, stats_fwd['K'].values, stats_fwd['S_mean'].values)
    S_bwd_interp = np.interp(K_common, stats_bwd['K'].values, stats_bwd['S_mean'].values)
    
    ax.plot(S_fwd_interp, S_bwd_interp, 'purple', linewidth=1)
    if len(K_common) > 0:
        ax.scatter(S_fwd_interp[0], S_bwd_interp[0], 
                   color='green', s=100, zorder=5, label=f'K={K_common[0]:.2f}')
        ax.scatter(S_fwd_interp[-1], S_bwd_interp[-1], 
                   color='red', s=100, zorder=5, label=f'K={K_common[-1]:.2f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No hysteresis')
    ax.set_xlabel('S (Forward)')
    ax.set_ylabel('S (Backward)')
    ax.set_title('Hysteresis Loop')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Susceptibility comparison
    ax = axes[1, 0]
    ax.plot(stats_fwd['K'], stats_fwd['chi_S'], 'b-', linewidth=2, label='Forward')
    ax.plot(stats_bwd['K'], stats_bwd['chi_S'], 'r-', linewidth=2, label='Backward')
    ax.axvline(K_c_fwd, color='blue', linestyle='--', alpha=0.5, label=f'K_c↑={K_c_fwd:.3f}')
    ax.axvline(K_c_bwd, color='red', linestyle='--', alpha=0.5, label=f'K_c↓={K_c_bwd:.3f}')
    ax.set_xlabel('K')
    ax.set_ylabel('χ_S (susceptibility)')
    ax.set_title('Susceptibility vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Difference plot (S_fwd - S_bwd)
    ax = axes[1, 1]
    S_diff = S_fwd_interp - S_bwd_interp
    ax.plot(K_common, S_diff, 'purple', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(K_c_fwd, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(K_c_bwd, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(K_common, 0, S_diff, 
                    where=S_diff > 0, 
                    alpha=0.3, color='blue', label='Forward > Backward')
    ax.fill_between(K_common, 0, S_diff, 
                    where=S_diff < 0, 
                    alpha=0.3, color='red', label='Backward > Forward')
    ax.set_xlabel('K')
    ax.set_ylabel('S_forward - S_backward')
    ax.set_title('Hysteresis Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hysteresis_analysis.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {output_dir / 'hysteresis_analysis.png'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Forward K_c:      {K_c_fwd:.4f}")
    print(f"Backward K_c:     {K_c_bwd:.4f}")
    print(f"Hysteresis width: {abs(K_c_fwd - K_c_bwd):.4f}")
    if K_c_fwd > K_c_bwd:
        print("→ Forward transition occurs at HIGHER K")
        print("  (disorder→order delayed when increasing K)")
    else:
        print("→ Backward transition occurs at HIGHER K")
        print("  (order→disorder delayed when decreasing K)")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="Analyze hysteresis K-sweep data")
    p.add_argument("--forward_dir", type=str, default="results/hysteresis/forward",
                   help="Directory with forward sweep CSVs")
    p.add_argument("--backward_dir", type=str, default="results/hysteresis/backward",
                   help="Directory with backward sweep CSVs")
    p.add_argument("--output_dir", type=str, default="results/hysteresis/analysis",
                   help="Output directory for plots and stats")
    
    args = p.parse_args()
    
    analyze_hysteresis(Path(args.forward_dir), Path(args.backward_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
