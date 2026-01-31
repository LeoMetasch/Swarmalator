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
        raise ValueError(f"No CSV files in {data_dir}")
    
    return pd.concat(all_dfs, ignore_index=True)


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby('K').agg({
        'S': ['mean', 'std', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'R': ['mean', 'std'],
        'V': ['mean'],
        'omega': ['mean'],
    }).reset_index()
    
    # Flatten columns
    stats.columns = ['K', 'S_mean', 'S_std', 'S_median', 'S_q25', 'S_q75', 
                     'R_mean', 'R_std', 'V_mean', 'omega_mean']
    stats['chi_S'] = 50 * df.groupby('K')['S'].var().values
    return stats




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
     
    # Save stats
    stats_fwd.to_csv(output_dir / 'stats_forward.csv', index=False)
    stats_bwd.to_csv(output_dir / 'stats_backward.csv', index=False)
    
    plt.plot(stats_fwd['K'], stats_fwd['S_mean'], 'b-', linewidth=2, label='Forward (K↑)')
    plt.fill_between(stats_fwd['K'], stats_fwd['S_q25'], stats_fwd['S_q75'], 
                    alpha=0.3, color='blue', label='Forward IQR')
    plt.plot(stats_bwd['K'], stats_bwd['S_mean'], 'r-', linewidth=2, label='Backward (K↓)')
    plt.fill_between(stats_bwd['K'], stats_bwd['S_q25'], stats_bwd['S_q75'], 
                    alpha=0.3, color='red')
    plt.xlabel('K')
    plt.ylabel('S')
    plt.title('S vs K')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hysteresis_analysis.png', dpi=150)
    plt.close()


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
