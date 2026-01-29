import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Optional

# Plotting parameters
# Plotting parameters
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
sns.set(style="whitegrid")

# SIZE PARAMEERS
TITLESIZE = 16
LABELSIZE = 20
TICKSIZE = 18

def plot_phase_heatmap(
    csv_path: str = "test.csv", 
    out_path: Optional[str] = "plots/heatmap_state.png"
) -> None:
    """
    Generate a categorical heatmap of the system state over control parameters J and K.

    Args:
        csv_path: Path to the input CSV file containing simulation results.
        out_path: Path to save the output heatmap image. If None, plots to screen.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Filter for a single seed to ensure unique (J, K) pairs
    if 'seed' in df.columns:
        unique_seeds = df['seed'].unique()
        print(f"Found seeds: {unique_seeds}. Using seed={unique_seeds[0]} for plotting phase heatmap.")
        df = df[df['seed'] == unique_seeds[0]]
    df.drop_duplicates(subset=['J', 'K'], keep='last', inplace=True)

    # Round J and K to ensure they pivot correctly
    df['J'] = df['J'].round(3)
    df['K'] = df['K'].round(3)

    # Pivot the data: K on y-axis, J on x-axis
    pivot_df = df.pivot(index="J", columns="K", values="state")
    
    # Sort index and columns to ensure correct axis ordering
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.sort_index(axis=1, ascending=True)

    # Get unique states to define numerical mapping and colors
    # Handle NaN if any
    unique_states = sorted([x for x in df['state'].unique() if pd.notna(x)])
    state_to_num = {state: i for i, state in enumerate(unique_states)}
    
    # Map the pivot table to numbers
    pivot_num = pivot_df.map(lambda x: state_to_num.get(x, np.nan))

    # Define a discrete colormap
    colors = sns.color_palette("husl", len(unique_states))
    cmap = mcolors.ListedColormap(colors)
    
    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(
        pivot_num, 
        cmap=cmap, 
        cbar=False, 
        xticklabels=5, 
        yticklabels=5
    )
    
    # Correct the y-axis direction
    ax.invert_yaxis()
    
    plt.title("Phase Diagram: State over J vs K", fontsize=TITLESIZE)
    plt.xlabel("K", fontsize=LABELSIZE)
    plt.ylabel("J", fontsize=LABELSIZE)
    
    # Create the legend
    patches = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(unique_states))]
    plt.legend(patches, unique_states, title="State", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"State Heatmap saved to {out_path}")
    
    # plt.show()

def plot_S_heatmap(
    csv_path: str = "test.csv", 
    out_path: Optional[str] = "plots/heatmap_S.png"
) -> None:
    """
    Generate a heatmap of the order parameter S over J and K.

    Args:
        csv_path: Path to the input CSV file.
        out_path: Path to save the output image. If None, plots to screen.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Filter for a single seed
    if 'seed' in df.columns:
        unique_seeds = df['seed'].unique()
        print(f"Found seeds: {unique_seeds}. Using seed={unique_seeds[0]} for plotting S heatmap.")
        df = df[df['seed'] == unique_seeds[0]]
        
    df.drop_duplicates(subset=['J', 'K'], keep='last', inplace=True)

    # Round J and K
    df['J'] = df['J'].round(3)
    df['K'] = df['K'].round(3)

    # Pivot: K (index), J (columns), values (S)
    pivot_df = df.pivot(index="J", columns="K", values="S")
    
    # Sort
    pivot_df = pivot_df.sort_index(ascending=True) 
    pivot_df = pivot_df.sort_index(axis=1, ascending=True) 

    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(
        pivot_df, 
        cmap="viridis", 
        cbar_kws={'label': 'Order Parameter S'},
        xticklabels=LABELSIZE,
        yticklabels=LABELSIZE
    )
    
    ax.invert_yaxis()
    plt.title("Phase Diagram: Order Parameter S over J vs K", fontsize=TITLESIZE)
    plt.xlabel("K", fontsize=LABELSIZE)
    plt.ylabel("J", fontsize=LABELSIZE)
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"S Heatmap saved to {out_path}")
    
    # plt.show()

def plot_transient_time_summary(
    csv_path: str = "./results_data/transient_times_summary.csv",
    out_path: Optional[str] = "transient_times_summary.png"
) -> None:
    """
    Generate a heatmap of transient times over J and K using summary data.

    Args:
        csv_path: Path to the summary CSV file containing transient times.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Aggregate across seeds (averaging transient time)
    if 'seed' in df.columns:
        # We group by J, K and take the mean of transient_time (and any other numeric cols)
        df = df.groupby(['J', 'K'])['transient_time'].mean().reset_index()

    plt.figure(figsize=(10, 8))
    
    pivot_df = df.pivot(index="J", columns="K", values="transient_time")
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.sort_index(axis=1, ascending=True)

    ax = sns.heatmap(
        pivot_df, 
        cmap="cividis", 
        cbar_kws={'label': 'Transient Time'},
        xticklabels=LABELSIZE,
        yticklabels=LABELSIZE
    )
    
    ax.invert_yaxis()
    plt.title("Transient Time over J vs K", fontsize=TITLESIZE)
    plt.xlabel("K", fontsize=LABELSIZE)
    plt.ylabel("J", fontsize=LABELSIZE)
    
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Transient Time Heatmap saved to {out_path}")
    
    # plt.show()

def plot_transient_times(
    csv_path: str = "results_data/transient_times_static_async.csv", 
    out_path: Optional[str] = "transient_times.png"
) -> None:
    """
    Plot transient time vs N with statistical aggregation (mean and CI).

    Args:
        csv_path: Path to the input CSV file containing transient times.
        out_path: Path to save the output plot. If None, plots to screen.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
        
    # If using the threshold method csv, filter for converged only
    if 'converged' in df.columns:
        df = df[df['converged'] == True]

    plt.figure(figsize=(10, 6))
    
    # Plot line with confidence intervals
    # sns.lineplot automatically aggregates multiple y values for the same x value 
    # and plots the mean and a confidence interval (default 95%)
    sns.lineplot(data=df, x="N", y="transient_time", marker="o", errorbar=('ci', 95))
    
    plt.title("Transient Time vs N")
    plt.xlabel("N")
    plt.ylabel("Transient Time")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Transient Time Plot saved to {out_path}")
    
    # plt.show()


if __name__ == "__main__":
    # plot_transient_time_summary(csv_path="results_data/heatmap_transient_times_mser.csv", out_path="plots/heatmap_transient_times_mser.png")
    plot_phase_heatmap(csv_path='N200_30_seed.csv', out_path="plots/N200_30_seed_heatmap.png")
    # plot_S_heatmap(csv_path='sweep_18775448.csv', out_path="plots/heatmap_S_n_100.png")
    # plot_transient_times(csv_path="results_data/static_sync_transient_times_mser.csv", out_path="plots/static_sync_transient_times_mser.png")