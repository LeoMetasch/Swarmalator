"""
Plotting functions for Swarmalator simulation results.
Generates heatmaps and line plots for various order parameters and transient times.
"""

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

def plot_param_heatmap(
    csv_path: str = "test.csv",
    param: str = "S",
    out_path: Optional[str] = "plots/heatmap_param.png"
) -> None:
    """
    Generate a heatmap of a specific order parameter over J and K, averaged across seeds.

    Args:
        csv_path: Path to the input CSV file.
        param: The column name of the parameter to plot (e.g., 'S', 'V', 'R', 'omega').
        out_path: Path to save the output image. If None, plots to screen.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    if param not in df.columns:
        print(f"Error: Parameter '{param}' not found in CSV columns: {df.columns.tolist()}")
        return

    # Check columns
    required = ['J', 'K']
    if not all(col in df.columns for col in required):
        print(f"Error: CSV must contain columns {required}")
        return

    # Group by J and K and calculate mean of the parameter
    # This averages across all seeds present for each (J, K) pair
    df_agg = df.groupby(['J', 'K'])[param].mean().reset_index()

    # Round J and K to ensure they pivot correctly
    df_agg['J'] = df_agg['J'].round(3)
    df_agg['K'] = df_agg['K'].round(3)

    # Pivot: K (index), J (columns), values (param)
    pivot_df = df_agg.pivot(index="J", columns="K", values=param)

    # Sort
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.sort_index(axis=1, ascending=True)

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(
        pivot_df,
        cmap="viridis",
        cbar_kws={'label': f'Mean {param}'},
        xticklabels=LABELSIZE,
        yticklabels=LABELSIZE
    )

    ax.invert_yaxis()
    plt.title(f"Phase Diagram: Mean {param} over J vs K", fontsize=TITLESIZE)
    plt.xlabel("K", fontsize=LABELSIZE)
    plt.ylabel("J", fontsize=LABELSIZE)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"{param} Heatmap saved to {out_path}")

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
    """Plot transient time versus $N$ with mean and confidence intervals (95%).

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

    plt.title("Transient Time vs N (J=0.2, K=-0.5)")
    plt.xlabel("N")
    plt.ylabel("Transient Time")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Transient Time Plot saved to {out_path}")

    # plt.show()

def plot_order_parameters_vs_K(
    csv_path: str,
    j_values: list[float],
    out_path: Optional[str] = None
) -> None:
    """Plot order parameters versus $K$ for multiple $J$ values.

    Args:
        csv_path: Path to the input CSV file.
        j_values: List of J values to filter by and plot.
        out_path: Path to save the output image. If None, plots to screen.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    metrics = ['R', 'S', 'V', 'omega']
    titles = {
        'R': 'Synchrony (R)',
        'S': 'Correlation (S)',
        'V': 'Mean Spatial Velocity (V)',
        'omega': 'Mean Phase Velocity (omega)'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for j_val in j_values:
        # df_j = df[np.isclose(df['J'], j_val)] if using floats carefully
        # Simple equality for now
        df_j = df[df['J'] == j_val].copy()

        if df_j.empty:
            print(f"No data found for J={j_val}")
            continue

        for i, metric in enumerate(metrics):
            ax = axes[i]
            if metric in df_j.columns:
                sns.lineplot(
                    data=df_j,
                    x='K',
                    y=metric,
                    ax=ax,
                    marker='o',
                    errorbar=('ci', 95),
                    label=f"J={j_val}"
                )
                ax.set_title(titles[metric], fontsize=TITLESIZE)
                ax.set_ylabel(metric, fontsize=LABELSIZE)
                ax.tick_params(axis='both', which='major', labelsize=TICKSIZE)
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, f"{metric} not in data", ha='center')

    # Set common X label
    for ax in axes[-2:]:
        ax.set_xlabel("K", fontsize=LABELSIZE)

    fig.suptitle(f"Order Parameters vs K for J={j_values}", fontsize=TITLESIZE + 2)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Order Parameter Plot saved to {out_path}")

    # plt.show()

if __name__ == "__main__":
    # plot_transient_time_summary(csv_path="results_data/heatmap_transient_times_mser.csv", out_path="plots/heatmap_transient_times_mser.png")
    # plot_phase_heatmap(csv_path='N200_30_seed.csv', out_path="plots/N200_30_seed_heatmap.png")
    plot_transient_times(csv_path="results_data/static_async_transient_times_mser.csv", out_path="plots/static_async_transient_times_mser_300.png")

    # Test averaged heatmap for S
    # plot_param_heatmap(csv_path='results_data/N200_30_seed.csv', param='V', out_path="plots/heatmap_V_avg.png")
    # plot_param_heatmap(csv_path='results_data/N200_30_seed.csv', param='S', out_path="plots/heatmap_S_avg.png")
    # plot_param_heatmap(csv_path='results_data/N200_30_seed.csv', param='R', out_path="plots/heatmap_R_avg.png")
    # plot_param_heatmap(csv_path='results_data/N200_30_seed.csv', param='omega', out_path="plots/heatmap_omega_avg.png")
    # plot_order_parameters_vs_K(csv_path='results_data/N200_30_seed.csv', j_values=[ 0.2, 0.4, 0.6, 0.8, 1.0], out_path="plots/order_params_vs_K_J0_1_5.png")
