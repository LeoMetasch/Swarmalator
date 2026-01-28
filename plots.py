import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
sns.set(style="whitegrid")

# SIZE PARAMEERS
TITLESIZE = 16
LABELSIZE = 20
TICKSIZE = 18

def plot_phase_heatmap(csv_path="test.csv", out_path="plots/heatmap_state.png"):
    """
    Generates a categorical heatmap of the system state over J and K.
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

def plot_S_heatmap(csv_path="test.csv", out_path="plots/heatmap_S.png"):
    """
    Generates a heatmap of the S parameter over J and K.
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

def plot_transient_time_summary(csv_path="./results_data/transient_times_summary.csv"):
    """
    Docstring for plot_transient_time_summary
    
    :param csv_path: Description
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
    
    if csv_path:
        out_path = csv_path.replace(".csv", "_heatmap.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Transient Time Heatmap saved to {out_path}")
    
    # plt.show()

def plot_transient_times(csv_path="results_data/transient_times_static_async.csv", out_path="transient_times.png"):
    """
    Plots the transient time vs N with statistical aggregation.
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
    
def plot_method_comparison(
    threshold_csv="results_data/transient_times_static_async.csv", 
    mser_csv="results_data/transient_times_mser.csv",
    out_path="transient_comparison.png"
):
    """
    Overlays plots from both methods to compare them.
    """
    plt.figure(figsize=(10, 6))
    
    # Load Threshold Data
    try:
        df_thresh = pd.read_csv(threshold_csv)
        if 'converged' in df_thresh.columns:
            df_thresh = df_thresh[df_thresh['converged'] == True]
        sns.lineplot(data=df_thresh, x="N", y="transient_time", marker="o", label="Threshold Method", errorbar=('ci', 95))
    except FileNotFoundError:
        print(f"Could not find {threshold_csv}")
        
    # Load MSER Data
    try:
        df_mser = pd.read_csv(mser_csv)
        sns.lineplot(data=df_mser, x="N", y="transient_time", marker="o", label="MSER Method", errorbar=('ci', 95), linestyle="--")
    except FileNotFoundError:
        print(f"Could not find {mser_csv}")
        
    plt.title("Comparison of Transient Time Estimation Methods")
    plt.xlabel("N")
    plt.ylabel("Transient Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    # plot_phase_heatmap(csv_path='sweep_18775448.csv', out_path="plots/heatmap_state_n_100.png")
    # plot_S_heatmap(csv_path='sweep_18775448.csv', out_path="plots/heatmap_S_n_100.png")
    # plot_transient_times(csv_path="results_data/transient_times_mser.csv", out_path="tansient_times_mser.png")
    plot_method_comparison()