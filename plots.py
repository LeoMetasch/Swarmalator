import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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
    
    plt.title("Phase Diagram: State over J vs K")
    plt.xlabel("K")
    plt.ylabel("J")
    
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
        xticklabels=5,
        yticklabels=5
    )
    
    ax.invert_yaxis()
    plt.title("Phase Diagram: Order Parameter S over J vs K")
    plt.xlabel("K")
    plt.ylabel("J")
    
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"S Heatmap saved to {out_path}")
    
    # plt.show()

if __name__ == "__main__":
    plot_phase_heatmap()
    plot_S_heatmap()