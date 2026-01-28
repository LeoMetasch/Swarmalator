import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import argparse

base_dir = Path(__file__).resolve().parent
log_path = base_dir / "logsFIRSTTEST"

def load_logs():
    files = sorted(log_path.glob("*.csv"))
    dfs = []
    for file in files:
        df = pd.read_csv(file).sort_values("step")
        df["filename"] = file.name
        dfs.append(df)
    return dfs

def _select_rows_by_JK(df, J, K, tol=1e-2):
    return df[(np.abs(df["J"] - J) < tol) & (np.abs(df["K"] - K) < tol)]

def _approx_existing_JK(values, target):
    values = np.array(sorted(values))
    return values[np.argmin(np.abs(values - target))]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--J", type=float, required=True)
    p.add_argument("--K", type=float, required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dfs = load_logs()

    all_J = [df.iloc[0]["J"] for df in dfs]
    all_K = [df.iloc[0]["K"] for df in dfs]

    J_target = _approx_existing_JK(all_J, args.J)
    K_target = _approx_existing_JK(all_K, args.K)

    print(f"Requested J={args.J}, K={args.K} -> using nearest J={J_target}, K={K_target}")

    selected_dfs = []
    for df in dfs:
        df_sel = _select_rows_by_JK(df, J_target, K_target)
        if not df_sel.empty:
            selected_dfs.append(df_sel)
    
    print(f"Found {len(selected_dfs)} matching log files for J={J_target}, K={K_target}")

    plt.figure()
    for df in selected_dfs:
        N = df.iloc[0]["N"]
        plt.plot(df["step"], df["R"], label=f"N={N}")
        plt.plot(df["step"], df["S"], linestyle="--", label=f"S, N={N}")
    
    plt.xlabel("step")
    plt.ylabel("R")
    plt.title(f"Order parameter R over time for J={J_target}, K={K_target}")
    plt.legend()
    plt.show()
    


    # for df in dfs:
    #     last_row = df.iloc[-1]
    #     print(
    #         'file', last_row["filename"],
    #         'J', last_row["J"],
    #         'K', last_row["K"],
    #         'N', last_row["N"],
    #         'S', last_row["S"],
    #         'V', last_row["V"],
    #         'omega', last_row["omega"],
    #         'R', last_row["R"],
    #     )

    # if len(dfs) > 0:
    #     df = dfs[0]
    #     plt.plot(df["step"], df["R"], label="R")
    #     plt.plot(df["step"], df["S"], label="S")
    #     plt.xlabel("step")
    #     plt.legend()
    #     plt.show()





