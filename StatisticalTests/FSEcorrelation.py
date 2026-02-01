import pandas as pd
from scipy.stats import spearmanr

sync = pd.read_csv("static_sync_transient_times_mser.csv")
async_ = pd.read_csv("static_async_transient_times_mser.csv")

sync_mean = sync.groupby("N")["transient_time"].mean().reset_index()
async_mean = async_.groupby("N")["transient_time"].mean().reset_index()

rho_sync, p_sync = spearmanr(
    sync_mean["N"],
    sync_mean["transient_time"]
)

rho_async, p_async = spearmanr(
    async_mean["N"],
    async_mean["transient_time"]
)

print("STATIC SYNC:")
print("Spearman rho =", rho_sync)
print("p-value =", p_sync)

print("\nSTATIC ASYNC:")
print("Spearman rho =", rho_async)
print("p-value =", p_async)

