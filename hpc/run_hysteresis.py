"""
Parallel hysteresis sweep runner using Python multiprocessing.

Runs 30 forward + 30 backward K-sweeps in parallel across available CPUs.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count


def run_single_sweep(args):
    """Run a single K-sweep with the given parameters."""
    direction, seed, base_dir = args
    
    if direction == "forward":
        kmin, kmax, dk = -1.0, 0.2, 0.001
    else:
        kmin, kmax, dk = 0.2, -1.0, -0.001
    
    output_path = Path(base_dir) / direction / f"ksweep_seed{seed}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "hpc/run_ksweep.py",
        "--N", "50",
        "--J", "0.5",
        "--Kmin", str(kmin),
        "--Kmax", str(kmax),
        "--dK", str(dk),
        "--steps_per_K", "1000",
        "--log_interval", "10",
        "--seed", str(seed),
        "--output", str(output_path),
    ]
    
    print(f"Starting {direction} sweep, seed={seed}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in {direction} seed={seed}: {result.stderr}", file=sys.stderr)
        return False
    
    print(f"Completed {direction} sweep, seed={seed}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run parallel hysteresis sweeps")
    parser.add_argument("--n_seeds", type=int, default=30, help="Number of seeds per direction")
    parser.add_argument("--n_workers", type=int, default=None, 
                        help="Number of parallel workers (default: all CPUs)")
    parser.add_argument("--output_dir", type=str, default="results/hysteresis",
                        help="Base output directory")
    
    args = parser.parse_args()
    
    n_workers = args.n_workers or cpu_count()
    print(f"Running hysteresis sweeps with {n_workers} parallel workers")
    print(f"  {args.n_seeds} forward + {args.n_seeds} backward = {2 * args.n_seeds} total runs")
    
    # Build job list
    jobs = []
    for seed in range(args.n_seeds):
        jobs.append(("forward", seed, args.output_dir))
    for seed in range(args.n_seeds):
        jobs.append(("backward", seed, args.output_dir))
    
    # Run in parallel
    with Pool(n_workers) as pool:
        results = pool.map(run_single_sweep, jobs)
    
    n_success = sum(results)
    n_total = len(results)
    print(f"\nCompleted {n_success}/{n_total} runs successfully")
    
    if n_success < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
