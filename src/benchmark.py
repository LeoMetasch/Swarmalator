#!/usr/bin/env python3
"""
Benchmark script to compare Naive (NumPy) vs Numba implementations of Swarmalator step().

Usage:
    python benchmark.py                    # Run with default N values
    python benchmark.py --N 50 100 200 500 # Custom N values
    python benchmark.py --steps 200        # More steps for accuracy
"""
import time
import argparse
import numpy as np
from swarm import Swarm


def benchmark_single(N: int, steps: int, use_numba: bool, warmup_steps: int = 10) -> float:
    """
    Run a single benchmark for given N and method.

    Args:
        N: Number of swarmalators
        steps: Number of steps to time
        use_numba: Whether to use Numba implementation
        warmup_steps: Steps to run before timing (for JIT compilation)

    Returns:
        Time in seconds
    """
    np.random.seed(42)  # Reproducibility
    swarm = Swarm(
        N=N, J=0.5, K=-0.3, dt=0.1, steps=steps,
        chirality=False, phase_coupling=False, predator=False,
        use_numba=use_numba
    )

    # Warmup (especially important for Numba JIT compilation)
    for _ in range(warmup_steps):
        swarm.step()

    # Timed run
    start = time.perf_counter()
    for _ in range(steps):
        swarm.step()
    elapsed = time.perf_counter() - start

    return elapsed


def run_benchmark(N_values: list[int], steps: int = 100, repeats: int = 3):
    """
    Run benchmarks comparing naive vs Numba for different N values.

    Args:
        N_values: List of N values to test
        steps: Number of simulation steps per benchmark
        repeats: Number of times to repeat each benchmark
    """
    print("=" * 70)
    print("Swarmalator Step Benchmark: Naive (NumPy) vs Numba JIT")
    print("=" * 70)
    print(f"Steps per benchmark: {steps}")
    print(f"Repeats: {repeats}")
    print("-" * 70)
    print(f"{'N':>8} | {'Naive (s)':>12} | {'Numba (s)':>12} | {'Speedup':>10} | {'Steps/s (Numba)':>15}")
    print("-" * 70)

    results = []

    for N in N_values:
        # Benchmark naive (only first repeat to save time for large N)
        naive_times = []
        for r in range(repeats):
            t = benchmark_single(N, steps, use_numba=False)
            naive_times.append(t)
        naive_mean = np.mean(naive_times)

        # Benchmark numba
        numba_times = []
        for r in range(repeats):
            t = benchmark_single(N, steps, use_numba=True)
            numba_times.append(t)
        numba_mean = np.mean(numba_times)

        speedup = naive_mean / numba_mean if numba_mean > 0 else float('inf')
        steps_per_sec = steps / numba_mean if numba_mean > 0 else float('inf')

        print(f"{N:>8} | {naive_mean:>12.4f} | {numba_mean:>12.4f} | {speedup:>9.1f}x | {steps_per_sec:>15.1f}")

        results.append({
            'N': N,
            'naive_time': naive_mean,
            'numba_time': numba_mean,
            'speedup': speedup,
            'steps_per_sec': steps_per_sec
        })

    print("-" * 70)

    # Summary statistics
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"\nAverage speedup: {avg_speedup:.1f}x")
    print("\nNote: First Numba call includes JIT compilation (~1s overhead).")
    print("      Subsequent calls use cached compiled code.")

    return results


def verify_correctness(N: int = 100, steps: int = 50, tol: float = 1e-10):
    """
    Verify that Numba and naive implementations produce identical results.

    Args:
        N: Number of swarmalators
        steps: Number of steps to run
        tol: Tolerance for floating-point comparison

    Returns:
        True if implementations match, False otherwise
    """
    print("\n" + "=" * 70)
    print("Correctness Verification: Comparing Naive vs Numba results")
    print("=" * 70)

    # Run with same seed for both
    np.random.seed(123)
    swarm_naive = Swarm(N=N, J=0.5, K=-0.3, dt=0.1, steps=steps, use_numba=False)

    np.random.seed(123)
    swarm_numba = Swarm(N=N, J=0.5, K=-0.3, dt=0.1, steps=steps, use_numba=True)

    # Run same number of steps
    for _ in range(steps):
        swarm_naive.step()
        swarm_numba.step()

    # Compare results
    x_match = np.allclose(swarm_naive.x_pos, swarm_numba.x_pos, rtol=tol, atol=tol)
    y_match = np.allclose(swarm_naive.y_pos, swarm_numba.y_pos, rtol=tol, atol=tol)
    phase_match = np.allclose(swarm_naive.phases, swarm_numba.phases, rtol=tol, atol=tol)

    # Compute order parameters
    S_naive = swarm_naive._correlation_order_parameter()
    S_numba = swarm_numba._correlation_order_parameter()
    R_naive = swarm_naive._synchrony_order_parameter()
    R_numba = swarm_numba._synchrony_order_parameter()

    S_match = np.isclose(S_naive, S_numba, rtol=tol, atol=tol)
    R_match = np.isclose(R_naive, R_numba, rtol=tol, atol=tol)

    print(f"N={N}, steps={steps}")
    print(f"  x positions match: {x_match}")
    print(f"  y positions match: {y_match}")
    print(f"  phases match: {phase_match}")
    print(f"  S order parameter: naive={S_naive:.10f}, numba={S_numba:.10f}, match={S_match}")
    print(f"  R order parameter: naive={R_naive:.10f}, numba={R_numba:.10f}, match={R_match}")

    all_match = x_match and y_match and phase_match and S_match and R_match
    if all_match:
        print("\n✓ SUCCESS: Naive and Numba implementations produce identical results!")
    else:
        print("\n✗ FAILURE: Results differ between implementations!")
        # Show max differences
        print(f"  Max x diff: {np.max(np.abs(swarm_naive.x_pos - swarm_numba.x_pos))}")
        print(f"  Max y diff: {np.max(np.abs(swarm_naive.y_pos - swarm_numba.y_pos))}")
        print(f"  Max phase diff: {np.max(np.abs(swarm_naive.phases - swarm_numba.phases))}")

    return all_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Swarmalator implementations")
    parser.add_argument("--N", type=int, nargs="+", default=[50, 100, 200, 500],
                        help="N values to benchmark (default: 50 100 200 500)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of steps per benchmark (default: 100)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repeats for averaging (default: 3)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip correctness verification")

    args = parser.parse_args()

    # First verify correctness
    if not args.skip_verify:
        verify_correctness(N=100, steps=50)

    # Then run performance benchmark
    run_benchmark(N_values=args.N, steps=args.steps, repeats=args.repeats)
