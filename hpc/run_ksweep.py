"""
Continuous K-sweep experiment for observing phase transitions in Swarmalators.

K steps discretely from K_min to K_max, with configurable time spent at each K.
Logs order parameters at every step to capture transition dynamics.
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from swarm import Swarm


def run_ksweep(
    N: int,
    J: float,
    K_min: float,
    K_max: float,
    dK: float,
    steps_per_K: int,
    dt: float,
    log_interval: int,
    seed: int,
    output_path: Path,
    independent: bool = False,
):
    """
    Run K-sweep with discrete K steps, logging all dynamics.
    
    Args:
        N: Number of swarmalators
        J: Fixed spatial coupling
        K_min, K_max: Range of phase coupling parameter
        dK: K increment between discrete values
        steps_per_K: Number of simulation steps at each K value
        dt: Integration timestep
        log_interval: Log order parameters every N steps
        seed: Random seed for reproducibility
        output_path: Output CSV path
    """
    np.random.seed(seed)
    
    # Generate K values (handles both forward and backward sweeps)
    if dK > 0:
        K_values = np.arange(K_min, K_max + dK/2, dK)
    else:
        # Backward sweep: dK is negative, K_min > K_max
        K_values = np.arange(K_min, K_max + dK/2, dK)
    n_K = len(K_values)
    total_steps = n_K * steps_per_K
    
    # Initialize swarm (only once if not independent)
    if not independent:
        swarm = Swarm(
            N=N, J=J, K=K_values[0], dt=dt, steps=total_steps,
            chirality=False, phase_coupling=False, predator=False,
            use_numba=True
        )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "step", "K", "t_at_K", "S", "R", "V", "omega", "state"
    ]
    
    print(f"Starting K-sweep ({'INDEPENDENT' if independent else 'ADIABATIC'})", file=sys.stderr)
    print(f"  K range: [{K_min}, {K_max}]", file=sys.stderr)
    print(f"  dK = {dK} → {n_K} K values", file=sys.stderr)
    print(f"  steps_per_K = {steps_per_K}", file=sys.stderr)
    print(f"  Total steps: {total_steps}", file=sys.stderr)
    print(f"  Logging every {log_interval} steps → {total_steps // log_interval} data points", file=sys.stderr)
    
    # Store previous state for velocity calculation
    # (Initialize with dummy values)
    prev_x = np.zeros(N)
    prev_y = np.zeros(N)
    prev_theta = np.zeros(N)
    
    global_step = 0
    
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        
        for i_K, K in enumerate(K_values):
            # If independent, re-initialize swarm at each K
            if independent:
                # Use a specific seed for each K to be deterministic but different
                steps_for_this_K = steps_per_K
                swarm = Swarm(
                    N=N, J=J, K=K, dt=dt, steps=steps_for_this_K,
                    chirality=False, phase_coupling=False, predator=False,
                    use_numba=True
                )
                # Re-seed if needed, or let it be random based on global seed + offset
                # np.random.seed(seed + i_K) # Optional: control seed per K
            else:
                # Set new K value (no re-initialization)
                swarm.K = K
            
            for t_at_K in range(steps_per_K):
                # Evolve one step
                swarm.step()
                
                # Log at intervals
                if global_step % log_interval == 0:
                    # Compute order parameters
                    S = swarm._correlation_order_parameter()
                    R = swarm._synchrony_order_parameter()
                    
                    # Need valid previous state for velocity
                    if t_at_K > 0:
                        V, omega = swarm._calculate_velocity_order_parameter(prev_x, prev_y, prev_theta)
                    else:
                        V, omega = 0.0, 0.0
                    
                    # Get state classification
                    state, _, _, _, _, _, _, _, _ = swarm.stability_analysis()
                    
                    writer.writerow({
                        "step": global_step,
                        "K": K,
                        "t_at_K": t_at_K,
                        "S": S,
                        "R": R,
                        "V": V,
                        "omega": omega,
                        "state": state,
                    })
                
                # Update previous state
                prev_x = swarm.x_pos.copy()
                prev_y = swarm.y_pos.copy()
                prev_theta = swarm.phases.copy()
                
                global_step += 1
            
            # Progress update after each K value
            if (i_K + 1) % max(1, n_K // 10) == 0 or i_K == 0:
                S = swarm._correlation_order_parameter()
                R = swarm._synchrony_order_parameter()
                print(f"  [{i_K+1}/{n_K}] K={K:.4f}, S={S:.3f}, R={R:.3f}", file=sys.stderr)
    
    print(f"\nResults saved to: {output_path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="K-sweep for observing phase transitions")
    p.add_argument("--N", type=int, default=100, help="Number of swarmalators")
    p.add_argument("--J", type=float, default=0.5, help="Fixed spatial coupling")
    p.add_argument("--Kmin", type=float, default=-1.0, help="Starting K value")
    p.add_argument("--Kmax", type=float, default=0.2, help="Ending K value")
    p.add_argument("--dK", type=float, default=0.01, help="K increment between values")
    p.add_argument("--steps_per_K", type=int, default=1000, help="Simulation steps at each K")
    p.add_argument("--dt", type=float, default=0.05, help="Integration timestep")
    p.add_argument("--log_interval", type=int, default=1, help="Log every N steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", type=str, default="results/ksweep.csv", help="Output CSV path")
    p.add_argument("--independent", action="store_true", help="Re-initialize swarm for each K (removes hysteresis)")
    
    args = p.parse_args()
    
    run_ksweep(
        N=args.N,
        J=args.J,
        K_min=args.Kmin,
        K_max=args.Kmax,
        dK=args.dK,
        steps_per_K=args.steps_per_K,
        dt=args.dt,
        log_interval=args.log_interval,
        seed=args.seed,
        output_path=Path(args.output),
        independent=args.independent
    )


if __name__ == "__main__":
    main()
