"""
Continuous K-sweep experiment for observing phase transitions in Swarmalators.

K ramps slowly from K_min to K_max while continuously logging order parameters.
No equilibration/measurement split — we log during the transition itself.
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.swarm import Swarm


def run_ksweep(
    N: int,
    J: float,
    K_min: float,
    K_max: float,
    total_steps: int,
    dt: float,
    log_interval: int,
    seed: int,
    output_path: Path,
):
    """
    Run continuous K-sweep with K ramping linearly over time.
    
    K(t) = K_min + (K_max - K_min) * (t / total_steps)
    
    Args:
        N: Number of swarmalators
        J: Fixed spatial coupling (typically 0.5)
        K_min, K_max: Range of phase coupling parameter
        total_steps: Total simulation steps (K ramps over this duration)
        dt: Integration timestep
        log_interval: Log order parameters every N steps
        seed: Random seed for reproducibility
        output_path: Output CSV path
    """
    np.random.seed(seed)
    
    # Initialize swarm at starting K
    swarm = Swarm(
        N=N, J=J, K=K_min, dt=dt, steps=total_steps,
        chirality=False, phase_coupling=False, predator=False
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "step", "K", "S", "R", "V", "omega", "state"
    ]
    
    # K ramp rate
    dK_per_step = (K_max - K_min) / total_steps
    
    print(f"Starting continuous K-sweep", file=sys.stderr)
    print(f"  K ramps from {K_min} to {K_max} over {total_steps} steps", file=sys.stderr)
    print(f"  dK/dt = {dK_per_step:.6f} per step", file=sys.stderr)
    print(f"  Logging every {log_interval} steps → {total_steps // log_interval} data points", file=sys.stderr)
    
    # Store previous state for velocity calculation
    prev_x = swarm.x_pos.copy()
    prev_y = swarm.y_pos.copy()
    prev_theta = swarm.phases.copy()
    
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        
        for step in range(total_steps):
            # Update K linearly with time
            K = K_min + dK_per_step * step
            swarm.K = K
            
            # Evolve one step
            swarm.step()
            
            # Log at intervals
            if step % log_interval == 0:
                # Compute order parameters
                S = swarm._correlation_order_parameter()
                R = swarm._synchrony_order_parameter()
                V, omega = swarm._calculate_velocity_order_parameter(prev_x, prev_y, prev_theta)
                
                # Get state classification
                state, _, _, _, _, _, _, _, _ = swarm.stability_analysis()
                
                writer.writerow({
                    "step": step,
                    "K": K,
                    "S": S,
                    "R": R,
                    "V": V,
                    "omega": omega,
                    "state": state,
                })
                
                # Progress update
                if step % (total_steps // 10) == 0:
                    print(f"  [{step}/{total_steps}] K={K:.4f}, S={S:.3f}, R={R:.3f}", file=sys.stderr)
            
            # Update previous state
            prev_x = swarm.x_pos.copy()
            prev_y = swarm.y_pos.copy()
            prev_theta = swarm.phases.copy()
    
    print(f"\nResults saved to: {output_path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Continuous K-sweep for observing phase transitions")
    p.add_argument("--N", type=int, default=100, help="Number of swarmalators")
    p.add_argument("--J", type=float, default=0.5, help="Fixed spatial coupling")
    p.add_argument("--Kmin", type=float, default=-1.0, help="Starting K value")
    p.add_argument("--Kmax", type=float, default=0.2, help="Ending K value")
    p.add_argument("--steps", type=int, default=120000, help="Total simulation steps")
    p.add_argument("--dt", type=float, default=0.1, help="Integration timestep")
    p.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", type=str, default="results/ksweep.csv", help="Output CSV path")
    
    args = p.parse_args()
    
    run_ksweep(
        N=args.N,
        J=args.J,
        K_min=args.Kmin,
        K_max=args.Kmax,
        total_steps=args.steps,
        dt=args.dt,
        log_interval=args.log_interval,
        seed=args.seed,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
