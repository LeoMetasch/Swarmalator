"""
Predator Hunting Experiments for Swarmalators

Two experiments:
1. Impact of hunting strength on order parameters (starting from static phase wave)
2. Relaxation speed analysis (time to return to stable state after predator removal)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from swarm import Swarm
from typing import Optional, Dict, List, Any, Tuple


# =============================================================================
# Experiment 1: Impact of Hunting Strength on Order Parameters
# =============================================================================

def experiment_hunting_strength_impact(
    N: int = 100,
    J: float = 1,
    K: float = 0,
    dt: float = 0.1,
    burnin_steps: int = 3000,
    predator_steps: int = 5000,
    hunting_strengths: Optional[List[float]] = None,
    seed: int = 42,
    save_dir: str = "results/hunting_strength",
) -> Dict[float, Dict[str, List[Any]]]:
    """
    Analyze the impact of hunting strength on order parameters.
    
    1. Initialize swarm and run to stable static phase wave (burnin)
    2. Introduce predator with varying hunting strengths
    3. Track order parameters (S, R, V, omega) over time
    
    Args:
        N: Number of swarmalators
        J: Phase attraction (positive -> phase wave)
        K: Spatial coupling (negative -> static phase wave)
        dt: Time step
        burnin_steps: Steps to reach stable phase wave before predator
        predator_steps: Steps to run with predator active
        hunting_strengths: List of hunting strength values to test
        seed: Random seed
        save_dir: Directory to save results
    """
    if hunting_strengths is None:
        hunting_strengths = [0.0]
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    results = {h: {"steps": [], "S": [], "V": [], "omega": [], "R": [], "state": []} 
               for h in hunting_strengths}
    
    for h_strength in hunting_strengths:
        print(f"Running hunting_strength = {h_strength}...")
        np.random.seed(seed)
        
        # Phase 1: Burnin to stable phase wave (no predator)
        swarm = Swarm(
            N=N, J=J, K=K, dt=dt, steps=burnin_steps,
            chirality=False, phase_coupling=False, predator=False
        )
        
        # Run burnin
        for _ in range(burnin_steps):
            swarm.step()
        
        # Verify we have a static phase wave
        state, S, V, omega, R, *_ = swarm.stability_analysis()
        print(f"  After burnin: state={state}, S={S:.3f}, V={V:.4f}")
        
        # Phase 2: Introduce predator
        swarm.predator = True
        swarm.hunting_strength = h_strength
        swarm.pred_x = np.random.uniform(-1, 1)
        swarm.pred_y = np.random.uniform(-1, 1)
        
        # Store initial positions for comparison
        x_prev = swarm.x_pos.copy()
        y_prev = swarm.y_pos.copy()
        theta_prev = swarm.phases.copy()
        
        # Track order parameters during predator phase
        for step in range(predator_steps):
            swarm.step()
            
            if step % 50 == 0:  # Sample every 50 steps
                S = swarm._correlation_order_parameter()
                V, omega = swarm._calculate_velocity_order_parameter(x_prev, y_prev, theta_prev)
                R = swarm._synchrony_order_parameter()
                state, *_ = swarm.stability_analysis()
                
                results[h_strength]["steps"].append(step)
                results[h_strength]["S"].append(S)
                results[h_strength]["V"].append(V)
                results[h_strength]["omega"].append(omega)
                results[h_strength]["R"].append(R)
                results[h_strength]["state"].append(state)
                
                x_prev = swarm.x_pos.copy()
                y_prev = swarm.y_pos.copy()
                theta_prev = swarm.phases.copy()
    
    # Plot S 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for h_strength in hunting_strengths:
        steps = results[h_strength]["steps"]
        ax.plot(steps, results[h_strength]["S"], label=f"h={h_strength}", alpha=0.8, linewidth=2)
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("S (Correlation Order Parameter)", fontsize=12)
    ax.set_title(f"Impact of Hunting Strength on Phase-Position Correlation\n(N={N}, J={J}, K={K})", fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path / "hunting_strength_impact.png", dpi=150)
    plt.close()
    
    # Save data to CSV
    all_data = []
    for h_strength in hunting_strengths:
        for i, step in enumerate(results[h_strength]["steps"]):
            all_data.append({
                "hunting_strength": h_strength,
                "step": step,
                "S": results[h_strength]["S"][i],
                "V": results[h_strength]["V"][i],
                "omega": results[h_strength]["omega"][i],
                "R": results[h_strength]["R"][i],
                "state": results[h_strength]["state"][i],
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(save_path / "hunting_strength_data.csv", index=False)
    
    print(f"Results saved to {save_path}")
    return results

def experiment_relaxation_speed(
    N: int = 500,
    J: float = 1,
    K: float = 0,
    dt: float = 0.1,
    burnin_steps: int = 1000,
    predator_steps: int = 1000,
    relaxation_steps: int = 1500,
    hunting_strengths: Optional[List[float]] = None,
    stability_threshold: float = 0.002,
    stability_window: int = 100,
    seed: int = 42,
    save_dir: str = "results/relaxation",
) -> Tuple[Dict[float, int], Dict[float, Dict[str, List[float]]]]:
    """Measure relaxation time after predator removal.
    
    Tracks V over the ENTIRE simulation (burnin + predator + relaxation).
    
    Args:
        N: Number of swarmalators.
        J: Phase attraction strength.
        K: Spatial coupling strength.
        dt: Time step.
        burnin_steps: Steps to reach stable phase wave before predator.
        predator_steps: Steps to run with predator active.
        relaxation_steps: Steps to run after removing predator.
        hunting_strengths: List of hunting strength values to test.
        stability_threshold: V/omega threshold for stability detection.
        stability_window: Number of consecutive stable steps required.
        seed: Random seed.
        save_dir: Directory to save results.
    
    Returns:
        Tuple of (relaxation_times dict, full_traces dict).
    """
    if hunting_strengths is None:
        hunting_strengths = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    relaxation_times = {}
    full_traces = {}  # Store traces for entire run
    
    total_steps = burnin_steps + predator_steps + relaxation_steps
    
    for h_strength in hunting_strengths:
        print(f"Running hunting_strength = {h_strength}...")
        np.random.seed(seed)
        
        # Initialize swarm
        swarm = Swarm(
            N=N, J=J, K=K, dt=dt, steps=total_steps,
            chirality=False, phase_coupling=False, predator=False
        )
        
        V_trace = []
        omega_trace = []
        S_trace = []
        
        x_prev = swarm.x_pos.copy()
        y_prev = swarm.y_pos.copy()
        theta_prev = swarm.phases.copy()
        
        for step in range(burnin_steps):
            swarm.step()
            
            S = swarm._correlation_order_parameter()
            V, omega = swarm._calculate_velocity_order_parameter(x_prev, y_prev, theta_prev)
            V_trace.append(V)
            omega_trace.append(omega)
            S_trace.append(S)
            
            x_prev = swarm.x_pos.copy()
            y_prev = swarm.y_pos.copy()
            theta_prev = swarm.phases.copy()
        
        print(f"  After burnin (step {burnin_steps}): S={S_trace[-1]:.3f}, V={V_trace[-1]:.6f}")
        
        swarm.predator = True
        swarm.hunting_strength = h_strength
        swarm.pred_x = np.random.uniform(-1, 1)
        swarm.pred_y = np.random.uniform(-1, 1)
        
        for step in range(predator_steps):
            swarm.step()
            
            S = swarm._correlation_order_parameter()
            V, omega = swarm._calculate_velocity_order_parameter(x_prev, y_prev, theta_prev)
            V_trace.append(V)
            omega_trace.append(omega)
            S_trace.append(S)
            
            x_prev = swarm.x_pos.copy()
            y_prev = swarm.y_pos.copy()
            theta_prev = swarm.phases.copy()
        
        print(f"  After predator (step {burnin_steps + predator_steps}): S={S_trace[-1]:.3f}, V={V_trace[-1]:.6f}")
        
        # ========== Phase 3: Relaxation ==========
        swarm.predator = False
        
        stable_count = 0
        relaxation_time = None
        
        for step in range(relaxation_steps):
            swarm.step()
            
            S = swarm._correlation_order_parameter()
            V, omega = swarm._calculate_velocity_order_parameter(x_prev, y_prev, theta_prev)
            V_trace.append(V)
            omega_trace.append(omega)
            S_trace.append(S)
            
            x_prev = swarm.x_pos.copy()
            y_prev = swarm.y_pos.copy()
            theta_prev = swarm.phases.copy()
            
            # Check stability
            if relaxation_time is None:
                if V < stability_threshold and omega < stability_threshold:
                    stable_count += 1
                    if stable_count >= stability_window:
                        relaxation_time = step
                        print(f"  Reached stability at relaxation step {step} (global step {burnin_steps + predator_steps + step})")
                else:
                    stable_count = 0
        
        if relaxation_time is None:
            relaxation_time = relaxation_steps
            print(f"  WARNING: Did not stabilize within {relaxation_steps} steps")
        
        relaxation_times[h_strength] = relaxation_time
        full_traces[h_strength] = {
            "V": V_trace,
            "omega": omega_trace,
            "S": S_trace,
        }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for h_strength in hunting_strengths:
        steps = np.arange(len(full_traces[h_strength]["V"]))
        ax.plot(steps, full_traces[h_strength]["V"], label=f"h={h_strength}", alpha=0.8)
    
    # Add vertical lines for phase transitions
    ax.axvline(x=burnin_steps, color='green', linestyle='--', linewidth=2, label='Predator Spawned')
    ax.axvline(x=burnin_steps + predator_steps, color='red', linestyle='--', linewidth=2, label='Predator Removed')
    ax.axhline(y=stability_threshold, color='gray', linestyle=':', label=f'Threshold ({stability_threshold})')
    
    ax.set_xlabel("Step (global)")
    ax.set_ylabel("V (Spatial Velocity)")
    ax.set_title(f"Velocity Over Entire Simulation\n(N={N}, J={J}, K={K}, burnin={burnin_steps}, predator={predator_steps})")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path / "full_velocity_trace.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    h_vals = list(relaxation_times.keys())
    t_vals = list(relaxation_times.values())
    
    ax.bar(range(len(h_vals)), t_vals, tick_label=[str(h) for h in h_vals], color='steelblue', edgecolor='black')
    ax.set_xlabel("Hunting Strength")
    ax.set_ylabel("Relaxation Time (steps)")
    ax.set_title("Relaxation Time vs Hunting Strength")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path / "relaxation_times.png", dpi=150)
    plt.close()
    
    # Save data
    summary_df = pd.DataFrame({
        "hunting_strength": h_vals,
        "relaxation_time": t_vals,
    })
    summary_df.to_csv(save_path / "relaxation_times.csv", index=False)
    
    print(f"Results saved to {save_path}")
    return relaxation_times, full_traces


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 1: Hunting Strength Impact on Order Parameters")
    print("=" * 60)
    experiment_hunting_strength_impact()
    
    print("\n" + "=" * 60)
    print("Experiment 2: Relaxation Speed Analysis")
    print("=" * 60)
    experiment_relaxation_speed()
    
    print("\nAll experiments complete!")

    
