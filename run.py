import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from src.swarm import Swarm
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

SCRIPT_DIR = Path(__file__).parent


def save_final_state(swarm, N, J, K, seed, state, log_path):
    os.makedirs(SCRIPT_DIR / "final_states", exist_ok=True)
    plt.scatter(swarm.x_pos, swarm.y_pos, c=swarm.phases, cmap='hsv')
    plt.colorbar(label='Phase (theta)')
    plt.title(f'Swarmalator Final State (N={N}, J={J}, K={K}, seed={seed})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"final_states/swarmalator_N{N}_J{J}_K{K}_seed{seed}.png")
    plt.close()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["J", "K", "N", "S", "V", "omega", "R", "state", "seed"]
    is_new_file = not log_path.exists()

    

    with log_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()

        # log initial state (step 0) with zero velocities
        state0, S0, V0, omega0, R0, best_k0, best_sep0, best_comp0, best_aniso0 = swarm.stability_analysis()
        writer.writerow({
            "J": swarm.J,
            "K": swarm.K,
            "N": swarm.N,
            "S": S0,
            "V": V0,
            "omega": omega0,
            "R": R0,
            "state": state0,
            "seed": seed,
        })



def run_once(N, J, K, seed, dt, steps, burnin, sample_every):
    np.random.seed(seed)

    # x = np.random.uniform(-1.0, 1.0, N)
    # y = np.random.uniform(-1.0, 1.0, N)
    # theta = np.random.uniform(-np.pi, np.pi, N)

    swarm = Swarm(N=N, J=J, K=K, dt=dt, steps=steps, chirality=False, phase_coupling=False, predator=False)

    R_vals = []
    S_vals = []

    # for t in range(steps):
    #     swarm.step()

    #     if t >= burnin:
    #         R_vals.append(np.abs(np.mean(np.exp(1j * swarm.phases))))
    #         S_vals.append(swarm._correlation_order_parameter())

    # R_mean = float(np.mean(R_vals))
    # S_mean = float(np.mean(S_vals))

    swarm.run_with_logging(steps=steps, log_path=SCRIPT_DIR / f"logs/temp_N{N}_J{J}_K{K}_seed{seed}.csv", log_interval=sample_every)

    state, S_parameter, V_parameter, omega_parameter, R_parameter, best_k, best_sep, best_comp, best_aniso = swarm.stability_analysis()

    save_final_state(swarm, N, J, K, seed, state, log_path="test.csv")
    print(f"{N},{J},{K},{seed},{R_parameter},{S_parameter},{state}")
    # return state, float(S_parameter), float(V_parameter), float(omega_parameter), float(R_parameter), best_k, best_sep, best_comp, best_aniso


    return f"{N},{J},{K},{seed},{R_parameter},{S_parameter},{state}"


def _run_once_wrapper(params):
    """Wrapper for run_once to unpack tuple arguments for multiprocessing."""
    N, J, K, seed, dt, steps, burnin, sample_every = params
    return run_once(N, J, K, seed, dt, steps, burnin, sample_every)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int)
    p.add_argument("--Nmin", type=int, default=50)
    p.add_argument("--Nmax", type=int, default=200)
    p.add_argument("--Nsteps", type=int, default=4)
    p.add_argument("--J", type=float, default=0.0)
    p.add_argument("--K", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--burnin", type=int, default=2000)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--Jmin", type=float, default=0.5)
    p.add_argument("--Jmax", type=float, default=1.0)
    p.add_argument("--Jsteps", type=int, default=51)
    p.add_argument("--Kmin", type=float, default=-.8)
    p.add_argument("--Kmax", type=float, default=.2)
    p.add_argument("--Ksteps", type=int, default=51)
    p.add_argument("--workers", type=int, default=None, 
                   help="Number of parallel workers (default: number of CPU cores)")

    args = p.parse_args()

    n_workers = args.workers if args.workers else cpu_count()

    seeds = [0, 1, 2]
    seeds = range(30)

    if args.sweep:
        Js = np.linspace(args.Jmin, args.Jmax, args.Jsteps)
        Ks = np.linspace(args.Kmin, args.Kmax, args.Ksteps)
        Ns = np.linspace(args.Nmin, args.Nmax, args.Nsteps, dtype=int)


        # Build list of all parameter combinations
        param_list = [
            (N, float(J), float(K), seed, args.dt, args.steps, args.burnin, args.sample_every)
            for J in Js
            for K in Ks
            for N in Ns
            for seed in seeds
        ]

        print(f"Running {len(param_list)} simulations with {n_workers} workers...", file=__import__('sys').stderr)
        print("N,J,K,seed,R,S,state")

        # Run in parallel and collect results (order is preserved by Pool.map)
        with Pool(processes=n_workers) as pool:
            results = pool.map(_run_once_wrapper, param_list)

        for result in results:
            print(result)

    else:
        print("N,J,K,seed,R,S,state")
        print(run_once(args.N, args.J, args.K, args.seed, args.dt, args.steps, args.burnin, args.sample_every))


if __name__ == "__main__":
    main()
