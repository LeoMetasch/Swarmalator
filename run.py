import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from src.swarm import Swarm



def save_final_state(swarm, N, J, K, seed):
    os.makedirs("final_states", exist_ok=True)

    plt.scatter(swarm.x_pos, swarm.y_pos, c=swarm.phases, cmap='hsv')
    plt.colorbar(label='Phase (theta)')
    plt.title(f'Swarmalator Final State (N={N}, J={J}, K={K}, seed={seed})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"final_states/swarmalator_N{N}_J{J}_K{K}_seed{seed}.png")
    plt.close()



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

    swarm.run_with_logging(steps=steps, log_path=f"logs/temp_N{N}_J{J}_K{K}_seed{seed}.csv", log_interval=sample_every)

    state, S_parameter, V_parameter, omega_parameter, R_parameter, best_k, best_sep, best_comp, best_aniso = swarm.stability_analysis()

    save_final_state(swarm, N, J, K, seed)
    # return state, float(S_parameter), float(V_parameter), float(omega_parameter), float(R_parameter), best_k, best_sep, best_comp, best_aniso


    return f"{N},{J},{K},{seed},{R_parameter},{S_parameter},{state}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--J", type=float, default=0.0)
    p.add_argument("--K", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--burnin", type=int, default=2000)
    p.add_argument("--sample_every", type=int, default=10)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--Jmin", type=float, default=-1.0)
    p.add_argument("--Jmax", type=float, default=2.0)
    p.add_argument("--Jsteps", type=int, default=21)
    p.add_argument("--Kmin", type=float, default=-2.0)
    p.add_argument("--Kmax", type=float, default=4.0)
    p.add_argument("--Ksteps", type=int, default=21)

    args = p.parse_args()


    seeds = [0]

    if args.sweep:
        Js = np.linspace(args.Jmin, args.Jmax, args.Jsteps)
        Ks = np.linspace(args.Kmin, args.Kmax, args.Ksteps)

        print("N,J,K,seed,R,S, state")

        outpath = "sweep.csv"
        with open(outpath, "w") as f:
            f.write("N,J,K,seed,R,S,state,S_param,V_param,omega_param\n")

            for J in Js:
                for K in Ks:
                    for seed in seeds:
                        print(run_once(args.N, float(J), float(K), seed, args.dt, args.steps, args.burnin, args.sample_every))

    else:
        print("N,J,K,seed,R,S, state")
        print(run_once(args.N, args.J, args.K, args.seed, args.dt, args.steps, args.burnin, args.sample_every))


if __name__ == "__main__":
    main()
