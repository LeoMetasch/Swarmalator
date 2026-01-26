import argparse
import numpy as np
from src.swarmalator import Swarmalator


def run_once(N, J, K, seed, dt, steps, burnin, sample_every):
    np.random.seed(seed)

    x = np.random.uniform(-1.0, 1.0, N)
    y = np.random.uniform(-1.0, 1.0, N)
    theta = np.random.uniform(-np.pi, np.pi, N)

    swarm = Swarmalator(N=N, x=x, y=y, theta=theta, J=J, K=K, dt=dt)

    R_vals = []
    S_vals = []

    for t in range(steps):
        swarm.time_step()

        if t >= burnin:
            R_vals.append(np.abs(np.mean(np.exp(1j * swarm.theta))))
            S_vals.append(swarm.correlation_order_parameter())

            R_vals.append(R)
            S_vals.append(S)

    R_mean = float(np.mean(R_vals))
    S_mean = float(np.mean(S_vals))
    

    return f"{N},{J},{K},{seed},{R_mean},{S_mean}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--J", type=float, default=0.0)
    p.add_argument("--K", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=4000)
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


    seeds = [0, 1, 2]

    if args.sweep:
        Js = np.linspace(args.Jmin, args.Jmax, args.Jsteps)
        Ks = np.linspace(args.Kmin, args.Kmax, args.Ksteps)

        print("N,J,K,seed,R,S")

        for J in Js:
            for K in Ks:
                for seed in seeds:
                    print(run_once(args.N, float(J), float(K), seed, args.dt, args.steps, args.burnin, args.sample_every))

    else:
        print("N,J,K,seed,R,S")
        print(run_once(args.N, args.J, args.K, args.seed, args.dt, args.steps, args.burnin, args.sample_every))


if __name__ == "__main__":
    main()
