from pathlib import Path

import numpy as np

from src.swarmalator import Swarmalator


def run_experiment():
	"""Run a sample simulation and log order parameters to CSV."""

	# Core parameters for this run
	N = 100
	J = 0.9
	K = 0.3
	dt = 0.5
	eps = 1e-6
	steps = 5000
	rng = np.random.default_rng(seed=1)
	x = rng.uniform(-1, 1, N)
	y = rng.uniform(-1, 1, N)
	theta = rng.uniform(-np.pi, np.pi, N)

	swarmalator = Swarmalator(N=N, J=J, K=K, dt=dt, x=x, y=y, theta=theta, eps=eps)
	swarmalator.animate(steps=1000)  # shows interactive plot

	log_file = Path("logs/experiment_log.csv")
	swarmalator.run_with_logging(steps=steps, log_path=log_file, log_interval=10)

	print(f"Logged order parameters to {log_file.resolve()}")


if __name__ == "__main__":
	run_experiment()
