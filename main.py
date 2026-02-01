from pathlib import Path

import numpy as np
from src.swarm import Swarm

def run_experiment() -> None:
	"""Run a sample simulation and log order parameters to CSV."""

	# Core parameters for this run
	N = 300
	J = 0.9
	K = 0
	dt = 0.1
	steps = 1000

	swarm = Swarm(N=N, dt=dt, J=J, K=K, steps=steps, chirality=False, phase_coupling=False, predator=False)
	swarm.animate(steps)

	order_parameters = swarm.stability_analysis()
	print(order_parameters)

	# log_file = Path("logs/experiment_log.csv")
	# swarm.run_with_logging(steps=steps, log_path=log_file, log_interval=1000)
	# swarm.run(steps=steps)

	# print(f"Logged order parameters to {log_file.resolve()}")


if __name__ == "__main__":
	run_experiment()
