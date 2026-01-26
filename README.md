# Swarmalator
Implementation of the swarmaltor model based on the paper Oscillators that sync and swarm by O'Keeffe et al. (2017) https://www.nature.com/articles/s41467-017-01190-3

# Implementation
![alt text](image.png)

# TO RUN
uv venv 
uv sync
.venv\Scripts\activate

# Logging order parameters
- Activate the virtualenv and run `python` on [main.py](main.py) to execute a sample experiment.
- The script writes order-parameter snapshots to [logs/experiment_log.csv](logs/experiment_log.csv) with columns: step, S, V, omega, R, J, K, N.
- Adjust parameters [main.py](main.py) (see `run_experiment`) as needed for custom sweeps.
