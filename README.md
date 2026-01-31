# Swarmalator
Implementation of the swarmalator model based on the paper "Oscillators that sync and swarm" by O'Keeffe et al. (2017) https://www.nature.com/articles/s41467-017-01190-3

![alt text](image.png)

## Quick Start
```bash
uv venv 
uv sync
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate on Windows
```

## Logging Order Parameters
- Run `python main.py` to execute a sample experiment
- Order-parameter snapshots are written to `logs/experiment_log.csv` with columns: step, S, V, omega, R, J, K, N
- Adjust parameters in `main.py` (see `run_experiment`) for custom sweeps

---

## Directory Structure

### `hpc/` - High Performance Computing Simulations

Scripts for running parameter sweeps and phase transition analysis on HPC clusters.

> **Note on Data Storage**: These scripts are designed for Snellius HPC cluster submission. Full experimental runs generate datasets exceeding 1 GB, which cannot be uploaded to GitHub (even with LFS). Results are stored locally in `hpc/results/` (which is gitignored).

**Simulation Runners:**
| File | Description |
|------|-------------|
| `run.py` | Parallel J/K parameter sweep with multiprocessing |
| `run_ksweep.py` | Continuous K-sweep for observing phase transitions |
| `run_hysteresis.py` | Forward + backward sweeps to detect hysteresis |
| `swarm.py` | Core Swarmalator model with Numba acceleration |

**Analysis Scripts (`hpc/analysis/`):**
| File | Description |
|------|-------------|
| `analyze_beta.py` | Extract critical exponent β via log-log regression |
| `analyze_fss.py` | Finite-size scaling analysis for critical exponents |
| `analyze_hysteresis.py` | Compare forward/backward sweeps for hysteresis loops |

**HPC Job Scripts (`hpc/scripts/`):**
| File | Description |
|------|-------------|
| `submit_sweep.sh` | SLURM script for parameter sweeps |
| `submit_ksweep.sh` | SLURM script for K-sweep experiments |
| `submit_hysteresis.sh` | SLURM script for hysteresis runs |

#### Local Testing (Small N)

For local testing before HPC submission, use these reduced parameter sets:

```bash
# Quick K-sweep (outputs to hpc/results/test/)
python hpc/run_ksweep.py \
  --N 50 \
  --J 0.5 \
  --Kmin -1.0 \
  --Kmax 0.2 \
  --dK 0.05 \
  --steps_per_K 500 \
  --log_interval 10 \
  --output hpc/results/test/ksweep_N50.csv

# Parameter sweep (outputs to stdout, redirect to file)
python hpc/run.py \
  --N 30 \
  --sweep \
  --Jmin 0.5 --Jmax 1.0 --Jsteps 5 \
  --Kmin -1.0 --Kmax 0.5 --Ksteps 5 \
  --workers 4 \
  > hpc/results/test/sweep_N30.csv

# Hysteresis test (30 seeds, outputs to hpc/results/test/hysteresis/)
python hpc/run_hysteresis.py \
  --n_seeds 5 \
  --n_workers 4 \
  --output_dir hpc/results/test/hysteresis
```

#### HPC Production Runs

For Snellius cluster (large N, full parameter ranges):

```bash
# Submit via SLURM scripts
sbatch hpc/scripts/submit_ksweep.sh    # N=100-400, outputs to hpc/results/ksweep/
sbatch hpc/scripts/submit_sweep.sh     # Full J/K grid, outputs to hpc/results/sweep/
sbatch hpc/scripts/submit_hysteresis.sh # 30 seeds, outputs to hpc/results/hysteresis/
```

#### Analysis Scripts

All analysis scripts read from `hpc/results/` and save outputs to the same directory.

```bash
# Critical exponent β (requires ksweep data)
python hpc/analysis/analyze_beta.py \
  --data_dir hpc/results/ksweep \
  --Kc_file hpc/results/Kc.json \
  --output_dir hpc/results/beta_analysis

# Finite-size scaling (requires multiple N values)
python hpc/analysis/analyze_fss.py \
  --data_dir hpc/results/fss \
  --N_values 10 20 40 \
  --output_dir hpc/results/fss_analysis

# Hysteresis analysis (requires forward/backward sweep data)
python hpc/analysis/analyze_hysteresis.py \
  --forward_dir hpc/results/hysteresis/forward \
  --backward_dir hpc/results/hysteresis/backward \
  --output_dir hpc/results/hysteresis_analysis
```

**Output files:**
- Analysis scripts generate: `*.png` (plots), `*.csv` (statistics), `*.json` (fitted parameters)
- All outputs saved to respective `--output_dir` directories

---

### `demo/` - Interactive Web Visualization

Browser-based interactive swarmalator simulation.

| File | Description |
|------|-------------|
| `index.html` | Main page with controls and visualization canvas |
| `simulation.js` | JavaScript implementation of swarmalator dynamics |
| `style.css` | Styling for the demo interface |

**Features:**
- Real-time visualization of swarmalator dynamics
- Adjustable parameters: J (spatial coupling), K (phase coupling), N (particle count)
- Order parameter plots: S (correlation), V (velocity), Ω (phase velocity), R (synchrony)
- Preset configurations for different collective states
- Predator mode: click to spawn, right-click to remove

**To run:**
```bash
# Option 1: Open directly
open demo/index.html

# Option 2: Local server (recommended)
python -m http.server 8000 --directory demo
# Then visit http://localhost:8000
```
