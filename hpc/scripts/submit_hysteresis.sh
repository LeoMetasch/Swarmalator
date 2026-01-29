#!/bin/bash
#SBATCH --job-name=hysteresis
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/hysteresis_%j.out
#SBATCH --error=slurm_logs/hysteresis_%j.err

# Hysteresis study: 30 forward + 30 backward K-sweeps
# Uses Python multiprocessing to parallelize across 60 CPUs

# Create output directories
mkdir -p slurm_logs results/hysteresis/forward results/hysteresis/backward

# Load modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source env/bin/activate

echo "Starting hysteresis sweep on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

# Run the parallel Python script
python hpc/run_hysteresis.py \
    --n_seeds 30 \
    --n_workers $SLURM_CPUS_PER_TASK \
    --output_dir results/hysteresis

echo "End time: $(date)"
echo "Results saved to results/hysteresis/"
