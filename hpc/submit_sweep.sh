#!/bin/bash
#SBATCH --job-name=swarmalator_sweep
#SBATCH --partition=genoa               # Options: genoa (192 cores), rome (128 cores), thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192            # Number of parallel workers
#SBATCH --time=24:00:00                 # Max walltime (HH:MM:SS)
#SBATCH --output=slurm_logs/sweep_%j.out
#SBATCH --error=slurm_logs/sweep_%j.err

# Create output directories
mkdir -p slurm_logs results final_states logs

# Load Python module (adjust version as needed)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source env/bin/activate

# Run parameter sweep with all available CPUs
echo "Starting sweep with $SLURM_CPUS_PER_TASK workers on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python hpc/run.py \
    --N 100 \
    --sweep \
    --Jmin 0 --Jmax 1 --Jsteps 100 \
    --Kmin -1.0 --Kmax 0.2 --Ksteps 100 \
    --steps 10000 \
    --workers $SLURM_CPUS_PER_TASK \
    > results/sweep_$SLURM_JOB_ID.csv

echo "End time: $(date)"
echo "Results saved to results/sweep_$SLURM_JOB_ID.csv"
