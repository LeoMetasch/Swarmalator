#!/bin/bash
#SBATCH --job-name=ksweep_critical
#SBATCH --partition=genoa               # Options: genoa (192 cores), rome (128 cores), thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00                 # Max walltime (HH:MM:SS)
#SBATCH --output=slurm_logs/ksweep_%j.out
#SBATCH --error=slurm_logs/ksweep_%j.err

# Create output directories
mkdir -p slurm_logs results

# Load Python module (adjust version as needed)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source env/bin/activate

# Run K-sweep
echo "Starting K-sweep on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# dK = 0.01 → 120 K values from -1.0 to 0.2
# steps_per_K = 1000 → observe 1000 steps at each K
# Total: 120,000 steps
python hpc/run_ksweep.py \
    --N 100 \
    --J 0.5 \
    --Kmin -1.0 \
    --Kmax 0.2 \
    --dK 0.05 \
    --steps_per_K 1000 \
    --log_interval 1 \
    --seed 42 \
    --output results/ksweep_$SLURM_JOB_ID.csv

echo "End time: $(date)"
echo "Results saved to results/ksweep_$SLURM_JOB_ID.csv"
