#!/bin/bash
#SBATCH --job-name=ksweep_critical
#SBATCH --account=          # <-- Replace with your project account
#SBATCH --partition=genoa               # Options: genoa (192 cores), rome (128 cores), thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1              # Single continuous simulation
#SBATCH --time=04:00:00                 # Max walltime (HH:MM:SS)
#SBATCH --output=slurm_logs/ksweep_%j.out
#SBATCH --error=slurm_logs/ksweep_%j.err

# Create output directories
mkdir -p slurm_logs results

# Load Python module (adjust version as needed)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source env/bin/activate

# Run continuous K-sweep
echo "Starting continuous K-sweep on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# K ramps from -1.0 to 0.2 over 120,000 steps
# dK/step ≈ 0.00001 (very slow to capture transitions)
python hpc/run_ksweep.py \
    --N 100 \
    --J 0.5 \
    --Kmin -1.0 \
    --Kmax 0.2 \
    --steps 120000 \
    --log_interval 10 \
    --seed 42 \
    --output results/ksweep_$SLURM_JOB_ID.csv

echo "End time: $(date)"
echo "Results saved to results/ksweep_$SLURM_JOB_ID.csv"
