#!/bin/bash
#SBATCH --job-name=fss_ksweep
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/fss_%A_%a.out
#SBATCH --error=slurm_logs/fss_%A_%a.err
#SBATCH --array=0-3              # 4 different N values

# Create output directories
mkdir -p slurm_logs results/fss

# Load Python module
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source env/bin/activate

# Define N values for finite-size scaling
N_VALUES=(50 100 200 400)
N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Starting finite-size scaling K-sweep"
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "N = $N"
echo "Start time: $(date)"

# Run K-sweep for this N value
python hpc/run_ksweep.py \
    --N $N \
    --J 0.5 \
    --Kmin -1.0 \
    --Kmax 0.2 \
    --dK 0.01 \
    --steps_per_K 1000 \
    --log_interval 1 \
    --seed 42 \
    --output results/fss/ksweep_N${N}.csv

echo "End time: $(date)"
echo "Results saved to results/fss/ksweep_N${N}.csv"
