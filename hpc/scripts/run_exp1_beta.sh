#!/bin/bash
# Experiment 1: β Exponent Extraction
# BEFORE SUBMITTING:   

#SBATCH --job-name=exp1_beta
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/exp1_%A_%a.out
#SBATCH --error=slurm_logs/exp1_%A_%a.err
#SBATCH --array=0-9  # 10 seeds

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source env/bin/activate

SEED=${SLURM_ARRAY_TASK_ID:-0}

echo "Experiment 1: β Exponent"
echo "N=400, J=0.5, K in [-0.6, -0.4]"
echo "Seed: $SEED"

python hpc/run_ksweep.py \
    --N 400 --J 0.5 \
    --Kmin -0.6 --Kmax -0.4 --dK 0.002 \
    --steps_per_K 10000 --log_interval 10 \
    --seed $SEED \
    --independent \
    --output results/exp1_beta/ksweep_seed${SEED}.csv

echo "Done!"
