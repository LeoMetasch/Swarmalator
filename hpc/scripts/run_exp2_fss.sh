#!/bin/bash
# Experiment 2: Finite-Size Scaling
# BEFORE SUBMITTING: mkdir -p slurm_logs results/exp2_fss

#SBATCH --job-name=exp2_fss
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/exp2_%A_%a.out
#SBATCH --error=slurm_logs/exp2_%A_%a.err
#SBATCH --array=0-14  # 3 N values × 5 seeds = 15 jobs

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source env/bin/activate

# Map array index to N and seed
N_VALUES=(100 200 400)
SEEDS=(0 1 2 3 4)

JOB_IDX=${SLURM_ARRAY_TASK_ID:-0}
N_IDX=$((JOB_IDX / 5))
SEED_IDX=$((JOB_IDX % 5))

N=${N_VALUES[$N_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Experiment 2: Finite-Size Scaling"
echo "N=$N, J=0.5, K in [-0.5, -0.4]"
echo "Seed: $SEED"

python hpc/run_ksweep.py \
    --N $N --J 0.5 \
    --Kmin -0.5 --Kmax -0.4 --dK 0.005 \
    --steps_per_K 5000 --log_interval 10 \
    --seed $SEED \
    --independent \
    --output results/exp2_fss/ksweep_N${N}_seed${SEED}.csv

echo "Done!"
