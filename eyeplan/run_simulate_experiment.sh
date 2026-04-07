#!/bin/bash
#SBATCH --job-name=ep
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-4

python -u simulate_experiment.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --cost=${1} \
    --kappa_squared=${2}