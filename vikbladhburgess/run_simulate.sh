#!/bin/bash
#SBATCH --job-name=ep
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=32G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-4

python -u simulate.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results