#!/bin/bash
#SBATCH --job-name=ec
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=1G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0

python -u plot_decoding.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results