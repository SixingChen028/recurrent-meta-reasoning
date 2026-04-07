#!/bin/bash
#SBATCH --job-name=ep
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=15G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0

python -u plot_noise.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results