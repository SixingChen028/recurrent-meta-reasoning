#!/bin/bash
#SBATCH --job-name=ec
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-4

python -u train.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --reward_std=${1}
