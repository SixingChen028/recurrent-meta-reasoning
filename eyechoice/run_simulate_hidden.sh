#!/bin/bash
#SBATCH --job-name=ec
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=4

python -u simulate_hidden.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --num_bandits=${1} \
    --stay_cost=${2} \
    --switch_cost=${3} \
    --beta_e_final=${4}
