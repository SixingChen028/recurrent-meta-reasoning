#!/bin/bash
#SBATCH --job-name=ec
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=5G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0

python -u plot_nn.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --reward_std=${1} \
    --stay_cost=${2} \
    --switch_cost=${3} \
    --beta_e_final=${4}


