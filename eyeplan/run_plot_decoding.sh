#!/bin/bash
#SBATCH --job-name=ep
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=5G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0

python -u plot_decoding.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --cost=${1} \
    --beta_e_final=${2} \
    --kappa_squared=${3}