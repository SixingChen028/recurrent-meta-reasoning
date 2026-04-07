#!/bin/bash
#SBATCH --job-name=ep
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=16G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-4

python -u analysis_cogsci.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --cost=${1} \
    --beta_e_final=${2} \
    --kappa_squared=${3}
