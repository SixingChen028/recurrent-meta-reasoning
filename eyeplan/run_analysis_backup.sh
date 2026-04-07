#!/bin/bash
#SBATCH --job-name=ep
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=16G
#SBATCH -e ./results/slurm-%A_%a.err
#SBATCH -o ./results/slurm-%A_%a.out
#SBATCH --array=0-4

python -u analysis_backup.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --cost=${1} \
    --beta_e_final=${2} \
    --kappa_squared=${3}


python -u analysis_backup_bellman.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --cost=${1} \
    --beta_e_final=${2} \
    --kappa_squared=${3}


python -u analysis_backup_bellman_opt.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --cost=${1} \
    --beta_e_final=${2} \
    --kappa_squared=${3}


python -u analysis_backup_predecessor.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --cost=${1} \
    --beta_e_final=${2} \
    --kappa_squared=${3}

python -u analysis_backup_predecessor_opt.py \
    --jobid=$SLURM_ARRAY_TASK_ID \
    --path=./results \
    --cost=${1} \
    --beta_e_final=${2} \
    --kappa_squared=${3}