#!/bin/bash

# for cost in 0.02 0.03 0.04
# do
#     for beta_e_final in 0.04
#     do
#         for kappa_squared in 0.0 0.1 0.2 0.3 0.4 0.5 0.6
#         do
#             sbatch run_analysis_backup.sh ${cost} ${beta_e_final} ${kappa_squared}
#         done
#     done
# done

for cost in 0.03
do
    for beta_e_final in 0.04
    do
        for kappa_squared in 0.0
        do
            sbatch run_analysis_backup.sh ${cost} ${beta_e_final} ${kappa_squared}
        done
    done
done