#!/bin/bash

# for cost in 0.02 0.025 0.03
# do
#     for beta_e_final in 0.01 0.02 0.03 0.04 0.05
#     do
#         sbatch run_plot_decoding.sh ${cost} ${beta_e_final}
#     done
# done

for cost in 0.03
do
    for beta_e_final in 0.04
    do
        for kappa_squared in 0.0
        do
            sbatch run_plot_decoding.sh ${cost} ${beta_e_final} ${kappa_squared}
        done
    done
done