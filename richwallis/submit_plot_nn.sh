#!/bin/bash


for reward_std in 2
do
    for stay_cost in 0.07
    do
        for switch_cost in 0.1
        do
            for beta_e_final in 0.04
            do
                sbatch run_plot_nn.sh ${reward_std} ${num_bandits} ${stay_cost} ${switch_cost} ${beta_e_final}
            done
        done
    done
done
