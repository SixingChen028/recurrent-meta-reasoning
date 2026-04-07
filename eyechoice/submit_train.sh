#!/bin/bash


for num_bandits in 2
do
    for stay_cost in 0.005 0.008
    do
        for switch_cost in 0.1 0.12
        do
            for beta_e_final in 0.015 0.02
            do
                sbatch run_train.sh ${num_bandits} ${stay_cost} ${switch_cost} ${beta_e_final}
            done
        done
    done
done


for num_bandits in 3
do
    for stay_cost in 0.005 0.008
    do
        for switch_cost in 0.1 0.12
        do
            for beta_e_final in 0.01 0.015
            do
                sbatch run_train.sh ${num_bandits} ${stay_cost} ${switch_cost} ${beta_e_final}
            done
        done
    done
done
