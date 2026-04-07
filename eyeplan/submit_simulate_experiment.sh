#!/bin/bash

for cost in 0.01
do
    for kappa_squared in 0.0 0.1 0.2 0.3 0.4 0.5 0.6
    do
        sbatch run_simulate_experiment.sh ${cost} ${kappa_squared}
    done
done