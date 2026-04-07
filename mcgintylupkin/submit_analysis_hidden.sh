#!/bin/bash

for reward_std in 2 #3 #4 6 8
do
    sbatch run_analysis_hidden.sh ${reward_std}
done
