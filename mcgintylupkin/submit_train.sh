#!/bin/bash

for reward_std in 2
do
    sbatch run_train.sh ${reward_std}
done