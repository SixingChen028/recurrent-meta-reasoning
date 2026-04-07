import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from modules import *

NUM_JOBS = 5





"""
Set environment
"""

# parse args
parser = ArgParser()
args = parser.args





"""
Read data
"""

data = []
for jobid in range(NUM_JOBS):
    exp_path = os.path.join(args.path, f'exp_{args.num_bandits}_{jobid}')

    with open(os.path.join(exp_path, 'data_training.p'), 'rb') as file:
        data_jobid = pickle.load(file)

    data.append(data_jobid)

# print(data[0].keys())




"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure_{args.num_bandits}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Training reward
"""

rewards = np.array([np.array(data_jobid['episode_reward']).reshape(100, -1).mean(axis = 1) for data_jobid in data])
mean_rewards = np.mean(rewards, axis = 0)
sem_rewards = np.std(rewards, axis = 0)
plt.figure(figsize = (2.8, 2.5))
plt.plot(mean_rewards)
plt.fill_between(range(len(mean_rewards)), mean_rewards - sem_rewards, mean_rewards + sem_rewards, alpha = 0.2)
plt.xlabel('Episode percentage')
plt.ylabel('Episode reward')
plt.savefig(os.path.join(exp_path, 'p_training_reward.svg'), bbox_inches = 'tight')





"""
Training lengths
"""

lengths = np.array([np.array(data_jobid['episode_length']).reshape(100, -1).mean(axis = 1) for data_jobid in data])
mean_lengths = np.mean(lengths, axis = 0)
sem_lengths = np.std(lengths, axis = 0)
plt.figure(figsize = (2.8, 2.5))
plt.plot(mean_lengths)
plt.fill_between(range(len(mean_lengths)), mean_lengths - sem_lengths, mean_lengths + sem_lengths, alpha = 0.2)
plt.xlabel('Episode percentage')
plt.ylabel('Episode length')
plt.savefig(os.path.join(exp_path, 'p_training_length.svg'), bbox_inches = 'tight')
