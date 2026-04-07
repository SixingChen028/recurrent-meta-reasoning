import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')

from modules import *





"""
Set environment
"""

# set random seed
seed = 15
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# parse args
parser = ArgParser()
args = parser.args

# set experiment path
exp_path = os.path.join(args.path, f'exp_{args.num_bandits}_{args.jobid}')

# # set printing path
# sys.stdout = open(os.path.join(exp_path, 'output_cog.txt'), 'w')

# initialize logger
logger = {}

# load net
net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

# load data
data = pickle.load(open(os.path.join(exp_path, f'data_training.p'), 'rb'))





"""
Rewards and episode lengths
"""

rewards = np.array(data['episode_reward']).reshape(100, -1)
mean_rewards = np.mean(rewards, axis = 1)
sem_rewards = np.std(rewards, axis = 1)
logger['mean_rewards'] = mean_rewards
logger['sem_rewards'] = sem_rewards

lengths = np.array(data['episode_length']).reshape(100, -1)
mean_lengths = np.mean(lengths, axis = 1)
sem_lengths = np.std(lengths, axis = 1)
logger['mean_lengths'] = mean_lengths
logger['sem_lengths'] = sem_lengths





"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_training.pkl'), 'wb') as f:
    pickle.dump(logger, f)