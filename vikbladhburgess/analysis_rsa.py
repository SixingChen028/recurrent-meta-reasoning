import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
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
exp_path = os.path.join(args.path, f'exp_{args.jobid}')

# # set printing path
# sys.stdout = open(os.path.join(exp_path, 'output_cog.txt'), 'w')

# initialize logger
logger = {}

# load net
net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

# set environment
env = MetaLearningWrapper(
    CircularRolloutEnv(
        reward_point = args.reward_point,
        default_point = args.default_point,
        t_max = args.t_max,
        cost = args.cost,
        aux_cost = 0.,
        scale_factor = args.scale_factor,
    )
)





"""
Preprocessing
"""

# load data
data = load_data(os.path.join(exp_path, f'data_simulation.p'))
data = preprocess(data, args)
num_trials = len(data['action_seqs'])
print('Keys:', data.keys())
print(num_trials)





"""
Set logger path
"""

# set experiment path
exp_path = os.path.join(args.path, f'logger_{args.jobid}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Test if rollout
"""

if_simulate_nexts = []
num_nodes = 9

for i in range(num_trials):
    length_ep, start_node_ep, fixation_seq_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'start_nodes', 'fixation_seqs', 'hidden_seqs'
    )

    if len(fixation_seq_ep[:-1]) == 2:
        # insert the start node
        inserted_fixation_seq_ep = fixation_seq_ep.copy()
        inserted_fixation_seq_ep.insert(0, start_node_ep)

        # compute if rollout
        for node, node_next in zip(inserted_fixation_seq_ep[:-1], inserted_fixation_seq_ep[1:-1]):
            if_simulate_nexts.append(int((node + 1) % num_nodes == node_next))

print(np.mean(if_simulate_nexts))
logger['if_simulate_next'] = np.mean(if_simulate_nexts)





"""
Collect data
"""

start_nodes = []
hidden_seqs = []

for i in range(num_trials):
    length_ep, start_node_ep, fixation_seq_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'start_nodes', 'fixation_seqs', 'hidden_seqs'
    )

    if len(fixation_seq_ep[:-1]) == 2:
        start_nodes.append(start_node_ep)
        hidden_seqs.append(hidden_seq_ep[:3])

start_nodes = np.array(start_nodes)
hidden_seqs = np.array(hidden_seqs)

print(hidden_seqs.shape)





"""
N vs. N RSA
"""

num_timepoints = hidden_seqs.shape[1]
num_splits = 1000
num_start_nodes = 9

# initialize matrix to accumulate correlations
all_matrix = np.zeros((num_splits, num_start_nodes, num_timepoints, num_timepoints))

# loop through splits
for split in range(num_splits):
    # randomly split all trials
    all_perm = np.random.permutation(len(hidden_seqs))
    half1_idx = all_perm[:len(hidden_seqs)//2]
    half2_idx = all_perm[len(hidden_seqs)//2:]
    
    # z-score across all trials in each half
    half1_z = np.zeros_like(hidden_seqs[half1_idx])
    half2_z = np.zeros_like(hidden_seqs[half2_idx])
    
    for t in range(num_timepoints):
        half1_z[:, t, :] = zscore(hidden_seqs[half1_idx, t, :], axis = 0)
        half2_z[:, t, :] = zscore(hidden_seqs[half2_idx, t, :], axis = 0)
    
    # compute correlations for each start node
    for start_node in range(num_start_nodes):
        # find which trials in each half belong to this start node
        idx_node_half1 = np.where(start_nodes[half1_idx] == start_node)[0]
        idx_node_half2 = np.where(start_nodes[half2_idx] == start_node)[0]
        
        # average the already z-scored values
        half1_mean = np.mean(half1_z[idx_node_half1], axis = 0)
        half2_mean = np.mean(half2_z[idx_node_half2], axis = 0)
        
        # compute correlations
        for i in range(num_timepoints):
            for j in range(num_timepoints):
                corr = np.corrcoef(half1_mean[i, :], half2_mean[j, :])[0, 1]
                all_matrix[split, start_node, i, j] = corr

# logger['matrix_n'] = all_matrix.mean(axis = (0, 1))
logger['matrix_n'] = np.nanmean(all_matrix, axis=(0, 1))




"""
N vs. N-1 RSA
"""

# N vs N-1 Analysis with global z-scoring
num_timepoints = hidden_seqs.shape[1]
num_splits = 1000
num_start_nodes = 9
num_trials_total = len(hidden_seqs)

# initialize matrix for N vs N-1 correlations
all_matrix = np.zeros((num_splits, num_start_nodes, num_timepoints, num_timepoints))

for split in range(num_splits):
    # randomly split all trials globally
    all_perm = np.random.permutation(len(hidden_seqs))
    half1_idx = all_perm[:len(hidden_seqs)//2]
    half2_idx = all_perm[len(hidden_seqs)//2:]
    
    # z-score across ALL trials in each half
    half1_z = np.zeros_like(hidden_seqs[half1_idx])
    half2_z = np.zeros_like(hidden_seqs[half2_idx])
    
    for t in range(num_timepoints):
        half1_z[:, t, :] = zscore(hidden_seqs[half1_idx, t, :], axis = 0)
        half2_z[:, t, :] = zscore(hidden_seqs[half2_idx, t, :], axis = 0)
    
    # compute N vs N-1 correlations
    for start_node_n in range(num_start_nodes):
        # get the previous node in the loop (N-1)
        start_node_n_minus_1 = (start_node_n - 1) % num_start_nodes
        
        # find trials for node N in each half
        idx_n_half1 = np.where(start_nodes[half1_idx] == start_node_n)[0]
        idx_n_half2 = np.where(start_nodes[half2_idx] == start_node_n)[0]
        
        # find trials for node N-1 in each half
        idx_n_minus_1_half1 = np.where(start_nodes[half1_idx] == start_node_n_minus_1)[0]
        idx_n_minus_1_half2 = np.where(start_nodes[half2_idx] == start_node_n_minus_1)[0]
        
        # average the already z-scored values for node N
        half1_mean_n = np.mean(half1_z[idx_n_half1], axis = 0)
        half2_mean_n = np.mean(half2_z[idx_n_half2], axis = 0)
        
        # average the already z-scored values for node N-1
        half1_mean_n_minus_1 = np.mean(half1_z[idx_n_minus_1_half1], axis = 0)
        half2_mean_n_minus_1 = np.mean(half2_z[idx_n_minus_1_half2], axis = 0)
        
        # compute cross-correlation matrix
        # Compare N (on x-axis) with N-1 (on y-axis)
        for i in range(num_timepoints):  # Time for N
            for j in range(num_timepoints):  # Time for N-1
                # correlate across both half comparisons and average
                corr1 = np.corrcoef(half1_mean_n[i, :], half1_mean_n_minus_1[j, :])[0, 1]
                corr2 = np.corrcoef(half2_mean_n[i, :], half2_mean_n_minus_1[j, :])[0, 1]
                
                all_matrix[split, start_node_n, i, j] = (corr1 + corr2) / 2

# logger['matrix_n_minus_1'] = all_matrix.mean(axis = (0, 1))
logger['matrix_n_minus_1'] = np.nanmean(all_matrix, axis=(0, 1))




"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_rsa.pkl'), 'wb') as f:
    pickle.dump(logger, f)