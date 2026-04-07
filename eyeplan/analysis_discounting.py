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
exp_path = os.path.join(args.path, f'exp_{args.cost}_{args.beta_e_final}_{args.kappa_squared}_{args.jobid}')

# # set printing path
# sys.stdout = open(os.path.join(exp_path, 'output_cog.txt'), 'w')

# initialize logger
logger = {}

# load net
net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

# set environment
env = MetaLearningWrapper(
    DecisionTreeEnv(
        num_nodes = args.num_nodes,
        t_max = args.t_max,
        cost = args.cost,
        scale_factor = args.scale_factor,
        shuffle_nodes = args.shuffle_nodes,
        mask_fixation = args.mask_fixation,
    )
)





"""
Preprocessing
"""

# load data
data = load_data(os.path.join(exp_path, f'data_simulation.p'))
data = preprocess(data, args, merge_fixations = True)
num_trials = len(data['action_seqs'])
print('Keys:', data.keys())
print(num_trials)





"""
Set logger path
"""

# set experiment path
exp_path = os.path.join(args.path, f'logger_{args.cost}_{args.beta_e_final}_{args.kappa_squared}_{args.jobid}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Fitting
"""

def neg_log_likelihood(params, trials):
    """
    Softmax choice model fitted to every sequential expansion step.

    params: [logit_gamma, log_beta]
        gamma = sigmoid(logit_gamma) in (0, 1)
        beta = exp(log_beta) > 0
    """
    gamma = 1.0 / (1.0 + np.exp(-params[0]))
    beta  = np.exp(params[1])
 
    nll = 0.0
 
    for child_dict, root_node, points, fixation_seq, decision_seq in trials:
        points = np.array(points, dtype=float)
 
        # knowledge mask: starts with root revealed
        node_mask = np.zeros(len(points))
        node_mask[root_node] = 1.0
        for node in fixation_seq:
            node_mask[node] = 1.0
        
        # Q values computed once from revealed rewards
        q = get_q_values(child_dict, points * node_mask, gamma=gamma)
 
        # agent starts at root; fixation_seq is the sequence of chosen children
        current = root_node
        for chosen in decision_seq:
            children = child_dict.get(current)
            if children is None or chosen not in children:
                break
 
            q_children = np.array([q[c] for c in children])
            chosen_idx  = children.index(chosen)
 
            shifted  = beta * q_children
            log_prob = shifted[chosen_idx] - (
                np.max(shifted) + np.log(np.sum(np.exp(shifted - np.max(shifted))))
            )
            nll -= log_prob
 
            # reveal chosen node and move into it
            current = chosen
 
    return nll

# gather data
trials = list(zip(
    data['child_dicts'], data['root_nodes'], data['points'],
    data['fixation_seqs'], data['decision_seqs']
))

print('Fitting gamma ...')

best_nll, best_params = np.inf, None

for _ in range(10):
    gamma0 = np.random.uniform(0.1, 1.0)
    beta0  = np.random.uniform(0.1, 2.0)
    x0 = [np.log(gamma0 / (1.0 - gamma0)), np.log(beta0)]
 
    res = minimize(
        neg_log_likelihood,
        x0,
        args = (trials,),
        method = 'Nelder-Mead',
        options = {
            'maxiter': 5000,
            'xatol': 1e-6,
            'fatol': 1e-6
        },
    )
    if res.fun < best_nll:
        best_nll, best_params = res.fun, res.x
    
    print(f'Finish {_} / 10')

gamma_fit = 1.0 / (1.0 + np.exp(-best_params[0]))
beta_fit  = np.exp(best_params[1])

print(f'  gamma = {gamma_fit:.4f}')
print(f'  beta  = {beta_fit:.4f}')
print(f'  NLL   = {best_nll:.2f}')

logger['gamma'] = gamma_fit
logger['beta'] = beta_fit
logger['nll'] = best_nll





"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_discount.pkl'), 'wb') as f:
    pickle.dump(logger, f)