import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
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
# sys.stdout = open(os.path.join(exp_path, 'output_logit.txt'), 'w')

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
data = preprocess(data, args, merge_fixations = False) #######
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
Parent's logit change given revealed node and it's sibling value
"""

records = []

for i in range(num_trials):
    length_ep, child_dict_ep, parent_dict_ep, root_node_ep, points_ep, depths_ep, fixation_seq_ep, logits_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'parent_dicts', 'root_nodes', 'points', 'depths', 'fixation_seqs', 'logits_seqs'
    )

    if length_ep >= args.t_max or len(fixation_seq_ep) < 2:
        continue
    
    # initialize node mask
    node_mask_ep = np.zeros((11))
    node_mask_ep[root_node_ep] = 1.

    # compute clipped q values sequence
    q_values_seq_ep = []
    for node in fixation_seq_ep:
        q_values_t = get_q_values(child_dict_ep, points_ep * node_mask_ep)
        q_values_seq_ep.append(q_values_t)
        node_mask_ep[node] = 1
    q_values_seq_ep = np.array(q_values_seq_ep)

    # build sibling dict: node -> list of siblings (same parent, different node)
    sibling_dict = {}
    for node, par in parent_dict_ep.items():
        siblings = [c for c in child_dict_ep.get(par, []) if c != node]
        sibling_dict[node] = siblings[0]

    # track which nodes have been fixated so far
    visited = set()
    visited.add(root_node_ep)

    for t in range(1, len(fixation_seq_ep) - 1):
        node = fixation_seq_ep[t]

        # node is fixated for the first time
        if node in visited:
            visited.add(node)
            continue
        
        # node has depth > 1
        if depths_ep[node] <= 1:
            visited.add(node)
            continue

        # get parent and sibling
        par = parent_dict_ep.get(node, None)
        sibling = sibling_dict.get(node, None)

        ######################################
        if fixation_seq_ep[t - 1] == sibling:
            visited.add(node)
            continue
        ######################################

        # get sibling value
        sibling_value = q_values_seq_ep[t, sibling]
        if sibling_value not in [-8, -4, -2, -1, 1, 2, 4, 8]:
            visited.add(node)
            continue

        # revealed reward of the fixated node itself
        revealed_reward = points_ep[node]

        # get parent's delta q
        delta_q_parent = q_values_seq_ep[t + 1, par] - q_values_seq_ep[t, par]

        records.append({
            'revealed_reward': revealed_reward,
            'sibling_value':   sibling_value,
            'delta_q_parent':   delta_q_parent,
        })

        visited.add(node)

df_sib = pd.DataFrame(records)

logger['df_sib'] = df_sib





"""
Grandparent's logit change given parent's Q value change and parent's sibling value
"""

records = []

for i in range(num_trials):
    length_ep, child_dict_ep, parent_dict_ep, root_node_ep, points_ep, depths_ep, fixation_seq_ep, logits_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'parent_dicts', 'root_nodes', 'points', 'depths', 'fixation_seqs', 'logits_seqs'
    )

    if length_ep >= args.t_max or len(fixation_seq_ep) < 2:
        continue

    # initialize node mask
    node_mask_ep = np.zeros((11))
    node_mask_ep[root_node_ep] = 1.

    # compute clipped q values sequence
    q_values_seq_ep = []
    for node in fixation_seq_ep:
        q_values_t = get_q_values(child_dict_ep, points_ep * node_mask_ep)
        q_values_seq_ep.append(q_values_t)
        node_mask_ep[node] = 1
    q_values_seq_ep = np.array(q_values_seq_ep)

    # build sibling dict: node -> sibling (same parent, different node)
    sibling_dict = {}
    for node, par in parent_dict_ep.items():
        siblings = [c for c in child_dict_ep.get(par, []) if c != node]
        sibling_dict[node] = siblings[0]

    # track which nodes have been fixated so far
    visited = set()
    visited.add(root_node_ep)

    for t in range(1, len(fixation_seq_ep) - 1):
        node = fixation_seq_ep[t]

        # node is being fixated for the first time
        if node in visited:
            visited.add(node)
            continue

        # node has depth > 2
        if depths_ep[node] <= 2:
            visited.add(node)
            continue

        # get parent and grandparent
        par = parent_dict_ep.get(node, None)
        grandpar = parent_dict_ep.get(par, None)

        # parent's sibling Q value
        par_sibling = sibling_dict.get(par, None)
        par_sibling_action_idx = args.num_nodes + par_sibling
        par_sibling_q = q_values_seq_ep[t, par_sibling]

        ######################################
        if fixation_seq_ep[t - 1] == par_sibling:
            visited.add(node)
            continue
        ######################################

        if par_sibling_q not in [-8, -4, -2, -1, 1, 2, 4, 8]:
            visited.add(node)
            continue

        # delta Q of parent
        delta_q_par = q_values_seq_ep[t + 1, par] - q_values_seq_ep[t, par]

        if delta_q_par not in [-8, -4, -2, -1, 1, 2, 4, 8]:
            visited.add(node)
            continue

        # delta q of the GRANDPARENT
        delta_q_grandpar = q_values_seq_ep[t + 1, grandpar] - q_values_seq_ep[t, grandpar]

        records.append({
            'revealed_reward': points_ep[node],
            'delta_q_par':    delta_q_par,
            'par_sibling_q':      par_sibling_q,
            'delta_q_grandpar': delta_q_grandpar,
        })

        visited.add(node)

df_sib = pd.DataFrame(records)

logger['df_sib_grand'] = df_sib





"""
Grandgrandparent's logit change given grandparent's Q value change and grandparent's sibling value
"""

records = []

for i in range(num_trials):
    length_ep, child_dict_ep, parent_dict_ep, root_node_ep, points_ep, depths_ep, fixation_seq_ep, logits_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'parent_dicts', 'root_nodes', 'points', 'depths', 'fixation_seqs', 'logits_seqs'
    )

    if length_ep >= args.t_max or len(fixation_seq_ep) < 2:
        continue

    # initialize node mask
    node_mask_ep = np.zeros((11))
    node_mask_ep[root_node_ep] = 1.

    # compute clipped q values sequence
    q_values_seq_ep = []
    for node in fixation_seq_ep:
        q_values_t = get_q_values(child_dict_ep, points_ep * node_mask_ep)
        q_values_seq_ep.append(q_values_t)
        node_mask_ep[node] = 1
    q_values_seq_ep = np.array(q_values_seq_ep)

    # build sibling dict: node -> sibling (same parent, different node)
    sibling_dict = {}
    for node, par in parent_dict_ep.items():
        siblings = [c for c in child_dict_ep.get(par, []) if c != node]
        sibling_dict[node] = siblings[0]

    # track which nodes have been fixated so far
    visited = set()
    visited.add(root_node_ep)

    for t in range(1, len(fixation_seq_ep) - 1):
        node = fixation_seq_ep[t]

        # node is being fixated for the first time
        if node in visited:
            visited.add(node)
            continue

        # node has depth > 3
        if depths_ep[node] <= 3:
            visited.add(node)
            continue

        # get parent, grandparent, and grandgrandparent
        par = parent_dict_ep.get(node, None)
        grandpar = parent_dict_ep.get(par, None)
        grandgrandpar = parent_dict_ep.get(grandpar, None)

        # grandparent's sibling Q value
        grandpar_sibling = sibling_dict.get(grandpar, None)
        grandpar_sibling_q = q_values_seq_ep[t, grandpar_sibling]

        ######################################
        if fixation_seq_ep[t - 1] == grandpar_sibling:
            visited.add(node)
            continue
        ######################################

        if grandpar_sibling_q not in [-8, -4, -2, -1, 1, 2, 4, 8]:
            visited.add(node)
            continue

        # delta Q of grandparent
        delta_q_grandpar = q_values_seq_ep[t + 1, grandpar] - q_values_seq_ep[t, grandpar]

        if delta_q_grandpar not in [-8, -4, -2, -1, 1, 2, 4, 8]:
            visited.add(node)
            continue

        # delta q of the GRANDGRANDPARENT
        delta_q_grandgrandpar = q_values_seq_ep[t + 1, grandgrandpar] - q_values_seq_ep[t, grandgrandpar]

        records.append({
            'revealed_reward': points_ep[node],
            'delta_q_grandpar':          delta_q_grandpar,
            'grandpar_sibling_q':        grandpar_sibling_q,
            'delta_q_grandgrandpar': delta_q_grandgrandpar,
        })

        visited.add(node)

df_sib = pd.DataFrame(records)

logger['df_sib_grandgrand'] = df_sib





"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_backup_bellman.pkl'), 'wb') as f:
    pickle.dump(logger, f)