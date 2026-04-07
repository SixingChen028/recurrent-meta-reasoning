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
Effect of fixation on the state
"""

points = []
logits_slices = []

for i in range(num_trials):
    length_ep, child_dict_ep, points_ep, logits_seq_ep, fixation_seq_ep, decision_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'points', 'logits_seqs', 'fixation_seqs', 'decision_seqs',
    )

    if length_ep < args.t_max and len(fixation_seq_ep) >= 4:
        # process logit
        logits_seq_ep = np.array(logits_seq_ep) # (num_timesteps, num_actions)
        logits_seq_ep = normalize_logits(child_dict_ep, logits_seq_ep)

        # loop slices of 4 time steps
        for t in range(0, len(fixation_seq_ep) - 4):
            node = fixation_seq_ep[t]

            logits_slices.append(logits_seq_ep[t : t + 4, args.num_nodes + node] - logits_seq_ep[t, args.num_nodes + node]) ###############
            points.append(points_ep[node])
    
points = np.array(points)
logits_slices = np.array(logits_slices)

df_combined = pd.DataFrame()
for i, point in enumerate(env.env.point_set):
    indices = np.where(points == point)[0]
    logits_slices_indices = logits_slices[indices].mean(axis = 0)
    df_point = pd.DataFrame(
        [[point, t, l] for t, l in enumerate(logits_slices_indices)],
        columns = ['points', 'time_steps', 'logits_slices']
    )
    df_combined = pd.concat([df_combined, df_point], ignore_index = True)

logger['df_logits'] = df_combined





"""
Effect of fixation on the state (longer slices)
"""

points = []
logits_slices = []

for i in range(num_trials):
    length_ep, child_dict_ep, points_ep, logits_seq_ep, fixation_seq_ep, decision_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'points', 'logits_seqs', 'fixation_seqs', 'decision_seqs',
    )

    if length_ep < args.t_max and len(fixation_seq_ep) >= 10:
        # process logit
        logits_seq_ep = np.array(logits_seq_ep) # (num_timesteps, num_actions)
        logits_seq_ep = normalize_logits(child_dict_ep, logits_seq_ep)

        # loop slices of 4 time steps
        for t in range(0, len(fixation_seq_ep) - 10):
            node = fixation_seq_ep[t]
            logits_slices.append(logits_seq_ep[t : t + 10, args.num_nodes + node])
            points.append(points_ep[node])
    
points = np.array(points)
logits_slices = np.array(logits_slices)

df_combined = pd.DataFrame()
for i, point in enumerate(env.env.point_set):
    indices = np.where(points == point)[0]
    logits_slices_indices = logits_slices[indices].mean(axis = 0)
    df_point = pd.DataFrame(
        [[point, t, l] for t, l in enumerate(logits_slices_indices)],
        columns = ['points', 'time_steps', 'logits_slices']
    )
    df_combined = pd.concat([df_combined, df_point], ignore_index = True)

logger['df_logits_long'] = df_combined





"""
Effect of fixation on the path
"""


points = []
depths = []
rel_levels = [] # 0 = self, 1 = parent, 2 = grandparent, ...
logits_path = []

for i in range(num_trials):
    length_ep, child_dict_ep, parent_dict_ep, root_node_ep, points_ep, depths_ep, fixation_seq_ep, logits_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'parent_dicts', 'root_nodes', 'points', 'depths', 'fixation_seqs', 'logits_seqs'
    )

    if length_ep < args.t_max and len(fixation_seq_ep) >= 2:
        # get paths (root node not included)
        paths_ep = [[] for _ in range(args.num_nodes)]
        for s in range(env.env.num_nodes):
            pointer = s
            paths_ep[s].append(pointer)
            while pointer in parent_dict_ep.keys():
                pointer = parent_dict_ep[pointer]
                paths_ep[s].append(pointer)
            paths_ep[s].reverse()  # [root, ..., grandparent, parent, self]

        # process logit
        logits_seq_ep = np.array(logits_seq_ep)
        logits_seq_ep = normalize_logits(child_dict_ep, logits_seq_ep)

        # initialize node mask
        node_mask_ep = np.zeros((env.env.num_nodes,))
        node_mask_ep[root_node_ep] = 1

        for t in range(0, len(fixation_seq_ep) - 4):
            node = fixation_seq_ep[t]
            path = paths_ep[node]  # [root, ..., grandparent, parent, self]

            if depths_ep[node] >= 1:
                for rel_level, node_path in enumerate(reversed(path)):
                    # rel_level: 0 = self, 1 = parent, 2 = grandparent, ...
                    if node_path != root_node_ep:
                        points.append(points_ep[node])
                        depths.append(depths_ep[node])
                        rel_levels.append(rel_level)
                        logits_path.append(logits_seq_ep[t: t + 4, args.num_nodes + node_path] - logits_seq_ep[t, args.num_nodes + node_path])

            node_mask_ep[node] = 1


points = np.array(points)
depths = np.array(depths)
rel_levels = np.array(rel_levels)
logits_path = np.array(logits_path)

df_combined = pd.DataFrame()
for j, rel_level in enumerate(range(int(rel_levels.max()) + 1)):
    for k, point in enumerate(env.env.point_set):
        indices = np.where((points == point) & (rel_levels == rel_level))[0]
        if len(indices) == 0:
            continue
        logits_path_mean = logits_path[indices].mean(axis = 0)
        df_rel = pd.DataFrame(
            [[rel_level, point, t, l] for t, l in enumerate(logits_path_mean)],
            columns = ['rel_levels', 'points', 'time_steps', 'logits_path']
        )
        df_combined = pd.concat([df_combined, df_rel], ignore_index = True)

logger['df_logits_path'] = df_combined





"""
Change of mind
"""

depth_1_logits = []
time_steps = []
depths = []
groups = []

for i in range(num_trials):
    length_ep, child_dict_ep, root_node_ep, points_ep, depths_ep, logits_seq_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'root_nodes', 'points', 'depths', 'logits_seqs', 'fixation_seqs'
    )

    if length_ep < args.t_max and len(fixation_seq_ep) > 1:

        # process logit
        logits_seq_ep = np.array(logits_seq_ep) # (num_timesteps, num_actions)
        logits_seq_ep = normalize_logits(child_dict_ep, logits_seq_ep)

        # initialize node mask
        node_mask_ep = np.zeros((env.env.num_nodes,))
        node_mask_ep[root_node_ep] = 1

        # get depth-1 states
        depth_1_node_1, depth_1_node_2 = child_dict_ep[root_node_ep]

        # initialize q values
        depth_1_q_values = [0., 0.]

        for t, node in enumerate(fixation_seq_ep):

            # update node mask
            node_mask_ep[node] = 1

            # get q valules
            q_values = get_q_values(child_dict_ep, points_ep * node_mask_ep)
            depth_1_q_values_updated = [q_values[depth_1_node_1], q_values[depth_1_node_2]]

            # check if there is change of mind
            if depth_1_q_values[0] - depth_1_q_values[1] > 0 and depth_1_q_values_updated[0] - depth_1_q_values_updated[1] < 0:
                for k in range(3):
                    depth_1_logits.append(logits_seq_ep[t + k, env.env.num_nodes + depth_1_node_1])
                    time_steps.append(k)
                    depths.append(depths_ep[node])
                    groups.append(0)
            
            elif depth_1_q_values[0] - depth_1_q_values[1] < 0 and depth_1_q_values_updated[0] - depth_1_q_values_updated[1] > 0:
                for k in range(3):
                    depth_1_logits.append(logits_seq_ep[t + k, env.env.num_nodes + depth_1_node_1])
                    time_steps.append(k)
                    depths.append(depths_ep[node])
                    groups.append(1)

            # update q values
            depth_1_q_values = depth_1_q_values_updated.copy()

depth_1_logits = np.array(depth_1_logits)
time_steps = np.array(time_steps)
groups = np.array(groups)

df = pd.DataFrame({
    'depth_1_logits': depth_1_logits,
    'time_steps': time_steps,
    'depths': depths,
    'groups': groups,
})

df_grouped = df.groupby(['time_steps', 'groups'])['depth_1_logits'].mean().reset_index()
logger['df_logits_change_of_mind'] = df_grouped

df_grouped = df.groupby(['time_steps', 'groups', 'depths'])['depth_1_logits'].mean().reset_index()
logger['df_logits_change_of_mind_grouped'] = df_grouped





"""
Q value
"""

points = []
depths = []
rel_levels = []       # 0 = self, 1 = parent, 2 = grandparent, ...
q_values_path = []

for i in range(num_trials):
    child_dict_ep = data['child_dicts'][i]
    parent_dict_ep = data['parent_dicts'][i]
    root_node_ep = data['root_nodes'][i]
    points_ep = data['points'][i]
    depths_ep = data['depths'][i]
    fixation_seq_ep = data['fixation_seqs'][i]

    # get paths
    paths_ep = [[] for _ in range(args.num_nodes)]
    for s in range(args.num_nodes):
        pointer = s
        paths_ep[s].append(pointer)
        while pointer in parent_dict_ep.keys():
            pointer = parent_dict_ep[pointer]
            paths_ep[s].append(pointer)
        paths_ep[s].reverse()  # [root, ..., grandparent, parent, self]

    node_mask_ep = np.zeros((args.num_nodes,))
    node_mask_ep[root_node_ep] = 1.

    q_values_seq_ep = []
    for node in fixation_seq_ep:
        q_values_t = get_q_values(child_dict_ep, points_ep * node_mask_ep)
        q_values_seq_ep.append(q_values_t)
        node_mask_ep[node] = 1

    q_values_seq_ep = np.array(q_values_seq_ep)

    node_mask_ep = np.zeros((args.num_nodes,))
    node_mask_ep[root_node_ep] = 1

    for t in range(0, len(fixation_seq_ep) - 4):
        node = fixation_seq_ep[t]
        path = paths_ep[node]  # [root, ..., grandparent, parent, self]

        if depths_ep[node] >= 1:
            for rel_level, node_path in enumerate(reversed(path)):
                # rel_level: 0 = self, 1 = parent, 2 = grandparent, ...
                if node_path != root_node_ep:
                    points.append(points_ep[node])
                    depths.append(depths_ep[node])
                    rel_levels.append(rel_level)
                    q_values_path.append(q_values_seq_ep[t: t + 4, node_path] - q_values_seq_ep[t, node_path])

        node_mask_ep[node] = 1


points = np.array(points)
depths = np.array(depths)
rel_levels = np.array(rel_levels)
q_values_path = np.array(q_values_path)

df_combined = pd.DataFrame()
for j, rel_level in enumerate(range(int(rel_levels.max()) + 1)):
    for k, point in enumerate(np.array([-8, -4, -2, -1, 1, 2, 4, 8])):
        indices = np.where((points == point) & (rel_levels == rel_level))[0]
        if len(indices) == 0:
            continue
        q_values_path_mean = q_values_path[indices].mean(axis=0)
        df_rel = pd.DataFrame(
            [[rel_level, point, t, q] for t, q in enumerate(q_values_path_mean)],
            columns=['rel_levels', 'points', 'time_steps', 'q_values_path']
        )
        df_combined = pd.concat([df_combined, df_rel], ignore_index=True)

logger['df_q_values_path'] = df_combined





"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_logit.pkl'), 'wb') as f:
    pickle.dump(logger, f)