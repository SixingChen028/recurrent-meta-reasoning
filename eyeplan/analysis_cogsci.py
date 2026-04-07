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
Performance analysis
"""

empirical_cum_points = []
max_cum_points = []
abortions = []

for i in range(num_trials):
    length_ep, cum_points_ep, leaf_nodes_ep, decision_seq_ep = pull(
        data, i, 'lengths', 'cum_points', 'leaf_nodes', 'decision_seqs'
    )

    if length_ep < args.t_max:
        # append the cumulative point of the last decision
        empirical_cum_points.append(cum_points_ep[decision_seq_ep[-1]])

        # append the maximum cumulative point
        max_cum_points.append(max([cum_points_ep[_] for _ in leaf_nodes_ep]))
    
    # append abortion
    abortions.append(length_ep == args.t_max)

logger['cum_point_proportion'] = np.mean(empirical_cum_points) / np.mean(max_cum_points)
logger['optimal_path_proportion'] = np.mean(np.array(max_cum_points) == np.array(empirical_cum_points))
logger['abortion_proportion'] = np.mean(abortions)





"""
Refixation analysis
"""

is_refixations = []

for i in range(num_trials):
    length_ep, root_node_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'root_nodes', 'fixation_seqs'
    )

    # initialize node mask
    node_mask_ep = np.zeros((args.num_nodes,))
    node_mask_ep[root_node_ep] = 1

    if length_ep < args.t_max:
        for node in fixation_seq_ep:
            # append if refixaiton
            is_refixations.append(node_mask_ep[node])

            # update node mask
            node_mask_ep[node] = 1

logger['refixation_proportion'] = np.mean(is_refixations)





"""
Action statistics
"""

fixation_counts = []
decision_counts = []

for i in range(num_trials):
    length_ep, fixation_seq_ep, decision_seq_ep = pull(
        data, i, 'lengths', 'fixation_seqs', 'decision_seqs'
    )

    if length_ep < args.t_max:
        # append decision and fixation count
        fixation_counts.append(len(fixation_seq_ep))
        decision_counts.append(len(decision_seq_ep))

logger['fixation_count'] = np.mean(fixation_counts)
logger['decision_count'] = np.mean(decision_counts)





"""
Fixation by depth
"""
max_depth = 5
fixation_counts_by_depth = [[] for _ in range(max_depth + 1)]

for i in range(num_trials):
    length_ep, depths_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'depths', 'fixation_seqs'
    )

    if length_ep < args.t_max:
        # initialize fixation count at different depths
        fixation_counts_by_depth_ep = np.zeros(max(depths_ep) + 1)

        # count fixations in the episode
        for node in fixation_seq_ep:
            fixation_counts_by_depth_ep[depths_ep[node]] += 1
        
        # compute number of nodes at each depth
        num_nodes_by_depth_ep = np.unique(depths_ep, return_counts = True)[1]

        # average counts for each depth
        for j in range(len(fixation_counts_by_depth_ep)):
            fixation_counts_by_depth[j].append(fixation_counts_by_depth_ep[j] / num_nodes_by_depth_ep[j])

logger['fixation_counts_by_depth'] = np.array([np.mean(_) for _ in fixation_counts_by_depth])





"""
Fixation by type
"""

fixation_counts_by_type = {'child': 0, 'parent': 0, 'sibling': 0, 'others': 0, 'self': 0}

for i in range(num_trials):
    length_ep, child_dict_ep, root_node_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'root_nodes', 'fixation_seqs'
    )
    # add starting root node
    fixation_seq_ep_inserted = fixation_seq_ep.copy()
    fixation_seq_ep_inserted.insert(0, root_node_ep)

    if length_ep < args.t_max:
        for node, node_next in list(zip(fixation_seq_ep_inserted, fixation_seq_ep_inserted[1:])):
            # count fixations by relationship
            fixation_counts_by_type[relationship(child_dict_ep, node, node_next)] += 1

# compute proportions
fixation_proportions_by_type = np.array(list(fixation_counts_by_type.values())) / np.sum(fixation_counts)

logger['fixation_proportions_by_type'] = fixation_proportions_by_type





"""
Continuation policy
"""

points = []
depths = []
continuations = []

for i in range(num_trials):
    length_ep, child_dict_ep, root_node_ep, points_ep, depths_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'root_nodes', 'points', 'depths', 'fixation_seqs'
    )
    # add starting root node
    fixation_seq_ep_inserted = fixation_seq_ep.copy()
    fixation_seq_ep_inserted.insert(0, root_node_ep)

    if length_ep < args.t_max:
        for node, node_next in list(zip(fixation_seq_ep_inserted, fixation_seq_ep_inserted[1:])):
            # if node is not root
            if node in child_dict_ep.keys():
                points.append(points_ep[node])
                depths.append(depths_ep[node])
                continuations.append(int(node_next in child_dict_ep[node]))

df = pd.DataFrame({'points': points, 'depths': depths, 'continuations': continuations})
df['jobid'] = args.jobid
logger['df_continuation'] = df

df_grouped = df[df['points'] != 0].groupby('points')['continuations'].mean().reset_index()
model = LinearRegression()
model.fit(df_grouped[['points']].values, df_grouped['continuations'].values)
logger['par_continuation_by_point'] = np.array([model.coef_[0], model.intercept_])

df_grouped = df.groupby('depths')['continuations'].mean().reset_index()
model = LinearRegression()
model.fit(df_grouped[['depths']].values, df_grouped['continuations'].values)
logger['par_continuation_by_depth'] = np.array([model.coef_[0], model.intercept_])





"""
Exploitation policy
"""

child1_fixation_counts = []
relative_q_values = []
q_groups = []

for i in range(num_trials):
    length_ep, child_dict_ep, root_node_ep, points_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'root_nodes', 'points', 'fixation_seqs'
    )

    # initialize node mask
    node_mask_ep = np.zeros((args.num_nodes,)) # root node does not matter here
    node_mask_ep[root_node_ep] = 1.

    if length_ep < args.t_max:
        for node, node_next in list(zip(fixation_seq_ep, fixation_seq_ep[1:])):
            # update node mask
            node_mask_ep[node] = 1

            # if current node has children and next node is a child of current node
            if relationship(child_dict_ep, node, node_next) == 'child':
                # get children
                children = child_dict_ep[node]
                child1, child2 = children[1], children[0]

                # compute action values
                q_values_ep = get_q_values(child_dict_ep, points_ep * node_mask_ep)

                # append if the next fixation is on the child
                child1_fixation_counts.append(int(node_next == child1))

                # append relative action value
                relative_q_values.append(q_values_ep[child1] - q_values_ep[child2])

                # append group
                if node_mask_ep[child1] == 0 and node_mask_ep[child2] == 0:
                    q_groups.append('none')
                elif node_mask_ep[child1] == 1 and node_mask_ep[child2] == 0:
                    q_groups.append('child 1')
                elif node_mask_ep[child1] == 0 and node_mask_ep[child2] == 1:
                    q_groups.append('child 2')
                elif node_mask_ep[child1] == 1 and node_mask_ep[child2] == 1:
                    q_groups.append('both')
                    
df = pd.DataFrame({
    'child1_fixation_counts': child1_fixation_counts,
    'relative_q_values': relative_q_values,
    'q_groups': q_groups,
})
df['jobid'] = args.jobid
logger['df_exploitation'] = df

model = LinearRegression()
model.fit(df[['relative_q_values']].values, df['child1_fixation_counts'].values)
logger['par_exploitation'] = np.array([model.coef_[0], model.intercept_])

pars = []
for i, group in enumerate(['child 1', 'child 2', 'both']):
    df_filtered = df[df['q_groups'] == group]
    model = LinearRegression()
    model.fit(df_filtered[['relative_q_values']].values, df_filtered['child1_fixation_counts'].values)
    pars.append(np.array([model.coef_[0], model.intercept_]))
logger['par_exploitation_by_type'] = np.array(pars)





"""
Exploration policy
"""

child1_fixation_counts = []
relative_fixation_counts = []
seen_groups = []
idx = 1 # pick a side

for i in range(num_trials):
    length_ep, child_dict_ep, points_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'points', 'fixation_seqs'
    )

    # initialize count
    fixation_counts_ep = np.zeros(args.num_nodes, dtype = int)

    if length_ep < args.t_max:
        for node, node_next in list(zip(fixation_seq_ep, fixation_seq_ep[1:])):
            # update node count
            fixation_counts_ep[node] += 1

            # if current node has children and next node is a child of current node
            if node in child_dict_ep.keys() and node_next in child_dict_ep[node]:
                # get children
                children = child_dict_ep[node]
                child1, child2 = children[1], children[0]

                # append if the next fixation is on the child
                child1_fixation_counts.append(int(node_next == child1))

                # append relative fixation count
                relative_fixation_counts.append(fixation_counts_ep[child1] - fixation_counts_ep[child2])

                # append group
                if fixation_counts_ep[child1] > 0 and fixation_counts_ep[child2] == 0:
                    seen_groups.append('first')
                elif fixation_counts_ep[child2] > 0 and fixation_counts_ep[child1] == 0:
                    seen_groups.append('second')
                elif fixation_counts_ep[child1] > 0 and fixation_counts_ep[child2] > 0:
                    seen_groups.append('both')
                elif fixation_counts_ep[child1] == 0 and fixation_counts_ep[child2] == 0:
                    seen_groups.append('neither')

df = pd.DataFrame({
    'child1_fixation_counts': child1_fixation_counts,
    'relative_fixation_counts': relative_fixation_counts,
    'seen_groups': seen_groups,
})
df['jobid'] = args.jobid
logger['df_exploration'] = df





"""
Switching policy
"""

jump_depths = []
jump_depths_baseline = []
jump_seens = []

for i in range(num_trials):
    length_ep, child_dict_ep, parent_dict_ep, depths_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'parent_dicts', 'depths', 'fixation_seqs'
    )

    # initialize node mask
    node_mask_ep = np.zeros((args.num_nodes,))

    if length_ep < args.t_max:
        for node, node_next in list(zip(fixation_seq_ep, fixation_seq_ep[1:])):
            # update node mask
            node_mask_ep[node] = 1

            # if next node is a jump
            # if relationship(child_dict_ep, node, node_next) == 'others':
            if relationship(child_dict_ep, node, node_next) != 'child': # non-child
                # append jump depth
                jump_depths.append(depths_ep[node_next])

                # append seen groups
                if node_mask_ep[node_next] == 0:
                    jump_seens.append('unseen')
                elif node_mask_ep[node_next] == 1:
                    jump_seens.append('seen')

                # initialize non-child nodes
                jump_nodes_ep = np.arange(args.num_nodes)

                # exclude the node itself
                jump_nodes_ep = jump_nodes_ep[jump_nodes_ep != node]

                # exclude children if it has children
                if node in child_dict_ep.keys():
                    children = child_dict_ep[node]
                    jump_nodes_ep = jump_nodes_ep[(jump_nodes_ep != children[0]) & (jump_nodes_ep != children[1])]

                # randomly choose a jump node
                jump_node_random = random.choice(jump_nodes_ep)
                jump_depths_baseline.append(depths_ep[jump_node_random])

df = pd.DataFrame({
    'jump_depths': jump_depths,
    'jump_depths_baseline': jump_depths_baseline,
    'jump_seens': jump_seens,
})
df['jobid'] = args.jobid
logger['df_jump'] = df





"""
Evidence accumulation
"""

chosens = []
points = []
fixation_counts_chosen = []

for i in range(num_trials):
    length_ep, points_ep, root_node_ep, fixation_seq_ep, decision_seq_ep = pull(
        data, i, 'lengths', 'points', 'root_nodes', 'fixation_seqs', 'decision_seqs'
    )

    if length_ep < args.t_max:
        for node in range(args.num_nodes):
            # if not root node
            if node != root_node_ep:
                # append if the node is chosen
                chosens.append(int(node in decision_seq_ep))

                # append point in the node
                points.append(points_ep[node])

                # append fixation count of the node
                fixation_counts_chosen.append(np.sum(fixation_seq_ep.count(node)))

# fixation_clip = 3
df = pd.DataFrame({
    'chosens': chosens,
    'points': points,
    'fixation_counts_chosen': fixation_counts_chosen
})
df['jobid'] = args.jobid
logger['df_evidence_accumulation'] = df





"""
Frontier fixation
"""

cum_points = []
frontier_fixation_counts = []

for i in range(num_trials):
    length_ep, child_dict_ep, root_node_ep, points_ep, cum_points_ep, depths_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'root_nodes', 'points', 'cum_points', 'depths', 'fixation_seqs'
    )

    # initialize frontier nodes
    frontier = np.array(child_dict_ep[root_node_ep])
    
    if length_ep < args.t_max:
        for node in fixation_seq_ep:
            # if the fixation is in frontier
            if node in frontier:
                # if both frontier nodes have been explored
                if (child_dict_ep[root_node_ep][0] not in frontier and child_dict_ep[root_node_ep][1] not in frontier):
                    # loop through frontier
                    for frontier_node in frontier:
                        # append cumulative point *leading to that node*
                        cum_points.append(cum_points_ep[frontier_node] - points_ep[frontier_node])

                        # append if the frontier is the fixated node
                        frontier_fixation_counts.append(int(frontier_node == node))
                
                # update frontier
                frontier = np.delete(frontier, np.where(frontier == node)[0])
                if node in child_dict_ep.keys():
                    frontier = np.append(frontier, child_dict_ep[node])

df = pd.DataFrame({
    'cum_points': cum_points,
    'frontier_fixation_counts': frontier_fixation_counts,
})
df['jobid'] = args.jobid
logger['df_frontier'] = df

df_filtered = df[df['cum_points'] != 0]
model = LinearRegression()
model.fit(df_filtered[['cum_points']].values, df_filtered['frontier_fixation_counts'].values)
logger['par_frontier'] = np.array([model.coef_[0], model.intercept_])








"""
Refixation Q value
"""

q_values = []
refixation_counts = []

for i in range(num_trials):
    length_ep, child_dict_ep, root_node_ep, points_ep, fixation_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'root_nodes', 'points', 'fixation_seqs'
    )

    # initialize node mask
    node_mask_ep = np.zeros((11))
    node_mask_ep[root_node_ep] = 1.

    if length_ep < args.t_max:
        for node in fixation_seq_ep:
            # if the fixation is a refixation (already visited)
            if node_mask_ep[node] == 1:
                # compute q values using only revealed points
                q_values_ep = get_q_values(child_dict_ep, points_ep * node_mask_ep)

                # loop through all previously visited nodes
                visited_nodes = np.where(node_mask_ep == 1)[0]
                for visited_node in visited_nodes:
                    # append q value of this visited node
                    q_values.append(q_values_ep[visited_node])

                    # append whether this visited node was the one refixated
                    refixation_counts.append(int(visited_node == node))

            # update node mask after the check
            node_mask_ep[node] = 1.

df = pd.DataFrame({
    'q_values': q_values,
    'refixation_counts': refixation_counts,
})
df = df[(df['q_values'] != 0)]
df['jobid'] = args.jobid
logger['df_refixation_q'] = df

model = LinearRegression()
model.fit(df[['q_values']].values, df['refixation_counts'].values)
logger['par_refixation_q'] = np.array([model.coef_[0], model.intercept_])










"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_cog.pkl'), 'wb') as f:
    pickle.dump(logger, f)