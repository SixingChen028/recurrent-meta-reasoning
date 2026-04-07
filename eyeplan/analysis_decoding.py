import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
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
data = load_data(os.path.join(exp_path, f'data_simulation_decoding.p'))
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
Fit point
"""

# collect data
hiddens = [[] for _ in range(args.num_nodes)]
points = [[] for _ in range(args.num_nodes)]

for i in range(num_trials):
    length_ep, child_dict_ep, points_ep, cum_points_ep, root_node_ep, fixation_seq_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'points', 'cum_points', 'root_nodes', 'fixation_seqs', 'hidden_seqs'
    )

    if length_ep < args.t_max:
        # initialize mask
        node_mask_ep = np.zeros(args.num_nodes)
        node_mask_ep[root_node_ep] = 1

        for t, node in enumerate(fixation_seq_ep):

            # if not fixated
            if node_mask_ep[node] == 0:
                # update mask
                node_mask_ep[node] = 1

            for d in range(args.num_nodes):
                # if fixated
                if node_mask_ep[d] == 1:
                    hiddens[d].append(hidden_seq_ep[t + 1])
                    points[d].append(points_ep[d])

# convert tensors into np.array
hiddens = [np.stack(_) for _ in hiddens]
points = [np.stack(_) for _ in points]

# initialize weights recording
weights_point = {
    'coefs': [],
    'intercepts': [],
}
r2s_point = []
mses_point = []
r2s_point_cv = []  # for storing CV scores
mses_point_cv = []  # for storing CV mse

# 5-fold cross-validation setup
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

for node in range(args.num_nodes):
    # initialize lists for this node's CV scores
    node_r2_cv = []
    node_mse_cv = []

    # perform 5-fold CV
    for train_idx, val_idx in kf.split(hiddens[node]):
        X_train, X_val = hiddens[node][train_idx], hiddens[node][val_idx]
        y_train, y_val = points[node][train_idx], points[node][val_idx]
        
        # train model on training fold
        model_cv = LinearRegression()
        model_cv.fit(X_train, y_train)
        
        # predict on validation fold
        y_val_pred = model_cv.predict(X_val)
        
        # calculate metrics for this fold
        node_r2_cv.append(r2_score(y_val, y_val_pred))
        node_mse_cv.append(mean_squared_error(y_val, y_val_pred))

    # store average CV scores for this node
    r2s_point_cv.append(np.mean(node_r2_cv))
    mses_point_cv.append(np.mean(node_mse_cv))
    
    # fit final model on all data for this node (for later use in error slices)
    model = LinearRegression()
    model.fit(hiddens[node], points[node])
    
    # predict on full data (for comparison with CV results)
    points_node_pred = model.predict(hiddens[node])
    
    # append full-data metrics
    r2s_point.append(r2_score(points[node], points_node_pred))
    mses_point.append(mean_squared_error(points[node], points_node_pred))
    
    # append weights from final model
    weights_point['coefs'].append(model.coef_)
    weights_point['intercepts'].append(model.intercept_)


# compute error slices
t_window = 7
model = LinearRegression()

point_error_slices = []
point_error_slices_baseline = []

y_true_by_k = [[] for _ in range(t_window)]
y_pred_by_k = [[] for _ in range(t_window)]
y_true_baseline_by_k = [[] for _ in range(t_window)]
y_pred_baseline_by_k = [[] for _ in range(t_window)]

# loop through episodes
for i in range(num_trials):
    length_ep, points_ep, root_node_ep, fixation_seq_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'points', 'root_nodes', 'fixation_seqs', 'hidden_seqs'
    )

    # fixate at least 2 times
    if length_ep < args.t_max and len(fixation_seq_ep) >= t_window:

        # initialize mask
        node_mask_ep = np.zeros(args.num_nodes)
        node_mask_ep[root_node_ep] = 1

        for t, node in enumerate(fixation_seq_ep[0:-t_window]):

            # if node fixated
            if node_mask_ep[node] == 0:

                # update mask
                node_mask_ep[node] = 1

                # load weights
                model.coef_ = weights_point['coefs'][node]
                model.intercept_ = weights_point['intercepts'][node]

                # predict
                point_pred_slice = model.predict(hidden_seq_ep[t : t + t_window]) # (t_window,)
                error_slice = np.abs(points_ep[node] - point_pred_slice)
                error_slice_baseline = np.abs(np.random.choice(points_ep) - point_pred_slice)

                # append
                point_error_slices.append(error_slice)
                point_error_slices_baseline.append(error_slice_baseline)

                true_point = points_ep[node] # scalar target for this node
                base_point = np.random.choice(points_ep)

                # fan-out by offset k
                for k in range(t_window):
                    y_true_by_k[k].append(true_point)
                    y_pred_by_k[k].append(point_pred_slice[k])

                    y_true_baseline_by_k[k].append(base_point)
                    y_pred_baseline_by_k[k].append(point_pred_slice[k])

point_error_slices = np.nanmean(point_error_slices, axis = 0)
point_error_slices_baseline = np.nanmean(point_error_slices_baseline, axis = 0)

point_r2_slices = np.array([r2_score(y_true_by_k[k], y_pred_by_k[k]) for k in range(t_window)])
point_r2_slices_baseline = np.array([r2_score(y_true_baseline_by_k[k], y_pred_baseline_by_k[k]) for k in range(t_window)])

np.save(os.path.join(exp_path, 'weights_point.npy'), weights_point, allow_pickle = True)
logger['point_error_slices'] = point_error_slices
logger['point_error_slices_baseline'] = point_error_slices_baseline
logger['point_r2_slices'] = point_r2_slices
logger['point_r2_slices_baseline'] = point_r2_slices_baseline
logger['r2s_point'] = r2s_point
logger['mses_point'] = mses_point
logger['r2s_point_cv'] = r2s_point_cv # average R2 from 5-fold CV
logger['mses_point_cv'] = mses_point_cv # average MSE from 5-fold CV





"""
Fit cumulative point
"""

hiddens = [[] for _ in range(args.num_nodes)]
cum_points = [[] for _ in range(args.num_nodes)]

for i in range(num_trials):
    length_ep, child_dict_ep, parent_dict_ep, points_ep, cum_points_ep, root_node_ep, fixation_seq_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'parent_dicts', 'points', 'cum_points', 'root_nodes', 'fixation_seqs', 'hidden_seqs'
    )

    if length_ep < args.t_max:
        # initialize mask
        node_mask_ep = np.zeros(args.num_nodes)
        node_mask_ep[root_node_ep] = 1

        for t, node in enumerate(fixation_seq_ep):

            # if not fixated
            if node_mask_ep[node] == 0:
                # update mask
                node_mask_ep[node] = 1
            
            for d in range(args.num_nodes):
                # not root & not fixated & parent fixated
                if d != root_node_ep and node_mask_ep[d] == 0 and node_mask_ep[parent_dict_ep[d]] == 1:
                    hiddens[d].append(hidden_seq_ep[t + 1])
                    cum_points[d].append(cum_points_ep[d] - points_ep[d]) # cumulative point leading to the node


# convert tensors into np.array
hiddens = [np.stack(_) for _ in hiddens]
cum_points = [np.stack(_) for _ in cum_points]

# initialize weights recording
weights_cum_point = {
    'coefs': [],
    'intercepts': [],
}
r2s_cum_point = []
mses_cum_point = []
r2s_cum_point_cv = [] # for storing CV scores
mses_cum_point_cv = [] # for storing CV mse

# 5-fold cross-validation setup
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

for node in range(args.num_nodes):
    # initialize lists for this node's CV scores
    node_r2_cv = []
    node_mse_cv = []

    # perform 5-fold CV
    for train_idx, val_idx in kf.split(hiddens[node]):
        X_train, X_val = hiddens[node][train_idx], hiddens[node][val_idx]
        y_train, y_val = cum_points[node][train_idx], cum_points[node][val_idx]
        
        # train model on training fold
        model_cv = LinearRegression()
        model_cv.fit(X_train, y_train)
        
        # predict on validation fold
        y_val_pred = model_cv.predict(X_val)
        
        # calculate metrics for this fold
        node_r2_cv.append(r2_score(y_val, y_val_pred))
        node_mse_cv.append(mean_squared_error(y_val, y_val_pred))
    
    # store average CV scores for this node
    r2s_cum_point_cv.append(np.mean(node_r2_cv))
    mses_cum_point_cv.append(np.mean(node_mse_cv))
    
    # fit final model on all data for this node
    model = LinearRegression()
    model.fit(hiddens[node], cum_points[node])

    # predict on full data (for comparison with CV results)
    cum_points_node_pred = model.predict(hiddens[node])

    # append full-data metrics
    r2s_cum_point.append(r2_score(cum_points[node], cum_points_node_pred))
    mses_cum_point.append(mean_squared_error(cum_points[node], cum_points_node_pred))

    # append weights from final model
    weights_cum_point['coefs'].append(model.coef_)
    weights_cum_point['intercepts'].append(model.intercept_)

np.save(os.path.join(exp_path, 'weights_cum_point.npy'), weights_cum_point, allow_pickle = True)
logger['r2s_cum_point'] = r2s_cum_point
logger['mses_cum_point'] = mses_cum_point
logger['r2s_cum_point_cv'] = r2s_cum_point_cv # average R2 from 5-fold CV
logger['mses_cum_point_cv'] = mses_cum_point_cv # average MSE from 5-fold CV





"""
Fit Q value
"""

hiddens = [[] for _ in range(args.num_nodes)]
q_values = [[] for _ in range(args.num_nodes)]

for i in range(num_trials):
    length_ep, child_dict_ep, points_ep, cum_points_ep, root_node_ep, fixation_seq_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'child_dicts', 'points', 'cum_points', 'root_nodes', 'fixation_seqs', 'hidden_seqs'
    )

    if length_ep < args.t_max:
        # initialize mask
        node_mask_ep = np.zeros(args.num_nodes)
        node_mask_ep[root_node_ep] = 1

        for t, node in enumerate(fixation_seq_ep):

            # if not fixated
            if node_mask_ep[node] == 0:
                # update mask
                node_mask_ep[node] = 1

            for d in range(args.num_nodes):
                # if fixated
                if node_mask_ep[d] == 1:
                    hiddens[d].append(hidden_seq_ep[t + 1])
                    q_values[d].append(get_q_values(child_dict_ep, points_ep * node_mask_ep)[d])

# convert tensors into np.array
hiddens = [np.stack(_) for _ in hiddens]
q_values = [np.stack(_) for _ in q_values]

# initialize weights recording
weights_q_value = {
    'coefs': [],
    'intercepts': [],
}
r2s_q_value = []
mses_q_value = []
r2s_q_value_cv = [] # for storing CV scores
mses_q_value_cv = [] # for storing CV mse

# 5-fold cross-validation setup
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

for node in range(args.num_nodes):
    # initialize lists for this node's CV scores
    node_r2_cv = []
    node_mse_cv = []

    # perform 5-fold CV
    for train_idx, val_idx in kf.split(hiddens[node]):
        X_train, X_val = hiddens[node][train_idx], hiddens[node][val_idx]
        y_train, y_val = q_values[node][train_idx], q_values[node][val_idx]
        
        # train model on training fold
        model_cv = LinearRegression()
        model_cv.fit(X_train, y_train)
        
        # predict on validation fold
        y_val_pred = model_cv.predict(X_val)
        
        # calculate metrics for this fold
        node_r2_cv.append(r2_score(y_val, y_val_pred))
        node_mse_cv.append(mean_squared_error(y_val, y_val_pred))
    
    # store average CV scores for this node
    r2s_q_value_cv.append(np.mean(node_r2_cv))
    mses_q_value_cv.append(np.mean(node_mse_cv))
    
    # fit final model on all data for this node
    model = LinearRegression()
    model.fit(hiddens[node], q_values[node])

    # predict on full data (for comparison with CV results)
    q_values_node_pred = model.predict(hiddens[node])

    # append full-data metrics
    r2s_q_value.append(r2_score(q_values[node], q_values_node_pred))
    mses_q_value.append(mean_squared_error(q_values[node], q_values_node_pred))

    # append weights from final model
    weights_q_value['coefs'].append(model.coef_)
    weights_q_value['intercepts'].append(model.intercept_)

np.save(os.path.join(exp_path, 'weights_q_value.npy'), weights_q_value, allow_pickle = True)
logger['r2s_q_value'] = r2s_q_value
logger['mses_q_value'] = mses_q_value
logger['r2s_q_value_cv'] = r2s_q_value_cv # average R2 from 5-fold CV
logger['mses_q_value_cv'] = mses_q_value_cv # average MSE from 5-fold CV





"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_decoding.pkl'), 'wb') as f:
    pickle.dump(logger, f)