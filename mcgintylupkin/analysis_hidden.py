import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.linalg import svd
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
exp_path = os.path.join(args.path, f'exp_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.jobid}')

# # set printing path
# sys.stdout = open(os.path.join(exp_path, 'output_cog.txt'), 'w')

# initialize logger
logger = {}

# load net
net = torch.load(os.path.join(exp_path, f'net.pth'), weights_only = False)

# set environment
env = MetaLearningWrapper(
    BanditEnv(
        num_bandits = args.num_bandits,
        value_min = args.value_min,
        value_max = args.value_max,
        value_mean = args.value_mean,
        value_std = args.value_std,
        reward_std = args.reward_std,
        noise_free_obs = args.noise_free_obs,
        t_max = args.t_max,
        stay_cost = args.stay_cost,
        switch_cost = args.switch_cost,
        scale_factor = args.scale_factor,
    )
)





"""
Preprocessing
"""

# load data
data = load_data(os.path.join(exp_path, f'data_simulation.p'))
data = preprocess(data, args, merge_fixations = False)
num_trials = len(data['values'])
print('Keys:', data.keys())
print(num_trials)





"""
Set logger path
"""

# set experiment path
exp_path = os.path.join(args.path, f'logger_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.jobid}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Utils
"""

def fit_lasso_decoder(X_train, y_train):
    """
    Fit LASSO model with 10-fold cross-validation to select lambda.
    Returns fitted model.
    """

    # use LassoCV for automatic lambda selection via cross-validation
    # alphas: regularization strengths to try (lambda in the paper)
    alphas = np.logspace(-3, 1, 100)
    
    lasso_cv = LassoCV(
        alphas = alphas,
        cv = 10, # 10-fold cross-validation as specified in paper
        max_iter = 10000,
        random_state = 42,
        n_jobs = -1
    )
    
    lasso_cv.fit(X_train, y_train)
    
    return lasso_cv


def compute_value_weights_for_session(X_train, y1_train, y2_train):
    """
    Compute beta1 and beta2 (LASSO weights) for a single session.
    
    Returns:
        beta1, beta2: weight vectors
    """

    # fit LASSO for 1st value
    lasso1 = fit_lasso_decoder(X_train, y1_train)
    beta1 = lasso1.coef_
    
    # fit LASSO for 2nd value
    lasso2 = fit_lasso_decoder(X_train, y2_train)
    beta2 = lasso2.coef_
    
    return beta1, beta2


def find_value_subspace_basis(beta1, beta2, X):
    """
    Find orthonormal basis for value subspace using Semedo et al. method.
    
    Parameters:
        beta1: weights from LASSO model for 1st value (n_cells,)
        beta2: weights from LASSO model for 2nd value (n_cells,)
        X: spike count matrix (n_trials, n_cells)
    
    Returns:
        Q: orthonormal basis for value subspace (n_cells, 2)
    """
    
    # concatenate weight vectors into n x 2 matrix B
    B = np.column_stack([beta1, beta2]) # (n_cells, 2)
    
    # compute covariance matrix of spike data
    cov_X = np.cov(X.T)
    
    # compute M = B^T × cov(X)
    M = B.T @ cov_X
    
    # SVD of M
    U, D, Vt = svd(M, full_matrices=False)
    V = Vt.T
    
    # take first 2 columns of V (corresponding to non-zero singular values)
    Q = V[:, :2] # (n_cells, 2)
    
    return Q





"""
Collect data
"""

# collect data
hiddens = [[] for _ in range(5)]
values = []
indices = []
first_fixations = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions', 'hidden_seqs'
    )

    if length_ep > 1 and length_ep < args.t_max:

        if True: #fixation_seq_ep[0] == 0:

            for t in range(5):
                hiddens[t].append(hidden_seq_ep[t])
            first_fixation = fixation_seq_ep[0]
            values.append(np.array([values_ep[first_fixation], values_ep[1 - first_fixation]]))
            indices.append(i)
            first_fixations.append(fixation_seq_ep[0])

for t in range(5):
    hiddens[t] = np.stack(hiddens[t])
values = np.stack(values)
indices = np.array(indices)
first_fixations = np.array(first_fixations)

print(hiddens[0].shape)
print(values.shape)





"""
Rotation analysis
"""

# get value space
first_val = values[:, 0]
second_val = values[:, 1]

X = hiddens[4]

train_idx = np.arange(len(indices))[:int(len(indices) / 2)]
test_idx = np.arange(len(indices))[int(len(indices) / 2):]

# make datasets
X_train, X_test = X[train_idx], X[test_idx]
y1_train, y1_test = first_val[train_idx], first_val[test_idx]
y2_train, y2_test = second_val[train_idx], second_val[test_idx]
first_fixations_train, first_fixations_test = first_fixations[train_idx], first_fixations[test_idx]

# compute betas
beta1, beta2 = compute_value_weights_for_session(
    X_train, y1_train, y2_train
)

# find value subspace basis using Semedo method
Q = find_value_subspace_basis(beta1, beta2, X_train)


# sequentially prefect data onto value space
enc1_projections = []
enc2_projections = []

for t in range(5):
    X = hiddens[t]

    # make datasets
    X_train, X_test = X[train_idx], X[test_idx]
    y1_train, y1_test = first_val[train_idx], first_val[test_idx]
    y2_train, y2_test = second_val[train_idx], second_val[test_idx]

    # compute betas
    beta1_t, beta2_t = compute_value_weights_for_session(
        X_train, y1_train, y2_train
    )

    # === NEW: covariance-weighted ("encoding") directions ===
    # cov across units (H,H)
    cov_X_t = np.cov(X_train.T)

    enc1_t = cov_X_t @ beta1_t   # (H,)
    enc2_t = cov_X_t @ beta2_t   # (H,)

    # project encoding directions into the (fixed) value subspace Q
    enc1_t_proj = enc1_t @ Q # (2,)
    enc2_t_proj = enc2_t @ Q # (2,)

    enc1_projections.append(enc1_t_proj)
    enc2_projections.append(enc2_t_proj)

enc1_projections = np.array(enc1_projections)
enc2_projections = np.array(enc2_projections)

logger['enc1_projections'] = enc1_projections
logger['enc2_projections'] = enc2_projections





"""
Signal-noise ratio
"""

# loop through time windows
within_vars = []
between_vars_val1 = []
between_vars_val2 = []

for t in range(5):
    X = hiddens[t]

    # make datasets
    X_train, X_test = X[train_idx], X[test_idx]
    y1_train, y1_test = first_val[train_idx], first_val[test_idx]
    y2_train, y2_test = second_val[train_idx], second_val[test_idx]

    # project test data into value subspace
    X_projected = X_test @ Q  # (n_test_trials, 2)

    # variance within each (val1, val2) pair
    within_pair_vars = []
    within_pair_sizes = []
        
    for v1 in [1, 2, 3, 4, 5]:
        for v2 in [1, 2, 3, 4, 5]:
            mask = (y1_test == v1) & (y2_test == v2)
            
            if np.sum(mask) > 1:
                cluster_points = X_projected[mask]
                var_total = np.var(cluster_points[:, 0]) + np.var(cluster_points[:, 1])
                within_pair_vars.append(var_total)
                within_pair_sizes.append(np.sum(mask))
    
    within_var = np.average(within_pair_vars, weights = within_pair_sizes)
    within_vars.append(within_var)
    
    # variance of cluster centers within each val1 group
    between_vals_val1 = []
    
    for v2 in [1, 2, 3, 4, 5]:
        cluster_means = []
        for v1 in [1, 2, 3, 4, 5]:
            mask = (y1_test == v1) & (y2_test == v2)
            if np.sum(mask) > 0:
                cluster_mean = X_projected[mask].mean(axis = 0)
                cluster_means.append(cluster_mean)
        
        if len(cluster_means) > 1:
            cluster_means = np.array(cluster_means)
            between_var = np.var(cluster_means[:, 0]) + np.var(cluster_means[:, 1])
            between_vals_val1.append(between_var)
    
    between_vars_val1.append(np.mean(between_vals_val1))
    
    # variance of cluster centers within each val2 group
    between_vals_val2 = []
    
    for v1 in [1, 2, 3, 4, 5]:
        cluster_means = []
        for v2 in [1, 2, 3, 4, 5]:
            mask = (y1_test == v1) & (y2_test == v2)
            if np.sum(mask) > 0:
                cluster_mean = X_projected[mask].mean(axis = 0)
                cluster_means.append(cluster_mean)
        
        if len(cluster_means) > 1:
            cluster_means = np.array(cluster_means)
            between_var = np.var(cluster_means[:, 0]) + np.var(cluster_means[:, 1])
            between_vals_val2.append(between_var)
    
    between_vars_val2.append(np.mean(between_vals_val2))

# compute SNR for this session
within_vars = np.array(within_vars)
between_vars_val1 = np.array(between_vars_val1)
between_vars_val2 = np.array(between_vars_val2)

snr_val1 = between_vars_val1 / (within_vars + 1e-10) # add small constant to avoid division by zero
snr_val2 = between_vars_val2 / (within_vars + 1e-10)

logger['snr_val1'] = snr_val1
logger['snr_val2'] = snr_val2





"""
Decoding
"""

r2s = []
for t in range(5):

    X = hiddens[t]

    # make datasets
    X_train, X_test = X[train_idx], X[test_idx]
    y1_train, y1_test = first_val[train_idx], first_val[test_idx]
    y2_train, y2_test = second_val[train_idx], second_val[test_idx]

    model1 = fit_lasso_decoder(X_train, y1_train)
    model2 = fit_lasso_decoder(X_train, y2_train)

    y1_pred = model1.predict(X_test)
    y2_pred = model2.predict(X_test)

    r2s.append(np.array([r2_score(y1_test, y1_pred), r2_score(y2_test, y2_pred)]))

r2s = np.array(r2s)
logger['r2s'] = r2s

r2s_2d = []
for t in range(5):

    X = hiddens[t] @ Q

    # make datasets
    X_train, X_test = X[train_idx], X[test_idx]
    y1_train, y1_test = first_val[train_idx], first_val[test_idx]
    y2_train, y2_test = second_val[train_idx], second_val[test_idx]

    model1 = fit_lasso_decoder(X_train, y1_train)
    model2 = fit_lasso_decoder(X_train, y2_train)

    y1_pred = model1.predict(X_test)
    y2_pred = model2.predict(X_test)

    r2s_2d.append(np.array([r2_score(y1_test, y1_pred), r2_score(y2_test, y2_pred)]))

r2s_2s = np.array(r2s_2d)
logger['r2s_2d'] = r2s_2d





"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_hidden.pkl'), 'wb') as f:
    pickle.dump(logger, f)