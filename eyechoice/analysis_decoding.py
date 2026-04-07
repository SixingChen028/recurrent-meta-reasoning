import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
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
exp_path = os.path.join(args.path, f'exp_{args.num_bandits}_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}_{args.jobid}')

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
data = load_data(os.path.join(exp_path, f'data_simulation_decoding.p'))
data = preprocess(data, args, merge_fixations = False)
num_trials = len(data['action_seqs'])
print('Keys:', data.keys())
print(num_trials)





"""
Set logger path
"""

# set experiment path
exp_path = os.path.join(args.path, f'logger_{args.num_bandits}_{args.jobid}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Data gathering
"""

hiddens = []
items = []
posterior_means = []
posterior_precisions = []

for i in range(num_trials):
    length_ep, hidden_seq_ep, item_seq_ep, posterior_means_seq_ep, posterior_precisions_seq_ep = pull(
        data, i, 'lengths', 'hidden_seqs', 'item_seqs', 'posterior_means_seqs', 'posterior_precisions_seqs'
    )

    if length_ep > 1 and length_ep < args.t_max:

        hiddens.append(np.array(hidden_seq_ep)[1:])
        items.append(item_seq_ep[:-1])
        posterior_means.append(np.array(posterior_means_seq_ep)[:-1])
        posterior_precisions.append(np.array(posterior_precisions_seq_ep)[:-1])

hiddens = np.concatenate(hiddens)
items = np.concatenate(items)
posterior_means = np.concatenate(posterior_means)
posterior_precisions = np.concatenate(posterior_precisions)

print(hiddens.shape)
print(items.shape)
print(posterior_means.shape)
print(posterior_precisions.shape)





"""
Decode attended item
"""

model = LogisticRegression(max_iter = 1000, random_state = 42)
scores_item = cross_val_score(model, hiddens, items, cv = 5, scoring = 'accuracy')

# shuffle the labels
items_shuffled = shuffle(items, random_state = 42)
baseline_scores_item = cross_val_score(model, hiddens, items_shuffled, cv = 5, scoring = 'accuracy')

print(f'accuracy: {np.mean(scores_item):.3f}')
print(f'baseline accuracy: {np.mean(baseline_scores_item):.3f}')

logger['scores_item'] = np.mean(scores_item)
logger['baseline_scores_item'] = np.mean(baseline_scores_item)





"""
Decode posterior mean
"""

model = LinearRegression()
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

errors_mean = []
r2s_mean = []

for train_idx, test_idx in kf.split(hiddens):
    X_train, X_test = hiddens[train_idx], hiddens[test_idx]
    y_train, y_test = posterior_means[train_idx], posterior_means[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # compute MSE per dimension and average
    errors_mean.append(np.mean(mean_squared_error(y_test, y_pred, multioutput = 'raw_values')))
    r2s_mean.append(np.mean(r2_score(y_test, y_pred, multioutput = 'raw_values')))

print(f'error: {np.mean(errors_mean):.3f}')
print(f'r2: {np.mean(r2s_mean):.3f}')

# shuffle the targets across all rows
posterior_means_shuffled = shuffle(posterior_means, random_state = 42)

baseline_errors_mean = []
baseline_r2s_mean = []

for train_idx, test_idx in kf.split(hiddens):
    X_train, X_test = hiddens[train_idx], hiddens[test_idx]
    y_train, y_test = posterior_means_shuffled[train_idx], posterior_means_shuffled[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # compute MSE per dimension and average
    baseline_errors_mean.append(np.mean(mean_squared_error(y_test, y_pred, multioutput = 'raw_values')))
    baseline_r2s_mean.append(np.mean(r2_score(y_test, y_pred, multioutput = 'raw_values')))

print(f'baseline error: {np.mean(baseline_errors_mean):.3f}')
print(f'baseline r2: {np.mean(baseline_r2s_mean):.3f}')

logger['errors_mean'] = np.mean(errors_mean)
logger['r2s_mean'] = np.mean(r2s_mean)

logger['baseline_errors_mean'] = np.mean(baseline_errors_mean)
logger['baseline_r2s_mean'] = np.mean(baseline_r2s_mean)





"""
Decode posterior precision
"""

model = LinearRegression()
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

errors_precision = []
r2s_precision = []

for train_idx, test_idx in kf.split(hiddens):
    X_train, X_test = hiddens[train_idx], hiddens[test_idx]
    y_train, y_test = posterior_precisions[train_idx], posterior_precisions[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # compute MSE per dimension and average
    errors_precision.append(np.mean(mean_squared_error(y_test, y_pred, multioutput = 'raw_values')))
    r2s_precision.append(np.mean(r2_score(y_test, y_pred, multioutput = 'raw_values')))

print(f'error: {np.mean(errors_precision):.3f}')
print(f'r2: {np.mean(r2s_precision):.3f}')

# shuffle the targets across all rows
posterior_precisions_shuffled = shuffle(posterior_precisions, random_state = 42)

baseline_errors_precision = []
baseline_r2s_precision = []

for train_idx, test_idx in kf.split(hiddens):
    X_train, X_test = hiddens[train_idx], hiddens[test_idx]
    y_train, y_test = posterior_precisions_shuffled[train_idx], posterior_precisions_shuffled[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # compute MSE per dimension and average
    baseline_errors_precision.append(np.mean(mean_squared_error(y_test, y_pred, multioutput = 'raw_values')))
    baseline_r2s_precision.append(np.mean(r2_score(y_test, y_pred, multioutput = 'raw_values')))

print(f'baseline error: {np.mean(baseline_errors_precision):.3f}')
print(f'baseline r2: {np.mean(baseline_r2s_precision):.3f}')

logger['errors_precision'] = np.mean(errors_precision)
logger['r2s_precision'] = np.mean(r2s_precision)

logger['baseline_errors_precision'] = np.mean(baseline_errors_precision)
logger['baseline_r2s_precision'] = np.mean(baseline_r2s_precision)





"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_decoding.pkl'), 'wb') as f:
    pickle.dump(logger, f)