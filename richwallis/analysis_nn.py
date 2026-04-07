import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
exp_path = os.path.join(args.path, f'exp_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}_{args.jobid}')

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
        t_wait = args.t_wait,
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
exp_path = os.path.join(args.path, f'logger_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}_{args.jobid}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Value LDA
"""

values = []
hiddens = []

for i in range(num_trials):
    length_ep, values_ep, item_seq_ep, fixation_seq_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'item_seqs', 'fixation_seqs', 'hidden_seqs'
    )

    if length_ep > 1 and length_ep < args.t_max:

        hiddens.append(np.array(hidden_seq_ep)[1:args.t_wait + 2])
        values.append(np.array(values_ep)[np.array(item_seq_ep[:args.t_wait + 1])])

values = np.concatenate(values)
hiddens = np.concatenate(hiddens)

print(values.shape)
print(hiddens.shape)

# perform LDA
X = hiddens
y = values.astype(int) # ensure integer classes

# split with stratification so each class is represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# LDA setup
lda_value = LinearDiscriminantAnalysis()

# fit
lda_value.fit(X_train, y_train)

# predict / probabilities
y_pred = lda_value.predict(X_test)
proba = lda_value.predict_proba(X_test) # shape (N_test, n_classes)

# metrics
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification report:\n', classification_report(y_test, y_pred, digits = 3))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))





"""
Decision LDA
"""

decisions = []
hiddens = []

for i in range(num_trials):
    length_ep, values_ep, item_seq_ep, fixation_seq_ep, decision_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'item_seqs', 'fixation_seqs', 'decisions', 'hidden_seqs'
    )

    if length_ep > 1 and length_ep < args.t_max:

        hiddens.append(np.array(hidden_seq_ep)[1:args.t_wait + 2])
        decisions.append(np.repeat(decision_ep, args.t_wait + 1))

decisions = np.concatenate(decisions)
hiddens = np.concatenate(hiddens)

print(values.shape)
print(hiddens.shape)

# perform LDA
X = hiddens
y = decisions.astype(int) # ensure integer classes

# split with stratification so each class is represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# LDA setup
lda_decision = LinearDiscriminantAnalysis()

# fit
lda_decision.fit(X_train, y_train)

# predict / probabilities
y_pred = lda_decision.predict(X_test)
proba = lda_decision.predict_proba(X_test) # shape (N_test, n_classes)

# metrics
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification report:\n', classification_report(y_test, y_pred, digits = 3))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))





"""
Count states
"""

num_states_chosen = []
num_states_unchosen = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions', 'hidden_seqs'
    )

    if values_ep[0] != values_ep[1]:

        # get values
        value_chosen = int(values_ep[decision_ep])
        value_unchosen = int(values_ep[1 - decision_ep])

        # transform into posterior
        posteriors_ep = lda_value.predict_proba(np.array(hidden_seq_ep)[1:args.t_wait + 2])

        # get posterior sequences
        post_chosen = posteriors_ep[:, value_chosen - 1]
        post_unchosen = posteriors_ep[:, value_unchosen - 1]

        # count states
        num_states_chosen_ep, _ = count_states_and_lengths(post_chosen, threshold = 0.5)
        num_states_unchosen_ep, _ = count_states_and_lengths(post_unchosen, threshold = 0.5)

        # append results
        num_states_chosen.append(num_states_chosen_ep)
        num_states_unchosen.append(num_states_unchosen_ep)

logger['num_states_chosen'] = np.mean(num_states_chosen)
logger['num_states_unchosen'] = np.mean(num_states_unchosen)





"""
Number of transitions by value
"""

num_transitions = []
values_chosen = []
values_unchosen = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions', 'hidden_seqs'
    )

    if values_ep[0] != values_ep[1]:

        # get values
        value_chosen = int(values_ep[decision_ep])
        value_unchosen = int(values_ep[1 - decision_ep])

        # transform into posterior
        posteriors_ep = lda_value.predict_proba(np.array(hidden_seq_ep)[1:args.t_wait + 2])

        # get posterior sequences
        post_chosen = posteriors_ep[:, value_chosen - 1]
        post_unchosen = posteriors_ep[:, value_unchosen - 1]

        # get transitions
        n = count_transitions(np.stack([post_chosen, post_unchosen], axis = 1), min_length = 1, threshold = 0.5)
        if n >= 0:
            num_transitions.append(n)
            values_chosen.append(value_chosen)
            values_unchosen.append(value_unchosen)

df = pd.DataFrame({
    'num_transitions': num_transitions,
    'values_chosen': values_chosen,
    'values_unchosen': values_unchosen,
})

logger['df_num_transitions_by_value'] = df





"""
Number of transitions by correctness
"""

correctness = []
num_transitions = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions', 'hidden_seqs'
    )

    if values_ep[0] != values_ep[1]:
        # get values
        value_chosen = int(values_ep[decision_ep])
        value_unchosen = int(values_ep[1 - decision_ep])

        # transform into posterior
        posteriors_ep = lda_value.predict_proba(np.array(hidden_seq_ep)[1:args.t_wait + 2])

        # get posterior sequences
        post_chosen = posteriors_ep[:, value_chosen - 1]
        post_unchosen = posteriors_ep[:, value_unchosen - 1]

        # get transitions
        n = count_transitions(np.stack([post_chosen, post_unchosen], axis = 1), min_length = 1, threshold = 0.5)
        if n >= 0:
            num_transitions.append(n)
            correctness.append(values_ep[decision_ep] > values_ep[1 - decision_ep])

df = pd.DataFrame({
    'num_transitions': num_transitions,
    'correctness': correctness,
})

logger['df_num_transitions_by_correctness'] = df





"""
DIR decoding
"""

dir_slices_chosen = []
dir_slices_unchosen = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions', 'hidden_seqs'
    )

    if values_ep[0] != values_ep[1]:

        # get values
        value_chosen = int(values_ep[decision_ep])
        value_unchosen = int(values_ep[1 - decision_ep])

        # transform into posterior
        posteriors_ep = lda_value.predict_proba(np.array(hidden_seq_ep)[1:args.t_wait + 2])

        # get posterior sequences
        post_chosen = posteriors_ep[:, value_chosen - 1]
        post_unchosen = posteriors_ep[:, value_unchosen - 1]

        starts_chosen_ep, durations_chosen_ep = find_start_indices_and_lengths(post_chosen, threshold = 0.5)
        starts_unchosen_ep, durations_unchosen_ep = find_start_indices_and_lengths(post_unchosen, threshold = 0.5)

        # get dir
        dirs_ep = lda_decision.predict_proba(np.array(hidden_seq_ep)[1:])

        for k in range(len(starts_chosen_ep)):
            if starts_chosen_ep[k] >= 1:
                dir_slice_chosen = dirs_ep[starts_chosen_ep[k] - 1: starts_chosen_ep[k] + durations_chosen_ep[k], decision_ep]
                dir_slices_chosen.append(dir_slice_chosen[:10])
    
        for k in range(len(starts_unchosen_ep)):
            if starts_unchosen_ep[k] >= 1:
                dir_slice_unchosen = dirs_ep[starts_unchosen_ep[k] - 1: starts_unchosen_ep[k] + durations_unchosen_ep[k], decision_ep]
                dir_slices_unchosen.append(dir_slice_unchosen[:10])

dirs_chosen = np.full((len(dir_slices_chosen), 10), np.nan, dtype = float)
for i, a in enumerate(dir_slices_chosen):
    dirs_chosen[i, :len(a)] = a

dirs_unchosen = np.full((len(dir_slices_unchosen), 10), np.nan, dtype = float)
for i, a in enumerate(dir_slices_unchosen):
    dirs_unchosen[i, :len(a)] = a

logger['dirs_chosen'] = np.nanmean(dirs_chosen, axis = 0)
logger['dirs_unchosen'] = np.nanmean(dirs_unchosen, axis = 0)





"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_nn.pkl'), 'wb') as f:
    pickle.dump(logger, f)