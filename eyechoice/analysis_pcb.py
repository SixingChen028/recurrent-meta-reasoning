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
data = load_data(os.path.join(exp_path, f'data_simulation.p'))
data = preprocess(data, args, merge_fixations = False)
num_trials = len(data['values'])
print('Keys:', data.keys())
print(num_trials)





"""
Set logger path
"""

# set experiment path
exp_path = os.path.join(args.path, f'logger_{args.num_bandits}_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}_{args.jobid}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Performance analysis
"""

correctness = []
abortions = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max:
        correctness.append(values_ep[decision_ep] == np.max(values_ep))
    abortions.append(length_ep == args.t_max)

logger['mean_accuracy'] = np.mean(correctness)
logger['abortion_rate'] = np.mean(abortions)





"""
Choosing left vs. left-right/left-mean other rating difference (Fig 3A)
"""

reward_differences = []
if_left_chosens = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max:
        if args.num_bandits == 2:
            reward_differences.append(values_ep[0] - values_ep[1])
        elif args.num_bandits == 3:
            reward_differences.append(values_ep[0] - np.mean(values_ep[1:3]))
        if_left_chosens.append(int(decision_ep == 0))

df = pd.DataFrame({
    'reward_differences': reward_differences,
    'if_left_chosens': if_left_chosens,
})

logger['df_choice_by_rating_difference'] = df





"""
Distribution of fixation duration (Fig 3B & 4A)
"""

fixation_counts = []
merged_fixation_counts = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max:
        fixation_counts.append(len(fixation_seq_ep))

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)
        merged_fixation_counts.append(len(merged_fixation_seq_ep))

uniques = np.arange(0, 60)
proportions = compute_aligned_proportions(fixation_counts, uniques)

merged_uniques = np.arange(0, 12)
merged_proportions = compute_aligned_proportions(merged_fixation_counts, merged_uniques)

# logger['fixation_count_uniques'] = uniques
# logger['fixation_count_proportions'] = proportions

logger['fixation_counts'] = fixation_counts

logger['merged_fixation_count_uniques'] = merged_uniques
logger['merged_fixation_count_proportions'] = merged_proportions




"""
Fixation duration vs. best-worst/best-mean other rating difference (Fig 3C & 4B)
"""

reward_differences = []
fixation_counts = []
merged_fixation_counts = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max:
        if args.num_bandits == 2:
            reward_differences.append(np.abs(values_ep[0] - values_ep[1]))
        elif args.num_bandits == 3:
            reward_differences.append(np.max(values_ep) - np.mean(np.delete(values_ep, np.argmax(values_ep))))

        fixation_counts.append(len(fixation_seq_ep))

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)
        merged_fixation_counts.append(len(merged_fixation_seq_ep))

df = pd.DataFrame({
    'reward_differences': reward_differences,
    'fixation_counts': fixation_counts,
    'merged_fixation_counts': merged_fixation_counts,
})

logger['df_fixaiton_num_by_rating_difference'] = df





"""
Fixaion duration vs. rating mean (Fig 3D)
"""

values = []
fixation_counts = []
merged_fixation_counts = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max:
        values.append(np.mean(values_ep))

        fixation_counts.append(len(fixation_seq_ep))

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)
        merged_fixation_counts.append(len(merged_fixation_seq_ep))

df = pd.DataFrame({
    'values': values,
    'fixation_counts': fixation_counts,
    'merged_fixation_counts': merged_fixation_counts,
})

logger['df_fixaiton_num_by_rating_mean'] = df





"""
Fixation duration vs. fixation number (Fig 4C)
"""

max_fixation_number = 7
fixation_lengths = [[] for _ in range(max_fixation_number)]

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max and len(fixation_seq_ep) > 0:

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)

        if len(fixation_len_seq_ep) <= max_fixation_number:
            fixation_lengths[len(fixation_len_seq_ep) - 1].append(fixation_len_seq_ep)

fixation_lengths = [np.array(_).mean(axis = 0) for _ in fixation_lengths]

logger['fixaiton_length_by_fixation_index'] = fixation_lengths




"""
Distribution of fixation advantage (Fig 5A)
"""

fixation_advantages = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max and len(fixation_seq_ep) > 0:

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)

        cum_fixation_len_ep = np.zeros(args.num_bandits) # running sum of cumulative fixation lengths for both options

        for t, fixation in enumerate(merged_fixation_seq_ep):
            if t >= 1: # exclude the first fixation
                if args.num_bandits == 2:
                    fixation_advantages.append(cum_fixation_len_ep[fixation] - cum_fixation_len_ep[1 - fixation])
                elif args.num_bandits == 3:
                    fixation_advantages.append(cum_fixation_len_ep[fixation] - np.mean(cum_fixation_len_ep[np.arange(3) != fixation]))
            cum_fixation_len_ep[fixation] += fixation_len_seq_ep[t]

logger['fixation_advantages'] = fixation_advantages





"""
Proportion fixate left vs. left-right/left-other rating difference (Fig 6A)
"""

reward_differences = []
fixation_proportions = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max and len(fixation_seq_ep) > 0:

        # randomly choose a reference option (instead of always 0)
        ref_idx = random.choice(range(args.num_bandits))

        if args.num_bandits == 2:
            # the "other" index is the one not equal to ref_idx
            other_idx = 1 - ref_idx
            reward_differences.append(values_ep[ref_idx] - values_ep[other_idx])

        elif args.num_bandits == 3:
            # take mean of the two others
            other_indices = [j for j in range(3) if j != ref_idx]
            reward_differences.append(values_ep[ref_idx] - np.mean([values_ep[j] for j in other_indices]))

        # fixation proportion relative to the chosen reference option
        fixation_proportions.append(fixation_seq_ep.count(ref_idx) / len(fixation_seq_ep))

df = pd.DataFrame({
    'reward_differences': reward_differences,
    'fixation_proportions': fixation_proportions,
})

logger['df_fixation_proportion_by_rating_difference'] = df





"""
First fixation duration vs. first fixated item rating (Fig 6B)
"""

first_fixation_rewards = []
first_fixation_lengths = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max and len(fixation_seq_ep) > 0:

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)

        first_fixation_rewards.append(values_ep[merged_fixation_seq_ep[0]])
        first_fixation_lengths.append(fixation_len_seq_ep[0])

df = pd.DataFrame({
    'first_fixation_rewards': first_fixation_rewards,
    'first_fixation_lengths': first_fixation_lengths,
})

logger['df_first_fixation_duration_by_rating'] = df





"""
Probability of fixating the worst vs. cumulative fixation time (Fig 6C)
"""

cum_fixation_lengths = []
if_fixate_worsts = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max and len(fixation_seq_ep) > 0 and len(np.unique(values_ep)) == args.num_bandits:

        cum_fixation_len_ep = 0 # running sum of cumulative fixation lengths

        for t, fixation in enumerate(fixation_seq_ep):
            cum_fixation_lengths.append(cum_fixation_len_ep)
            if_fixate_worsts.append(int(values_ep[fixation] == min(values_ep)))

            cum_fixation_len_ep += 1

df = pd.DataFrame({
    'cum_fixation_lengths': cum_fixation_lengths,
    'if_fixate_worsts': if_fixate_worsts,
})

logger['df_worst_fixation_proportion_by_cumulative_length'] = df





"""
Choosing last fixated item vs. rating difference (Fig 7A)
"""

reward_differences = []
if_last_chosens = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max and len(fixation_seq_ep) > 0:

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)

        final_fixation = merged_fixation_seq_ep[-1]

        values_ep = np.array(values_ep)
        if args.num_bandits == 2:
            reward_differences.append(values_ep[final_fixation] - values_ep[1 - final_fixation])
        elif args.num_bandits == 3:
            reward_differences.append(values_ep[final_fixation] - np.mean(values_ep[np.arange(3) != final_fixation]))
        if_last_chosens.append(int(decision_ep == final_fixation))

df = pd.DataFrame({
    'reward_differences': reward_differences,
    'if_last_chosens': if_last_chosens,
})

logger['df_last_fixation_chosen_by_rating_difference'] = df





"""
Choosing left vs. time advantage left (Fig 7B)
"""


final_fixation_advantages = []
if_left_chosens = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max and len(fixation_seq_ep) > 0:

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)
        merged_fixation_seq_ep = np.array(merged_fixation_seq_ep)
        fixation_len_seq_ep = np.array(fixation_len_seq_ep)

        if args.num_bandits == 2:
            left_indices = np.where(merged_fixation_seq_ep == 0)[0]
            right_indices = np.where(merged_fixation_seq_ep == 1)[0]
            time_advantage = fixation_len_seq_ep[left_indices].sum() - fixation_len_seq_ep[right_indices].sum()
        elif args.num_bandits == 3:
            left_indices = np.where(merged_fixation_seq_ep == 0)[0]
            other1_indices = np.where(merged_fixation_seq_ep == 1)[0]
            other2_indices = np.where(merged_fixation_seq_ep == 2)[0]
            time_advantage = np.sum(fixation_len_seq_ep[left_indices]) - np.mean([
                np.sum(fixation_len_seq_ep[other1_indices]),
                np.sum(fixation_len_seq_ep[other2_indices]),
            ])
        final_fixation_advantages.append(time_advantage)
        if_left_chosens.append(int(decision_ep == 0))

df = pd.DataFrame({
    'final_fixation_advantages': final_fixation_advantages,
    'if_left_chosens': if_left_chosens,
})

logger['df_choice_by_fixation_advantage'] = df





"""
Choosing first fixated vs. first fixated furation (Figure 7C)
"""

first_fixation_lengths = []
if_first_chosens = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions'
    )
    
    if length_ep < args.t_max and len(fixation_seq_ep) > 0:

        merged_fixation_seq_ep, fixation_len_seq_ep = merge(fixation_seq_ep)

        # randomly choose a reference option
        ref_idx = random.choice(range(args.num_bandits))

        # find the first fixation on that reference option, if any
        if ref_idx in merged_fixation_seq_ep:
            ref_pos = merged_fixation_seq_ep.index(ref_idx)
            first_fixation_lengths.append(fixation_len_seq_ep[ref_pos])
            if_first_chosens.append(int(decision_ep == ref_idx))
        else:
            # if the reference option was never fixated, skip this trial
            continue

df = pd.DataFrame({
    'first_fixation_lengths': first_fixation_lengths,
    'if_first_chosens': if_first_chosens,
})

logger['df_choice_by_first_fixation_duration'] = df



"""
Save data
"""
# save the logger dictionary
with open(os.path.join(exp_path, 'logger_pcb.pkl'), 'wb') as f:
    pickle.dump(logger, f)