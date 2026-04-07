import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.font_manager as fm
import pandas as pd
import torch
import pickle
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from modules import *

plt.rcParams['font.size'] = 14
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = fm.FontProperties(fname = './fonts/Arial.ttf')
fm.fontManager.addfont('./fonts/Arial.ttf') # registers the font
plt.rcParams['font.family'] = font.get_name()

NUM_JOBS = 5





"""
Set environment
"""

# parse args
parser = ArgParser()
args = parser.args





"""
Read data
"""

data = []
for jobid in range(NUM_JOBS):
    exp_path = os.path.join(args.path, f'logger_{args.num_bandits}_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}_{jobid}')

    with open(os.path.join(exp_path, 'logger_pcb.pkl'), 'rb') as file:
        data_jobid = pickle.load(file)

    data.append(data_jobid)

# print(data[0].keys())




"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure_{args.num_bandits}_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Choosing left vs. left-right/left-mean other rating difference (Fig 3A)
"""

df = pd.concat([data_jobid['df_choice_by_rating_difference'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
df_filtered = df[(df['reward_differences'] >= -5) & (df['reward_differences'] <= 5)]
df_grouped = df_filtered.groupby(['jobid', 'reward_differences'])['if_left_chosens'].mean().reset_index()
means = df_grouped.groupby(['reward_differences'])['if_left_chosens'].mean().reset_index()
errors = df_grouped.groupby(['reward_differences'])['if_left_chosens'].std(ddof = 1).reset_index()
errors['if_left_chosens'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.25, 2.8))
plt.errorbar(means['reward_differences'], means['if_left_chosens'], yerr = errors['if_left_chosens'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axvline(x = 0, color = 'k', linestyle = '--', linewidth = 1)
plt.axhline(y = 1 / args.num_bandits, color = 'k', linestyle = '--', linewidth = 1)
plt.ylim((0, 1))
if args.num_bandits == 2:
    plt.xlabel('Left rating - right rating')
elif args.num_bandits == 3:
    plt.xlabel('Left rating - mean other')
plt.ylabel('p(left chosen)')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_choice_by_rating_difference.pdf'), bbox_inches = 'tight')





"""
Distribution of fixation duration (Fig 3B & 4A)
"""

fixation_counts = np.concatenate([data_jobid['fixation_counts'] for data_jobid in data])

plt.figure(figsize = (3.25, 2.8))
sns.kdeplot(fixation_counts, bw_adjust = 2.5, gridsize = 1000, color = 'black')
plt.xticks([0, 20, 40, 60])
plt.xlim(-5, 80)
plt.ylim(0, 0.1)
plt.xlabel('Total sampling time')
plt.ylabel('Density')
plt.title('Agent', pad = 20)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText = True))
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0)) # force scientific notation
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_duration_distribution.pdf'), bbox_inches = 'tight')


merged_uniques = np.stack([data_jobid['merged_fixation_count_uniques'] for data_jobid in data])
merged_proportions = np.stack([data_jobid['merged_fixation_count_proportions'] for data_jobid in data])

plt.figure(figsize = (3.25, 2.8))
means = np.nanmean(merged_proportions, axis = 0)
errors = np.nanstd(merged_proportions, axis = 0, ddof = 1) / np.sqrt(np.sum(~np.isnan(merged_proportions), axis = 0)) # per-column sample size
plt.bar(x = merged_uniques[0], height = means, color = 'k')
plt.xticks([1, 5, 10])
plt.xlim(0, 11)
plt.ylim((0, 0.4))
plt.xlabel('Number of samples')
plt.ylabel('Proportion')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_duration_distribution_merged.pdf'), bbox_inches = 'tight')





"""
Fixation duration vs. best-worst/best-mean other rating difference (Fig 3C & 4B)
"""

df = pd.concat([data_jobid['df_fixaiton_num_by_rating_difference'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)

df_grouped = df.groupby(['jobid', 'reward_differences'])['fixation_counts'].mean().reset_index()
means = df_grouped.groupby(['reward_differences'])['fixation_counts'].mean().reset_index()
errors = df_grouped.groupby(['reward_differences'])['fixation_counts'].std(ddof = 1).reset_index()
errors['fixation_counts'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.25, 2.8))
plt.errorbar(means['reward_differences'], means['fixation_counts'], yerr = errors['fixation_counts'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 1))
if args.num_bandits == 2:
    plt.xlabel('Best rating - worst rating')
elif args.num_bandits == 3:
    plt.xlabel('Best rating - mean other rating')
plt.ylabel('# of samples')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_fixation_num_by_rating_difference.pdf'), bbox_inches = 'tight')


df_grouped = df.groupby(['jobid', 'reward_differences'])['merged_fixation_counts'].mean().reset_index()
means = df_grouped.groupby(['reward_differences'])['merged_fixation_counts'].mean().reset_index()
errors = df_grouped.groupby(['reward_differences'])['merged_fixation_counts'].std(ddof = 1).reset_index()
errors['merged_fixation_counts'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.25, 2.8))
plt.errorbar(means['reward_differences'], means['merged_fixation_counts'], yerr = errors['merged_fixation_counts'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.xticks([0, 5, 10])
plt.ylim((2, 6))
if args.num_bandits == 2:
    plt.xlabel('Best rating - worst rating')
elif args.num_bandits == 3:
    plt.xlabel('Best rating - mean other')
plt.ylabel('Number of samples')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_fixation_num_by_rating_difference_merged.pdf'), bbox_inches = 'tight')





"""
Fixaion duration vs. rating mean (Fig 3D)
"""

df = pd.concat([data_jobid['df_fixaiton_num_by_rating_mean'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)

df_grouped = df.groupby(['jobid', 'values'])['fixation_counts'].mean().reset_index()
means = df_grouped.groupby(['values'])['fixation_counts'].mean().reset_index()
errors = df_grouped.groupby(['values'])['fixation_counts'].std(ddof = 1).reset_index()
errors['fixation_counts'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.8))
plt.errorbar(means['values'], means['fixation_counts'], yerr = errors['fixation_counts'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 1))
plt.xlabel('Mean item rating')
plt.ylabel('# of samples')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_fixation_num_by_rating_mean.pdf'), bbox_inches = 'tight')


df_grouped = df.groupby(['jobid', 'values'])['merged_fixation_counts'].mean().reset_index()
means = df_grouped.groupby(['values'])['merged_fixation_counts'].mean().reset_index()
errors = df_grouped.groupby(['values'])['merged_fixation_counts'].std(ddof = 1).reset_index()
errors['merged_fixation_counts'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.25, 2.8))
plt.errorbar(means['values'], means['merged_fixation_counts'], yerr = errors['merged_fixation_counts'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 1))
plt.xticks([0, 5, 10])
plt.ylim((1.8, 6.2))
plt.xlabel('Mean item rating')
plt.ylabel('Number of samples')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_fixation_num_by_rating_mean_merged.pdf'), bbox_inches = 'tight')





"""
Fixation duration vs. fixation number (Fig 4C)
"""

fixation_lengths = [data_jobid['fixaiton_length_by_fixation_index'] for data_jobid in data]
max_fixation_number = 7
plt.figure(figsize = (3.3, 2.8))
colors = ['#000000', '#232323', '#464646', '#696969', '#8c8c8c', '#afafaf', '#d2d2d2']
for i in range(max_fixation_number):
    fixation_lengths_i = np.stack([_[i] for _ in fixation_lengths])
    means = fixation_lengths_i.mean(axis = 0)
    errors = fixation_lengths_i.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
    plt.errorbar(np.arange(1, len(means) + 1), means, yerr = errors, color = colors[i], fmt = 'o-', elinewidth = 1, capsize = 0, zorder = -i)
plt.xticks(ticks = np.arange(1, max_fixation_number + 1))
plt.ylim(2, 5)
plt.xlabel('Sample number')
plt.ylabel('Sample duration')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_fixation_length_by_fixation_index.pdf'), bbox_inches = 'tight')





"""
Distribution of fixation advantage (Fig 5A)
"""

fixation_advantages = np.concatenate([data_jobid['fixation_advantages'] for data_jobid in data])

plt.figure(figsize = (3.4, 2.8))
sns.kdeplot(fixation_advantages, bw_adjust = 4.5, gridsize = 1000, color = 'black')
plt.axvline(x = 0, color = 'k', linestyle = '--', linewidth = 1)
plt.xlim(-11, 11)
plt.ylim(0, 0.23)
plt.xlabel('Time advantage')
plt.ylabel('Density')
plt.title('Agent', pad = 20)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText = True))
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0)) # force scientific notation
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_fixation_advantage.pdf'), bbox_inches = 'tight')





"""
Proportion fixate left vs. left-right/left-mean other rating difference (Fig 6A)
"""

df = pd.concat([data_jobid['df_fixation_proportion_by_rating_difference'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
df_filtered = df[(df['reward_differences'] >= -5) & (df['reward_differences'] <= 5)]
df_grouped = df_filtered.groupby(['jobid', 'reward_differences'])['fixation_proportions'].mean().reset_index()
means = df_grouped.groupby(['reward_differences'])['fixation_proportions'].mean().reset_index()
errors = df_grouped.groupby(['reward_differences'])['fixation_proportions'].std(ddof = 1).reset_index()
errors['fixation_proportions'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.25, 2.8))
plt.errorbar(means['reward_differences'], means['fixation_proportions'], yerr = errors['fixation_proportions'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 1 / args.num_bandits, color = 'k', linestyle = '--', linewidth = 1)
plt.xticks([-5, 0, 5])
plt.ylim((0.1, 0.65))
if args.num_bandits == 2:
    plt.xlabel('Left rating - right rating')
elif args.num_bandits == 3:
    plt.xlabel('Left rating - mean other')
plt.ylabel('Proportion sample left')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_fixation_proportion_by_rating_difference.pdf'), bbox_inches = 'tight')





"""
First fixation duration vs. first fixated item rating (Fig 6B)
"""

df = pd.concat([data_jobid['df_first_fixation_duration_by_rating'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
df_grouped = df.groupby(['jobid', 'first_fixation_rewards'])['first_fixation_lengths'].mean().reset_index()
means = df_grouped.groupby(['first_fixation_rewards'])['first_fixation_lengths'].mean().reset_index()
errors = df_grouped.groupby(['first_fixation_rewards'])['first_fixation_lengths'].std(ddof = 1).reset_index()
errors['first_fixation_lengths'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.8))
plt.errorbar(means['first_fixation_rewards'], means['first_fixation_lengths'], yerr = errors['first_fixation_lengths'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.ylim((0, 5))
plt.xlabel('First sampled item rating')
plt.ylabel('First sample duration')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_first_fixation_duration_by_rating.pdf'), bbox_inches = 'tight')





"""
Probability of fixating the worst vs. cumulative fixation time (Fig 6C)
"""

df = pd.concat([data_jobid['df_worst_fixation_proportion_by_cumulative_length'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
df_grouped = df.groupby(['jobid', 'cum_fixation_lengths'])['if_fixate_worsts'].mean().reset_index()
df_grouped = df_grouped[df_grouped['cum_fixation_lengths'] <= 15] #######

df_grouped.to_csv(os.path.join(exp_path, 'df_worst.csv'), index = False)

means = df_grouped.groupby(['cum_fixation_lengths'])['if_fixate_worsts'].mean().reset_index()
errors = df_grouped.groupby(['cum_fixation_lengths'])['if_fixate_worsts'].std(ddof = 1).reset_index()
errors['if_fixate_worsts'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.25, 2.8))
plt.errorbar(means['cum_fixation_lengths'], means['if_fixate_worsts'], yerr = errors['if_fixate_worsts'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 1 / args.num_bandits, color = 'k', linestyle = '--', linewidth = 1)
plt.xticks([0, 5, 10, 15])
plt.ylim((0.18, 0.6))
plt.xlabel('Cumulative sampling time')
plt.ylabel('p(sample worst)')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_worst_fixation_proportion_by_cumulative_length.pdf'), bbox_inches = 'tight')





"""
Choosing last fixated item vs. rating difference (Fig 7A)
"""

df = pd.concat([data_jobid['df_last_fixation_chosen_by_rating_difference'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
df_filtered = df[(df['reward_differences'] >= -5) & (df['reward_differences'] <= 5)]
df_grouped = df_filtered.groupby(['jobid', 'reward_differences'])['if_last_chosens'].mean().reset_index()
means = df_grouped.groupby(['reward_differences'])['if_last_chosens'].mean().reset_index()
errors = df_grouped.groupby(['reward_differences'])['if_last_chosens'].std(ddof = 1).reset_index()
errors['if_last_chosens'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.25, 3))
plt.errorbar(means['reward_differences'], means['if_last_chosens'], yerr = errors['if_last_chosens'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 1 / args.num_bandits, color = 'k', linestyle = '--', linewidth = 1)
plt.axvline(x = 0, color = 'k', linestyle = '--', linewidth = 1)
plt.xticks([-5, 0, 5])
plt.ylim((-0.05, 1.05))
if args.num_bandits == 2:
    plt.xlabel('Last sampled rating -\nother rating')
elif args.num_bandits == 3:
    plt.xlabel('Last sampled rating -\nmean other')
plt.ylabel('p(last sampled chosen)')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_last_fixation_chosen_by_rataing_difference.pdf'), bbox_inches = 'tight')





"""
Choosing left vs. time advantage left (Fig 7B)
"""

df = pd.concat([data_jobid['df_choice_by_fixation_advantage'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
bins = [-10.5, -7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5]
labels = [-9, -6, -3, 0, 3, 6, 9]
df['group'] = pd.cut(df['final_fixation_advantages'], bins = bins, labels = labels, include_lowest = False, right = False)
df_grouped = df.groupby(['jobid', 'group'])['if_left_chosens'].mean().reset_index()
# df_grouped = df_grouped[(df_grouped['final_fixation_advantages'] >= -10) & (df_grouped['final_fixation_advantages'] <= 10)]
means = df_grouped.groupby(['group'])['if_left_chosens'].mean().reset_index()
errors = df_grouped.groupby(['group'])['if_left_chosens'].std(ddof = 1).reset_index()
errors['if_left_chosens'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.25, 3))
plt.errorbar(means['group'], means['if_left_chosens'], yerr = errors['if_left_chosens'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 1 / args.num_bandits, color = 'k', linestyle = '--', linewidth = 1)
plt.axvline(x = 0, color = 'k', linestyle = '--', linewidth = 1)
plt.ylim((-0.05, 1.05))
plt.xlabel('Final time\nadvantage left')
plt.ylabel('p(left chosen)')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_choice_by_fixation_advantage.pdf'), bbox_inches = 'tight')





"""
Choosing first fixated vs. first fixated furation (Figure 7C)
"""

df = pd.concat([data_jobid['df_choice_by_first_fixation_duration'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
df_grouped = df.groupby(['jobid', 'first_fixation_lengths'])['if_first_chosens'].mean().reset_index()
df_grouped = df_grouped[df_grouped['first_fixation_lengths'] <= 10]
means = df_grouped.groupby(['first_fixation_lengths'])['if_first_chosens'].mean().reset_index()
errors = df_grouped.groupby(['first_fixation_lengths'])['if_first_chosens'].std(ddof = 1).reset_index()
errors['if_first_chosens'] /= np.sqrt(NUM_JOBS)
plt.figure(figsize = (3.2, 2.8))
plt.errorbar(means['first_fixation_lengths'], means['if_first_chosens'], yerr = errors['if_first_chosens'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 1 / args.num_bandits, color = 'k', linestyle = '--', linewidth = 1)
plt.xticks([0, 5, 10])
plt.ylim((0, 1))
plt.xlabel('First sample duration')
plt.ylabel('p(first sampled chosen)')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_pcb_choice_by_first_fixation_duration.pdf'), bbox_inches = 'tight')

