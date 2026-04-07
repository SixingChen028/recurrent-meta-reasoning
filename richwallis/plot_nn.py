import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.formula.api as smf
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
    exp_path = os.path.join(args.path, f'logger_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}_{jobid}')

    with open(os.path.join(exp_path, 'logger_nn.pkl'), 'rb') as file:
        data_jobid = pickle.load(file)

    data.append(data_jobid)

# print(data[0].keys())




"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{args.beta_e_final}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
State count
"""

num_states_chosen = np.array([data_jobid['num_states_chosen'] for data_jobid in data])
num_states_unchosen = np.array([data_jobid['num_states_unchosen'] for data_jobid in data])

means = np.array([np.mean(num_states_chosen), np.mean(num_states_unchosen)])
errors = np.array([np.std(num_states_chosen, ddof = 1), np.std(num_states_unchosen, ddof = 1)])
errors /= np.sqrt(NUM_JOBS)

print(np.stack([num_states_chosen, num_states_unchosen]))

print(means, errors)

plt.figure(figsize = (2.35, 2.8))
plt.bar(['Ch.', 'Unch.'], means, color = ['#C90017', '#0068A8'], width = 0.5, yerr = errors, ecolor = 'black', capsize = 0)
plt.xlim(-0.5, 1.5)
plt.xlabel('Offer type')
plt.ylabel('States per trial')
plt.title('Agent', pad = 15)
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_state_count.pdf'), bbox_inches = 'tight')


from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(
    num_states_chosen,
    num_states_unchosen,
)

print(f"t = {t_stat}, p = {p_value}")
print(len(num_states_chosen) - 1)





"""
Number of transitions by value
"""

df = pd.concat([data_jobid['df_num_transitions_by_value'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)

df_grouped = df.groupby(['jobid', 'values_chosen'])['num_transitions'].mean().reset_index()
df_summary = df_grouped.groupby(['values_chosen'])['num_transitions'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])

# ---- fit linear mixed model ----
# random intercepts per session
model = smf.mixedlm('num_transitions ~ values_chosen', df, groups = df['jobid'])
result = model.fit(reml = False)
# print(result.summary())
# ---- get fitted values for plotting ----
x_pred = np.linspace(1, 4, 200)
df_pred = pd.DataFrame({'values_chosen': x_pred})
df_pred['num_transitions'] = result.predict(df_pred)

plt.figure(figsize = (2.6, 2.8))
plt.errorbar(df_summary['values_chosen'], df_summary['mean'], yerr = df_summary['se'], fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# sns.regplot(data = df, x = 'values_chosen', y = 'num_transitions', ci = False, scatter = False, color = 'k')
plt.plot(df_pred['values_chosen'], df_pred['num_transitions'], color = 'k', linewidth = 2)
plt.xlim((0.7, 4.3))
plt.xticks([1, 2, 3, 4])
plt.ylim((1.6, 3.6))
# plt.yticks([2, 2.5, 3])
plt.xlabel('Chosen value')
plt.ylabel('Transitions per trial')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_num_transitions_by_chosen_value.pdf'), bbox_inches = 'tight')

print(df_grouped)

# ---- fit linear mixed model ----
# random intercepts per session
model = smf.mixedlm('num_transitions ~ values_chosen + values_unchosen', df, groups = df['jobid'])
result = model.fit(reml = False)
print(result.summary())
print(result.pvalues)





"""
Number of transitions by correctness
"""

df = pd.concat([data_jobid['df_num_transitions_by_correctness'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
df_grouped = df.groupby(['jobid', 'correctness'])['num_transitions'].mean().reset_index()
df_summary = df_grouped.groupby(['correctness'])['num_transitions'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])
plt.figure(figsize = (2.2, 2.8))
plt.bar(['Cor.', 'Inc.'], df_summary['mean'][::-1], color = ['#FFC34D', '#999999'], width = 0.5, yerr = df_summary['se'][::-1], ecolor = 'black', capsize = 0)
plt.xlim(-0.5, 1.5)
plt.xlabel('Correctness')
plt.ylabel('Transitions per trial')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_num_transitions_by_correctness.pdf'), bbox_inches = 'tight')


# pivot the dataframe so each session_id has two columns: True and False
df_pivot = df_grouped.pivot(index = 'jobid', columns = 'correctness', values = 'num_transitions')

# drop rows that have missing data (just in case)
df_pivot = df_pivot.dropna()

# run paired t-test
t_stat, p_val = ttest_rel(df_pivot[True], df_pivot[False])

print(f't = {t_stat:.3f}, p = {p_val:.3g}')
print(len(df_pivot) - 1)





"""
DIR
"""

dirs_chosen = np.array([data_jobid['dirs_chosen'] for data_jobid in data])
dirs_unchosen = np.array([data_jobid['dirs_unchosen'] for data_jobid in data])

means_chosen = np.mean(dirs_chosen, axis = 0)
errors_chosen = np.std(dirs_chosen, axis = 0, ddof = 1)
errors_chosen /= np.sqrt(NUM_JOBS)

means_unchosen = np.mean(dirs_unchosen, axis = 0)
errors_unchosen = np.std(dirs_unchosen, axis = 0, ddof = 1)
errors_unchosen /= np.sqrt(NUM_JOBS)

plt.figure(figsize = (3, 2.8))
plt.errorbar(np.arange(-1, len(means_chosen) - 1), means_chosen, yerr = errors_chosen, color = '#C90017', ecolor = '#C90017', fmt = 'o-', elinewidth = 1, capsize = 0)
plt.errorbar(np.arange(-1, len(means_unchosen) - 1), means_unchosen, yerr = errors_unchosen, color = '#0068A8', ecolor = '#0068A8', fmt = 'o-', elinewidth = 1, capsize = 0)
plt.axvline(x = 0, color = 'k', linestyle = '--', linewidth = 1)
# plt.xlim(-0.5, 1.5)
plt.xlabel('Time since OFC state')
plt.ylabel('DIR')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_dir.pdf'), bbox_inches = 'tight')