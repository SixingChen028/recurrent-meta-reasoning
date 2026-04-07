import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import torch
import pickle
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
    exp_path = os.path.join(args.path, f'logger_{args.cost}_{args.beta_e_final}_{args.kappa_squared}_{jobid}')

    with open(os.path.join(exp_path, 'logger_logit.pkl'), 'rb') as file:
        data_jobid = pickle.load(file)

    data.append(data_jobid)

# print(data[0].keys())




"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure_{args.cost}_{args.beta_e_final}_{args.kappa_squared}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Effect of fixation on the state
"""

df = pd.concat([data_jobid['df_logits'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
plt.figure(figsize = (3, 2.8)) # (2.7, 2.8)
colors = ['#F5191D', '#E97000', '#E79912', '#E9BA20', '#C1C88D', '#8ABD94', '#4CAFA1', '#3B99B1']
for i, point in enumerate(df['points'].unique()):
    df_point = df[df['points'] == point]
    means = df_point.groupby(['time_steps'])['logits_slices'].mean().reset_index()
    errors = df_point.groupby(['time_steps'])['logits_slices'].std().reset_index()
    errors['logits_slices'] /= np.sqrt(NUM_JOBS)
    plt.errorbar(means['time_steps'] - 1, means['logits_slices'], yerr = errors['logits_slices'], label = point, fmt = 'o-', color = colors[i], ecolor = colors[i], elinewidth = 1, capsize = 0)
plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth = 1)
plt.xlim((-1.2, 2.2))
plt.ylim((-8, 12))
plt.xlabel('Time from fixation')
plt.ylabel('Logit')
plt.title('Self', pad = 15)
# plt.legend(title = 'Reward', bbox_to_anchor = (1.05, 1), loc = 'upper left', frameon = False, fontsize = 'small')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_logit_change.pdf'), bbox_inches = 'tight')





"""
Effect of fixation on the state (longer slices)
"""

df = pd.concat([data_jobid['df_logits_long'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
plt.figure(figsize = (4, 2.8))
colors = ['#F5191D', '#E97000', '#E79912', '#E9BA20', '#C1C88D', '#8ABD94', '#4CAFA1', '#3B99B1']
for i, point in enumerate(df['points'].unique()):
    df_point = df[df['points'] == point]
    means = df_point.groupby(['time_steps'])['logits_slices'].mean().reset_index()
    errors = df_point.groupby(['time_steps'])['logits_slices'].std().reset_index()
    errors['logits_slices'] /= np.sqrt(NUM_JOBS)
    plt.errorbar(means['time_steps'] - 1, means['logits_slices'], yerr = errors['logits_slices'], label = point, fmt = 'o-', color = colors[i], ecolor = colors[i], elinewidth = 1, capsize = 0)
plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth = 1)
plt.ylim((-15, 20))
plt.xlabel('Time from fixation')
plt.ylabel('Logit')
plt.legend(title = 'Reward', bbox_to_anchor = (1, 0.6), loc = 'center left', frameon = False, fontsize = 'small')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_logit_change_long.pdf'), bbox_inches = 'tight')




"""
Change of mind
"""

df = pd.concat([data_jobid['df_logits_change_of_mind'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
plt.figure(figsize = (6, 3))
for group in [0, 1]:
    df_filtered = df[df['groups'] == group]
    means = df_filtered.groupby(['time_steps'])['depth_1_logits'].mean().reset_index()
    errors = df_filtered.groupby(['time_steps'])['depth_1_logits'].std().reset_index()
    errors['depth_1_logits'] /= np.sqrt(NUM_JOBS)
    if group == 0:
        label = r'$Q(a_0)>Q(a_1)$ -> $Q(a_0)<Q(a_1)$'
    elif group == 1:
        label = r'$Q(a_0)<Q(a_1)$ -> $Q(a_0)>Q(a_1)$'
    plt.errorbar(means['time_steps'] - 1, means['depth_1_logits'], yerr = errors['depth_1_logits'], label = label, fmt = 'o-', elinewidth = 1, capsize = 0)
    plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth = 1)
    plt.xticks(np.arange(-1, 2))
    plt.ylim((-6, 6))
    plt.xlabel('Time from fixation')
    plt.ylabel(r'Logit($a_0$)')
plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', frameon = False, fontsize = 'small')
plt.tight_layout(pad = 1.5)
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_logit_change_of_mind.pdf'), bbox_inches = 'tight')

df = pd.concat([data_jobid['df_logits_change_of_mind_grouped'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)
plt.figure(figsize = (11, 2.8))
for i, depth in enumerate(range(1, 6)):
    plt.subplot(1, 5, i + 1)
    for group in [0, 1]:
        df_filtered = df[(df['groups'] == group) & (df['depths'] == depth)]
        means = df_filtered.groupby(['time_steps'])['depth_1_logits'].mean().reset_index()
        errors = df_filtered.groupby(['time_steps'])['depth_1_logits'].std().reset_index()
        errors['depth_1_logits'] /= np.sqrt(NUM_JOBS)
        if group == 0:
            label = r'Negative'
        elif group == 1:
            label = r'Positive'
        plt.errorbar(means['time_steps'] - 1, means['depth_1_logits'], yerr = errors['depth_1_logits'], label = label, fmt = 'o-', elinewidth = 1, capsize = 0)
        plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth = 1)
        plt.xticks(np.arange(-1, 2))
        plt.xlim((-1 - 0.15, 1 + 0.15))
        plt.ylim((-7, 7))
        if depth > 1:
            plt.yticks([])
        if depth == 3:
            plt.xlabel('Time from querying a state')
        if i == 0:
            plt.ylabel('Depth-1 decision logit')
        plt.title(f'Depth-{depth} state', pad = 15)
plt.legend(title = 'Flip direction', bbox_to_anchor = (1.05, 1), loc = 'upper left', frameon = False)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_logit_change_of_mind_by_depth.pdf'), bbox_inches = 'tight')







"""
All logits
"""

fig = plt.figure(figsize = (9.5, 2.9))

df_path = pd.concat([data_jobid['df_logits_path'] for data_jobid in data], keys=range(len(data)), names=['jobid']).reset_index(level=0)

colors = ['#F5191D', '#E97000', '#E79912', '#E9BA20', '#C1C88D', '#8ABD94', '#4CAFA1', '#3B99B1']
rel_labels = ['Self', 'Parent', 'Parent$^2$', 'Parent$^3$', 'Parent$^4$']

max_rel_level = int(df_path['rel_levels'].max())
num_panels = max_rel_level + 1

for j in range(num_panels):
    ax = plt.subplot(1, num_panels, j + 1)

    for k, point in enumerate(df_path['points'].unique()):
        df_point = df_path[(df_path['rel_levels'] == j) & (df_path['points'] == point)]
        means = df_point.groupby(['time_steps'])['logits_path'].mean().reset_index()
        errors = df_point.groupby(['time_steps'])['logits_path'].std().reset_index()
        errors['logits_path'] /= np.sqrt(NUM_JOBS)
        ax.errorbar(means['time_steps'] - 1, means['logits_path'], yerr=errors['logits_path'], label=point, fmt='o-', color=colors[k], ecolor=colors[k], elinewidth=1, capsize=0)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlim((-1.2, 2.2))
    ax.set_ylim((-8, 14))
    ax.set_title(rel_labels[j], pad=15)

    if j == 0:
        ax.set_ylabel(r'$\Delta$ Decision logit')
    else:
        ax.set_yticks([])

    if j == num_panels // 2:
        ax.set_xlabel('Time from querying a state')

ax.legend(title='Reward', bbox_to_anchor=(1, 0.5), loc='center left', frameon=False, fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(exp_path, f'p_logit_all.pdf'), bbox_inches='tight')





"""
All Q values
"""

fig = plt.figure(figsize = (9.5, 2.9))

df_path = pd.concat([data_jobid['df_q_values_path'] for data_jobid in data], keys=range(len(data)), names=['jobid']).reset_index(level=0)

colors = ['#F5191D', '#E97000', '#E79912', '#E9BA20', '#C1C88D', '#8ABD94', '#4CAFA1', '#3B99B1']
rel_labels = ['Self', 'Parent', 'Parent$^2$', 'Parent$^3$', 'Parent$^4$']

max_rel_level = int(df_path['rel_levels'].max())
num_panels = max_rel_level + 1

for j in range(num_panels):
    ax = plt.subplot(1, num_panels, j + 1)

    for k, point in enumerate(df_path['points'].unique()):
        df_point = df_path[(df_path['rel_levels'] == j) & (df_path['points'] == point)]
        means = df_point.groupby(['time_steps'])['q_values_path'].mean().reset_index()
        errors = df_point.groupby(['time_steps'])['q_values_path'].std().reset_index()
        errors['q_values_path'] /= np.sqrt(NUM_JOBS)
        ax.errorbar(means['time_steps'] - 1, means['q_values_path'], yerr=errors['q_values_path'],
                    label=point, fmt='o-', color=colors[k], ecolor=colors[k], elinewidth=1, capsize=0)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlim((-1.2, 2.2))
    ax.set_ylim((-8, 14))
    ax.set_title(rel_labels[j], pad=15)

    if j == 0:
        ax.set_ylabel(r'$\Delta$ Q value')
    else:
        ax.set_yticks([])

    if j == num_panels // 2:
        ax.set_xlabel('Time from querying a state')

ax.legend(title='Reward', bbox_to_anchor=(1, 0.5), loc='center left', frameon=False, fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(exp_path, f'p_q_values_all.pdf'), bbox_inches='tight')