import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')

from modules import *

plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = fm.FontProperties(fname = './fonts/Arial.ttf')
fm.fontManager.addfont('./fonts/Arial.ttf') # registers the font
plt.rcParams['font.family'] = font.get_name()





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
        noise_free_obs = True, # no observation noise
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
data = load_data(os.path.join(exp_path, f'data_simulation_hidden.p'))
data = preprocess(data, args, merge_fixations = False)
num_trials = len(data['action_seqs'])
print('Keys:', data.keys())
print(num_trials)





"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Data gathering
"""

hiddens = []
values = []
time_steps = []
trial_indices = []

for i in range(num_trials):
    length_ep, values_ep, fixation_seq_ep, decision_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'decisions', 'hidden_seqs'
    )

    if length_ep > 1 and length_ep < args.t_max:
        hiddens.append(np.array(hidden_seq_ep))
        values.append(values_ep)
        time_steps.append([_ for _ in range(len(hidden_seq_ep))])
        trial_indices.append([i for _ in range(len(hidden_seq_ep))])

hiddens = np.concatenate(hiddens)
values = np.stack(values)
time_steps = np.concatenate(time_steps)
trial_indices = np.concatenate(trial_indices)

print(hiddens.shape)
print(values.shape)
print(time_steps.shape)
print(trial_indices.shape)





"""
PCA
"""

from sklearn.decomposition import PCA

# fit PCA
pca = PCA(n_components = 5)
pca.fit(hiddens)




"""
PCA: posterior mean colored by current attended item
"""

t_plot = 9

colors = ['#dd4d03', '#2676b8']

plt.figure(figsize = (2.9, 2.5))

for i in range(25000):
    length_ep, values_ep, fixation_seq_ep, hidden_seq_ep, posterior_means_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'hidden_seqs', 'posterior_means_seqs'
    )

    if values_ep[0] == 4 and values_ep[1] == 4 and length_ep > t_plot:

        hidden_seq_pca_ep = pca.transform(hidden_seq_ep)

        colors_ep = [np.array([0.0, 0.0, 0.0, 1.0])]
        for fixation in fixation_seq_ep:
            colors_ep.append(colors[fixation])

        plt.plot(hidden_seq_pca_ep[:t_plot, 0], -hidden_seq_pca_ep[:t_plot, 2], color = 'black', alpha = 0.07, zorder = -1)
        sc = plt.scatter(hidden_seq_pca_ep[:t_plot, 0], -hidden_seq_pca_ep[:t_plot, 2], c = colors_ep[:t_plot], alpha = 1, s = 22, zorder = 3)

plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 3')
legend_elements = [
    Line2D([0], [0], marker = 'o', color = 'w', label = 'Item 1', markerfacecolor = '#2676b8', markersize = 6),
    Line2D([0], [0], marker = 'o', color = 'w', label = 'Item 2', markerfacecolor = '#dd4d03', markersize = 6)
]
plt.legend(handles = legend_elements, bbox_to_anchor = (0.85, 1), loc = 'upper left', frameon = False, handlelength = 0)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_hidden_item.pdf'), bbox_inches = 'tight')





"""
PCA: posterior mean colored by posterior mean of item 1
"""

t_plot = 9

plt.figure(figsize = (3.1, 2.5))

for i in range(25000):
    length_ep, values_ep, fixation_seq_ep, hidden_seq_ep, posterior_means_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'hidden_seqs', 'posterior_means_seqs'
    )

    if values_ep[0] == 4 and values_ep[1] == 4 and length_ep > t_plot:

        hidden_seq_pca_ep = pca.transform(hidden_seq_ep)
        
        colors_ep = np.array(posterior_means_seq_ep)[:, 1]
        colors_ep = np.insert(colors_ep, 0, 0)

        plt.plot(hidden_seq_pca_ep[:t_plot, 0], -hidden_seq_pca_ep[:t_plot, 2], color = 'black', alpha = 0.07, linewidth = 1.5, zorder = -1)
        sc = plt.scatter(hidden_seq_pca_ep[:t_plot, 0], -hidden_seq_pca_ep[:t_plot, 2], c = colors_ep[:t_plot], cmap = 'viridis', vmin = 0, vmax = 3.1, alpha = 1, s = 22, zorder = 3)

plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 3')
cbar = plt.colorbar(sc)
cbar.set_label(r'Posterior mean of item 1', rotation = 270, labelpad = 10, va = 'center')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_hidden_item1.pdf'), bbox_inches = 'tight')





"""
PCA: posterior mean colored by posterior mean of item 2
"""

t_plot = 9

plt.figure(figsize = (3.1, 2.5))

for i in range(25000):
    length_ep, values_ep, fixation_seq_ep, hidden_seq_ep, posterior_means_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'hidden_seqs', 'posterior_means_seqs'
    )

    if values_ep[0] == 4 and values_ep[1] == 4 and length_ep > t_plot:

        hidden_seq_pca_ep = pca.transform(hidden_seq_ep)
        
        colors_ep = np.array(posterior_means_seq_ep)[:, 0]
        colors_ep = np.insert(colors_ep, 0, 0)

        plt.plot(hidden_seq_pca_ep[:t_plot, 0], -hidden_seq_pca_ep[:t_plot, 2], color = 'black', alpha = 0.07, linewidth = 1.5, zorder = -1)
        sc = plt.scatter(hidden_seq_pca_ep[:t_plot, 0], -hidden_seq_pca_ep[:t_plot, 2], c = colors_ep[:t_plot], cmap = 'plasma', vmin = 0, vmax = 3.1, alpha = 1, s = 22, zorder = 3)

plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 3')
cbar = plt.colorbar(sc)
cbar.set_label(r'Posterior mean of item 2', rotation = 270, labelpad = 10, va = 'center')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_hidden_item2.pdf'), bbox_inches = 'tight')





"""
PCA: attended item
"""

plt.figure(figsize = (3.3, 2.5))

colors = ['#dd4d03', '#2676b8']

for i in range(500):
    length_ep, values_ep, fixation_seq_ep, hidden_seq_ep = pull(
        data, i, 'lengths', 'values', 'fixation_seqs', 'hidden_seqs'
    )

    if values_ep[0] > 0 and values_ep[1] > 0:

        hidden_seq_pca_ep = pca.transform(hidden_seq_ep)

        colors_ep = [np.array([0.0, 0.0, 0.0, 1.0])]
        for fixation in fixation_seq_ep:
            colors_ep.append(colors[fixation])

        plt.plot(hidden_seq_pca_ep[:, 0], hidden_seq_pca_ep[:, 1], color = 'black', alpha = 0.03, zorder = 0)
        plt.scatter(hidden_seq_pca_ep[:, 0], hidden_seq_pca_ep[:, 1], c = colors_ep, alpha = 1, s = 20, zorder = 20)

legend_elements = [
    Line2D([0], [0], marker = 'o', color = 'w', label = 'Item 1', markerfacecolor = '#2676b8', markersize = 6),
    Line2D([0], [0], marker = 'o', color = 'w', label = 'Item 2', markerfacecolor = '#dd4d03', markersize = 6)
]
plt.legend(handles = legend_elements, bbox_to_anchor = (1, 1), loc = 'upper left', frameon = False, handlelength = 0)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_hidden_attend.pdf'), bbox_inches = 'tight')