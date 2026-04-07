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
# plt.rcParams['axes.spines.right'] = False
# plt.rcParams['axes.spines.top'] = False
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

    with open(os.path.join(exp_path, 'logger_backup.pkl'), 'rb') as file:
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
Parent's logit change given revealed node and it's sibling
"""

import matplotlib.colors as mcolors

df = pd.concat([data_jobid['df_sib'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)

pivot = (df
         .groupby(['sibling_value', 'revealed_reward'])['delta_logit_parent']
         .mean()
         .unstack('revealed_reward')) # rows = sibling_value, cols = revealed_reward

fig, ax = plt.subplots(figsize = (3.4, 3))

vmax = 10
vmax = np.ceil(vmax) # round up for a clean colorbar
norm = mcolors.TwoSlopeNorm(vmin = -vmax, vcenter = 0, vmax = vmax)
cmap = 'RdBu_r'

im = ax.imshow(
    pivot.values,
    aspect='auto',
    origin='lower',
    cmap=cmap,
    norm=norm,
)

# axes labels (discrete ticks)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([str(int(v)) for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([str(int(v)) for v in pivot.index])

ax.set_xlabel('Revealed reward')
ax.set_ylabel('Sibling value')
ax.set_title('Agent', pad = 10)

cbar = fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
cbar.set_label(r'$\Delta$ Decision logit of parent', rotation = 270, labelpad = 10, va = 'center', fontsize = 13)
cbar.set_ticks([-8, -4, 0, 4, 8])

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_backup_heatmap.pdf'), bbox_inches = 'tight')





"""
Grandparent's logit change given revealed node and parent's sibling
"""

df = pd.concat([data_jobid['df_sib_grand'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)

pivot = (df
         .groupby(['par_sibling_q', 'revealed_reward'])['delta_logit_grandpar']
         .mean()
         .unstack('revealed_reward'))

fig, ax = plt.subplots(figsize = (3.4, 3))

vmax = 10
vmax = np.ceil(vmax) # round up for a clean colorbar
norm = mcolors.TwoSlopeNorm(vmin = -vmax, vcenter = 0, vmax = vmax)
cmap = 'RdBu_r'

im = ax.imshow(
    pivot.values,
    aspect='auto',
    origin='lower',
    cmap=cmap,
    norm=norm,
)

# axes labels (discrete ticks)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([str(int(v)) for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([str(int(v)) for v in pivot.index])

# ax.set_xlabel(r'$\Delta$ Q value of parent')
ax.set_xlabel('Revealed reward')
ax.set_ylabel("Pibling value")
ax.set_title('Agent', pad = 10)

cbar = fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
cbar.set_label(r'$\Delta$ Decision logit of parent$^2$', rotation = 270, labelpad = 10, va = 'center', fontsize = 13)
cbar.set_ticks([-8, -4, 0, 4, 8])

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_backup_grandpar_heatmap.pdf'), bbox_inches = 'tight')





"""
Grandgrandparent's logit change given grandparent's Q value change and grandparent's sibling value
"""

df = pd.concat([data_jobid['df_sib_grandgrand'] for data_jobid in data], keys = range(len(data)), names = ['jobid']).reset_index(level = 0)

pivot = (df
         .groupby(['grandpar_sibling_q', 'revealed_reward'])['delta_logit_grandgrandpar']
         .mean()
         .unstack('revealed_reward'))

fig, ax = plt.subplots(figsize = (3.4, 3))

vmax = 10
vmax = np.ceil(vmax) # round up for a clean colorbar
norm = mcolors.TwoSlopeNorm(vmin = -vmax, vcenter = 0, vmax = vmax)
cmap = 'RdBu_r'

im = ax.imshow(
    pivot.values,
    aspect='auto',
    origin='lower',
    cmap=cmap,
    norm=norm,
)

# axes labels (discrete ticks)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([str(int(v)) for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([str(int(v)) for v in pivot.index])

# ax.set_xlabel(r'$\Delta$ Q value of parent$^2$')
ax.set_xlabel('Revealed reward')
ax.set_ylabel('Grandpibling value')
ax.set_title('Agent', pad = 10)

cbar = fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
cbar.set_label(r'$\Delta$ Decision logit of parent$^3$', rotation = 270, labelpad = 10, va = 'center', fontsize = 13)
cbar.set_ticks([-8, -4, 0, 4, 8]) 

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_backup_grandgrandpar_heatmap.pdf'), bbox_inches = 'tight')



