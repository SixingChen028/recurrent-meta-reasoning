import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
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
COSTS = [0.02, 0.03, 0.04]
BETA_E_FINAL = 0.04
NOISE = 0.0



"""
Set environment
"""

# parse args
parser = ArgParser()
args = parser.args





"""
Read data
"""

data = [[] for _ in range(len(COSTS))]
for i, cost in enumerate(COSTS):
    for jobid in range(NUM_JOBS):
        exp_path = os.path.join(args.path, f'logger_{cost}_{BETA_E_FINAL}_{NOISE}_{jobid}')

        with open(os.path.join(exp_path, 'logger_cog.pkl'), 'rb') as file:
            data_noise_jobid = pickle.load(file)

        data[i].append(data_noise_jobid)

print(data[0][0].keys())





"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure_cost')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Frontier fixation
"""

k_frontier = np.zeros((NUM_JOBS, len(COSTS)))
for i, cost in enumerate(COSTS):
    for j, jobid in enumerate(range(NUM_JOBS)):
        par = data[i][j]['par_frontier']
        k_frontier[j, i] = par[0]
means = k_frontier.mean(axis = 0)
errors = k_frontier.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.5, 2.6))
plt.errorbar(np.array(COSTS) * 8, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth = 1, zorder = 0)
plt.xlim((0.16 - 0.02, 0.32 + 0.02))
# plt.ylim((-0.001, 0.013))
plt.xticks([0.16, 0.32])
plt.yticks([0, 0.01])
plt.xlabel(r'Cost')
plt.ylabel(r'$k_{\text{frontier} \sim \text{path value}}$')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cost_frontier.pdf'), bbox_inches = 'tight')




