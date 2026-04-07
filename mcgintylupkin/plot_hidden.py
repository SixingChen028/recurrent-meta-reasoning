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
for jobid in range(5):

    exp_path = os.path.join(args.path, f'logger_{args.reward_std}_{args.stay_cost}_{args.switch_cost}_{jobid}')

    with open(os.path.join(exp_path, 'logger_hidden.pkl'), 'rb') as file:
        data_jobid = pickle.load(file)

    data.append(data_jobid)

# print(data[0].keys())




"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure_{args.reward_std}_{args.stay_cost}_{args.switch_cost}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Plot r2
"""

r2s = np.zeros((NUM_JOBS, 5, 2))
for jobid in range(NUM_JOBS):
    data_jobid = data[jobid]
    r2s[jobid] = data_jobid['r2s']

means = np.mean(r2s, axis = 0)
errors = np.nanstd(r2s, axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)

colors = ['#41b6c4', '#fd8f3c']
plt.figure(figsize = (3.2, 2.8))
plt.errorbar(np.arange(5) - 0.025, means[:, 0], yerr = errors[:, 0], fmt = 'o-', color = colors[0], ecolor = colors[0], elinewidth = 1, capsize = 0)
plt.errorbar(np.arange(5) + 0.025, means[:, 1], yerr = errors[:, 1], fmt = 'o-', color = colors[1], ecolor = colors[1], elinewidth = 1, capsize = 0)
plt.xlabel('Time from trial onset')
plt.title('Agent', pad = 15)
plt.ylabel(r'$r^2$')
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_hidden_r2.pdf'), bbox_inches = 'tight')


r2s_2d = np.zeros((NUM_JOBS, 5, 2))
for jobid in range(NUM_JOBS):
    data_jobid = data[jobid]
    r2s_2d[jobid] = data_jobid['r2s_2d']

means = np.mean(r2s_2d, axis = 0)
errors = np.nanstd(r2s_2d, axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)

colors = ['#41b6c4', '#fd8f3c']
plt.figure(figsize = (2.6, 3))
plt.errorbar(np.arange(5) - 0.025, means[:, 0], yerr = errors[:, 0], fmt = 'o-', color = colors[0], ecolor = colors[0], elinewidth = 1, capsize = 0)
plt.errorbar(np.arange(5) + 0.025, means[:, 1], yerr = errors[:, 1], fmt = 'o-', color = colors[1], ecolor = colors[1], elinewidth = 1, capsize = 0)
plt.ylim((-0.03, 0.65))
plt.xticks([0, 2, 4])
plt.yticks([0, 0.5])
plt.xlabel('Time from\ntrial onset')
plt.title('Agent', pad = 15)
plt.ylabel(r'$r^2$')
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_hidden_r2_2d.pdf'), bbox_inches = 'tight')





"""
Plot signal-noise ratio
"""

snrs = np.zeros((NUM_JOBS, 5, 2))
for jobid in range(NUM_JOBS):
    data_jobid = data[jobid]
    snrs[jobid, :, 0] = data_jobid['snr_val1']
    snrs[jobid, :, 1] = data_jobid['snr_val2']

means = np.mean(snrs, axis = 0)
errors = np.nanstd(snrs, axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)

colors = ['#41b6c4', '#fd8f3c']
plt.figure(figsize = (3.2, 2.8))
plt.errorbar(np.arange(5) - 0.025, means[:, 0], yerr = errors[:, 0], fmt = 'o-', color = colors[0], ecolor = colors[0], elinewidth = 1, capsize = 0)
plt.errorbar(np.arange(5) + 0.025, means[:, 1], yerr = errors[:, 1], fmt = 'o-', color = colors[1], ecolor = colors[1], elinewidth = 1, capsize = 0)
plt.xlabel('Time from trial start')
plt.ylabel('Signal-to-noise ratio')
plt.title('Agent', pad = 15)
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_hidden_snr.pdf'), bbox_inches = 'tight')





"""
Rotation angles
"""

enc1_projections = np.zeros((NUM_JOBS, 5, 2))
for jobid in range(NUM_JOBS):
    data_jobid = data[jobid]
    enc1_projections[jobid, :, :] = data_jobid['enc1_projections']

enc1_projections_normalized = enc1_projections / np.linalg.norm(enc1_projections, axis = -1, keepdims = True)
angles = np.degrees(np.arctan2(enc1_projections_normalized[:, :, 1], enc1_projections_normalized[:, :, 0]))
# angles = np.where(angles < 0, angles + 360, angles)
angles = angles % 360

means = np.mean(angles, axis = 0)
errors = np.nanstd(angles, axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)

plt.figure(figsize = (3.4, 2.8))
plt.errorbar(np.arange(1, 5), means[1:], yerr = errors[1:], fmt = 'o-', color = 'k', ecolor = 'k', elinewidth = 1, capsize = 0)
plt.xticks([1, 2, 3, 4])
plt.xlabel('Time from trial start')
plt.ylabel('Angle of 1st item\'s\nvalue gradient')
plt.title('Agent', pad = 15)
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_hidden_angle.pdf'), bbox_inches = 'tight')





enc1_projections = np.zeros((NUM_JOBS, 5, 2))
enc2_projections = np.zeros((NUM_JOBS, 5, 2))
for jobid in range(NUM_JOBS):
    data_jobid = data[jobid]
    enc1_projections[jobid, :, :] = data_jobid['enc1_projections']
    enc2_projections[jobid, :, :] = data_jobid['enc2_projections']

enc1_projections_normalized = enc1_projections / np.linalg.norm(enc1_projections, axis = -1, keepdims = True)
angles = np.degrees(np.arctan2(enc1_projections_normalized[:, :, 1], enc1_projections_normalized[:, :, 0]))
# angles = np.where(angles < 0, angles + 360, angles)
angles1 = angles % 360

enc2_projections_normalized = enc2_projections / np.linalg.norm(enc2_projections, axis = -1, keepdims = True)
angles2 = np.degrees(np.arctan2(enc2_projections_normalized[:, :, 1], enc2_projections_normalized[:, :, 0]))
# angles = np.where(angles < 0, angles + 360, angles)
angles2 = angles2 % 360

baseline = angles1[:, 1][:, np.newaxis].copy()
angles1 -= baseline
angles2 -= baseline

print(angles1)
print(angles2)

colors = ['#41b6c4', '#fd8f3c']
plt.figure(figsize = (2.55, 3))

means = np.mean(angles1, axis = 0)
errors = np.nanstd(angles1, axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.errorbar(np.arange(1, 5), means[1:], yerr = errors[1:], fmt = 'o-', color = colors[0], ecolor = colors[0], elinewidth = 1, capsize = 0)

means = np.mean(angles2, axis = 0)
errors = np.nanstd(angles2, axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.errorbar(np.arange(2, 5), means[2:], yerr = errors[2:], fmt = 'o-', color = colors[1], ecolor = colors[1], elinewidth = 1, capsize = 0)

plt.xticks([1, 2, 3, 4])
plt.xlabel('Time from\ntrial start')
plt.ylabel('Rotation angle')
plt.title('Agent', pad = 15)
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_hidden_angle.pdf'), bbox_inches = 'tight')