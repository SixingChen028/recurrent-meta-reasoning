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
COST = 0.03
BETA_E_FINAL = 0.04
NOISES = [0.0, 0.3, 0.6]




"""
Set environment
"""

# parse args
parser = ArgParser()
args = parser.args





"""
Read data
"""

data = [[] for _ in range(len(NOISES))]
for i, noise in enumerate(NOISES):
    for jobid in range(NUM_JOBS):

        exp_path = os.path.join(args.path, f'logger_{COST}_{BETA_E_FINAL}_{noise}_{jobid}')

        with open(os.path.join(exp_path, 'logger_cog.pkl'), 'rb') as file:
            data_noise_jobid = pickle.load(file)

        data[i].append(data_noise_jobid)

print(data[0][0].keys())





"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure_noise')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Performance by noise
"""

cum_point_proportions = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        cum_point_proportions[j, i] = data[i][j]['cum_point_proportion']
means = cum_point_proportions.mean(axis = 0)
errors = cum_point_proportions.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.5))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.ylim((0, 1))
plt.xlabel(r'$\kappa^2$')
plt.ylabel('Reward proportion')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_performance.pdf'), bbox_inches = 'tight')





"""
Refixation by noise
"""

refixation_proportions = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        refixation_proportions[j, i] = data[i][j]['refixation_proportion']
means = refixation_proportions.mean(axis = 0)
errors = refixation_proportions.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.5))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 0.3))
plt.xlabel(r'$\kappa^2$')
plt.ylabel('Refixation proportion')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_refixation.pdf'), bbox_inches = 'tight')





"""
Fixation count
"""

fixation_counts = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        fixation_counts[j, i] = data[i][j]['fixation_count']
means = fixation_counts.mean(axis = 0)
errors = fixation_counts.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.5))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 0.3))
plt.xlabel(r'$\kappa^2$')
plt.ylabel('# of fixations')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_fixation_count.pdf'), bbox_inches = 'tight')





"""
Decision count
"""

decision_counts = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        decision_counts[j, i] = data[i][j]['decision_count']
means = decision_counts.mean(axis = 0)
errors = decision_counts.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.5))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 0.3))
plt.xlabel(r'$\kappa^2$')
plt.ylabel('# of decisions')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_decision_count.pdf'), bbox_inches = 'tight')





"""
Fixation by type
"""

fixation_proportions_by_type = np.zeros((4, NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        for k, type in enumerate(['child', 'parent', 'sibling', 'others']):
            fixation_proportions_by_type[k, j, i] = data[i][j]['fixation_proportions_by_type'][k]
means = fixation_proportions_by_type.mean(axis = 1)
errors = fixation_proportions_by_type.std(axis = 1, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.5))
for k, type in enumerate(['child', 'parent', 'sibling', 'others']):
    plt.errorbar(NOISES, means[k], yerr = errors[k], fmt = 'o-', elinewidth = 1, capsize = 0, label = type)
# plt.ylim((0, 0.3))
plt.xlabel(r'$\kappa^2$')
plt.ylabel('Proportion')
plt.legend(frameon = False, bbox_to_anchor = (1.05, 1), loc = 'upper left', fontsize = 'small')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_fixation_by_type.pdf'), bbox_inches = 'tight')

child_fixation_proportions = fixation_proportions_by_type[0, :, :]
means = child_fixation_proportions.mean(axis = 0)
errors = child_fixation_proportions.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (3, 2.6))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'k', elinewidth = 1, capsize = 0)
# plt.xlim((-0.05, 0.65))
# plt.xticks([0, 0.2, 0.4, 0.6])
# plt.yticks([0.2, 0.25, 0.3])
plt.xlabel('Noise')
plt.ylabel('Proportion to children')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_child_fixation.pdf'), bbox_inches = 'tight')

colors = ['#5B4D7A', '#7A5B91', '#A26FA1', '#C88BAA', '#E3A7A0', '#F4C59A', '#F8DDA5']
plt.figure(figsize = (3, 2.6))
# plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
bars = plt.bar(x = NOISES, height = means, width = 0.08, color = colors)
plt.errorbar(NOISES, means, yerr = errors, fmt = 'none', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.xticks([0, 0.2, 0.4, 0.6])
# plt.ylim((0, 0.35))
plt.xlabel('Noise')
plt.ylabel('Proportion to children')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_child_fixation_hist.pdf'), bbox_inches = 'tight')





"""
Continuation policy
"""

k_continuation_by_point = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        par = data[i][j]['par_continuation_by_point']
        k_continuation_by_point[j, i] = par[0]
means = k_continuation_by_point.mean(axis = 0)
errors = k_continuation_by_point.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.5))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 0.3))
plt.xlabel(r'$\kappa^2$')
plt.ylabel(r'$k_{\text{continuation} \sim \text{point}}$')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_continuation_by_point.pdf'), bbox_inches = 'tight')

k_continuation_by_depth = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        par = data[i][j]['par_continuation_by_depth']
        k_continuation_by_depth[j, i] = par[0]
means = k_continuation_by_depth.mean(axis = 0)
errors = k_continuation_by_depth.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.5))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 0.3))
plt.xlabel(r'$\kappa^2$')
plt.ylabel(r'$k_{\text{continuation} \sim \text{depth}}$')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_continuation_by_depth.pdf'), bbox_inches = 'tight')





"""
Exploitation policy
"""

k_exploitation = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        par = data[i][j]['par_exploitation']
        k_exploitation[j, i] = par[0]
means = k_exploitation.mean(axis = 0)
errors = k_exploitation.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.5, 2.6))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth = 1, zorder = 0)
plt.xlim((0 - 0.02, 0.2 + 0.02))
# plt.ylim((-0.002, 0.012))
# plt.xticks([0, 0.2, 0.4, 0.6])
plt.yticks([0.00, 0.01])
# plt.xlabel(r'$\kappa^2$')
plt.xlabel('Noise')
plt.ylabel(r'$k_{\text{child} \sim \text{action value}}$')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_exploitation.pdf'), bbox_inches = 'tight')

print(k_exploitation)

k_exploitation_by_type = np.zeros((3, NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        par = data[i][j]['par_exploitation_by_type']
        for k in range(3):
            k_exploitation_by_type[k, j, i] = par[k, 0]
means = k_exploitation_by_type.mean(axis = 1)
errors = k_exploitation_by_type.std(axis = 1, ddof = 1) / np.sqrt(NUM_JOBS)
colors = ['#2493BF', '#D94E4E', '#8E6AA6']
plt.figure(figsize = (2.8, 2.5))
for k, type in enumerate(['child 1', 'child 2', 'both']):
    plt.errorbar(NOISES, means[k], yerr = errors[k], fmt = 'o-', color = colors[k], ecolor = colors[k], elinewidth = 1, capsize = 0, label = type)
# plt.ylim((0, 0.3))
plt.xlabel(r'$\kappa^2$')
plt.ylabel(r'$k_\text{exploitation}$')
plt.legend(frameon = False, bbox_to_anchor = (1.05, 1), loc = 'upper left', fontsize = 'small')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_exploitation_by_type.pdf'), bbox_inches = 'tight')





"""
Frontier fixation
"""

k_frontier = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        par = data[i][j]['par_frontier']
        k_frontier[j, i] = par[0]
means = k_frontier.mean(axis = 0)
errors = k_frontier.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.8, 2.5))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.ylim((0, 0.3))
plt.xlabel(r'$\kappa^2$')
plt.ylabel(r'$k_\text{frontier}$')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_frontier.pdf'), bbox_inches = 'tight')





"""
Switch policy
"""

depth_1_jumps = np.zeros((NUM_JOBS, len(NOISES)))
depth_1_jumps_baseline = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        df = data[i][j]['df_jump']
        # df_filtered = df[df['jump_seens'] == 'seen'] ###########
        df_grouped = df.groupby(['jobid', 'jump_depths']).size().reset_index(name = 'counts')
        df_grouped['proportions'] = df_grouped.groupby('jobid')['counts'].transform(lambda x: x / x.sum())

        df_grouped_baseline = df.groupby(['jobid', 'jump_depths_baseline']).size().reset_index(name = 'counts')
        df_grouped_baseline['proportions'] = df_grouped_baseline.groupby('jobid')['counts'].transform(lambda x: x / x.sum())

        depth_1_jumps[j, i] = df_grouped['proportions'][1]
        depth_1_jumps_baseline[j, i] = df_grouped_baseline['proportions'][1]

means = depth_1_jumps.mean(axis = 0)
errors = depth_1_jumps.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)

means_baseline = depth_1_jumps_baseline.mean(axis = 0)
errors_baseline = depth_1_jumps_baseline.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)

colors = ['#5B4D7A', '#7A5B91', '#A26FA1', '#C88BAA', '#E3A7A0', '#F4C59A', '#F8DDA5']
plt.figure(figsize = (3, 2.6))
# plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
bars = plt.bar(x = NOISES, height = means, width = 0.08, color = colors)
plt.errorbar(NOISES, means, yerr = errors, fmt = 'none', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
for j, bar in enumerate(bars):
    bar_height = means_baseline[j] #bar.get_height()
    plt.hlines(y = bar_height, xmin = bar.get_x(), xmax = bar.get_x() + bar.get_width(), color = 'black', linestyle = (0, (6.5, 3.)), linewidth = 1)
# plt.errorbar(NOISES, means_baseline, yerr = errors_baseline, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
# plt.xticks([0, 0.2, 0.4, 0.6])
# plt.ylim((0, 0.42))
plt.xlabel('Noise')
plt.ylabel('Proportion to depth-1')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_jump.pdf'), bbox_inches = 'tight')
    




"""
Exploitation policy
"""

k_refixation = np.zeros((NUM_JOBS, len(NOISES)))
for i, noise in enumerate(NOISES):
    for j, jobid in enumerate(range(NUM_JOBS)):
        par = data[i][j]['par_refixation_q']
        k_refixation[j, i] = par[0]
means = k_refixation.mean(axis = 0)
errors = k_refixation.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)
plt.figure(figsize = (2.55, 2.6))
plt.errorbar(NOISES, means, yerr = errors, fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth = 1, zorder = 0)
plt.xlim((0 - 0.06, 0.6 + 0.06))
# plt.ylim((-0.002, 0.012))
plt.xticks([0, 0.6])
# plt.yticks([0.00, 0.01])
# plt.xlabel(r'$\kappa^2$')
plt.xlabel('Noise')
plt.ylabel(r'$k_{\text{re-query} \sim \text{Q value}}$')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_noise_refixation_q.pdf'), bbox_inches = 'tight')