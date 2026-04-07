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

data_2arm = []
for jobid in range(NUM_JOBS):
    exp_path = os.path.join(args.path, f'logger_{2}_{jobid}')

    with open(os.path.join(exp_path, 'logger_decoding.pkl'), 'rb') as file:
        data_jobid = pickle.load(file)

    data_2arm.append(data_jobid)

data_3arm = []
for jobid in range(NUM_JOBS):
    exp_path = os.path.join(args.path, f'logger_{3}_{jobid}')

    with open(os.path.join(exp_path, 'logger_decoding.pkl'), 'rb') as file:
        data_jobid = pickle.load(file)

    data_3arm.append(data_jobid)

# print(data[0].keys())




"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)



plt.figure(figsize = (5, 2.8)) # (4,5, 2,5)



"""
Posterior mean decoding
"""

r2s_mean_2arm = np.array([data_jobid['r2s_mean'] for data_jobid in data_2arm])
mean_2arm = np.mean(r2s_mean_2arm)
error_2arm = np.std(r2s_mean_2arm) / np.sqrt(NUM_JOBS)

baseline_r2s_mean_2arm = np.array([data_jobid['baseline_r2s_mean'] for data_jobid in data_2arm])
baseline_2arm = np.mean(baseline_r2s_mean_2arm)

r2s_mean_3arm = np.array([data_jobid['r2s_mean'] for data_jobid in data_3arm])
mean_3arm = np.mean(r2s_mean_3arm)
error_3arm = np.std(r2s_mean_3arm) / np.sqrt(NUM_JOBS)

baseline_r2s_mean_3arm = np.array([data_jobid['baseline_r2s_mean'] for data_jobid in data_3arm])
baseline_3arm = np.mean(baseline_r2s_mean_3arm)

means = np.array([mean_2arm, mean_3arm])
errors = np.array([error_2arm, error_3arm])
baselines = np.array([baseline_2arm, baseline_3arm])

plt.subplot(1, 3, 1)
bars = plt.bar(x = [2, 3], height = means, yerr = errors, capsize = 0, color = 'lightblue')
for j, bar in enumerate(bars):
    bar_height = baselines[j]
    plt.hlines(y = bar_height, xmin = bar.get_x(), xmax = bar.get_x() + bar.get_width(), color = 'black', linestyle = (0, (4, 3.)), linewidth = 1)
plt.xlim((1.4, 3.6))
plt.ylim((-0.05, 1.05))
plt.xticks([2, 3])
plt.xlabel('Task')
plt.ylabel(r'$r^2$')
plt.title('Means', pad = 15)





"""
Posterior precision decoding
"""

r2s_precision_2arm = np.array([data_jobid['r2s_precision'] for data_jobid in data_2arm])
mean_2arm = np.mean(r2s_precision_2arm)
error_2arm = np.std(r2s_precision_2arm) / np.sqrt(NUM_JOBS)

baseline_r2s_precision_2arm = np.array([data_jobid['baseline_r2s_precision'] for data_jobid in data_2arm])
baseline_2arm = np.mean(baseline_r2s_precision_2arm)

r2s_precision_3arm = np.array([data_jobid['r2s_precision'] for data_jobid in data_3arm])
mean_3arm = np.mean(r2s_precision_3arm)
error_3arm = np.std(r2s_precision_3arm) / np.sqrt(NUM_JOBS)

baseline_r2s_precision_3arm = np.array([data_jobid['baseline_r2s_precision'] for data_jobid in data_3arm])
baseline_3arm = np.mean(baseline_r2s_precision_3arm)

means = np.array([mean_2arm, mean_3arm])
errors = np.array([error_2arm, error_3arm])
baselines = np.array([baseline_2arm, baseline_3arm])

plt.subplot(1, 3, 2)
bars = plt.bar(x = [2, 3], height = means, yerr = errors, capsize = 0, color = 'lightblue')
for j, bar in enumerate(bars):
    bar_height = baselines[j]
    plt.hlines(y = bar_height, xmin = bar.get_x(), xmax = bar.get_x() + bar.get_width(), color = 'black', linestyle = (0, (4, 3.)), linewidth = 1)
plt.xlim((1.4, 3.6))
plt.ylim((-0.05, 1.05))
plt.xticks([2, 3])
plt.xlabel('Task')
plt.ylabel(r'$r^2$')
plt.title('Precisions', pad = 15)


"""
Item decoding
"""

scores_item_2arm = np.array([data_jobid['scores_item'] for data_jobid in data_2arm])
mean_2arm = np.mean(scores_item_2arm)
error_2arm = np.std(scores_item_2arm) / np.sqrt(NUM_JOBS)

baseline_scores_item_2arm = np.array([data_jobid['baseline_scores_item'] for data_jobid in data_2arm])
baseline_2arm = np.mean(baseline_scores_item_2arm)

scores_item_3arm = np.array([data_jobid['scores_item'] for data_jobid in data_3arm])
mean_3arm = np.mean(scores_item_3arm)
error_3arm = np.std(scores_item_3arm) / np.sqrt(NUM_JOBS)

baseline_scores_item_3arm = np.array([data_jobid['baseline_scores_item'] for data_jobid in data_3arm])
baseline_3arm = np.mean(baseline_scores_item_3arm)

means = np.array([mean_2arm, mean_3arm])
errors = np.array([error_2arm, error_3arm])
baselines = np.array([baseline_2arm, baseline_3arm])

plt.subplot(1, 3, 3)
bars = plt.bar(x = [2, 3], height = means, yerr = errors, capsize = 0, color = 'lightblue')
for j, bar in enumerate(bars):
    bar_height = baselines[j]
    plt.hlines(y = bar_height, xmin = bar.get_x(), xmax = bar.get_x() + bar.get_width(), color = 'black', linestyle = (0, (4, 3.)), linewidth = 1)
plt.xlim((1.4, 3.6))
plt.ylim((-0.05, 1.05))
plt.xticks([2, 3])
plt.xlabel('Task')
plt.ylabel('Accuracy')
plt.title('Attended item', pad = 15)


plt.tight_layout()
plt.subplots_adjust(wspace = 1.5)
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_decoding.pdf'), bbox_inches = 'tight')