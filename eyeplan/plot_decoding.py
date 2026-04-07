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

    with open(os.path.join(exp_path, 'logger_decoding.pkl'), 'rb') as file:
        data_jobid = pickle.load(file)

    data.append(data_jobid)

print(data[0].keys())





"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure_{args.cost}_{args.beta_e_final}_{args.kappa_squared}')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
Load decoding errors
"""

r2s_point = np.zeros(NUM_JOBS)
for j, jobid in enumerate(range(NUM_JOBS)):
    r2s_point[j] = np.mean(data[j]['r2s_point_cv'])
means_point = r2s_point.mean()
errors_point = r2s_point.std(ddof = 1) / np.sqrt(NUM_JOBS)

r2s_cum_point = np.zeros(NUM_JOBS)
for j, jobid in enumerate(range(NUM_JOBS)):
    r2s_cum_point[j] = np.mean(data[j]['r2s_cum_point_cv'])
means_cum_point = r2s_cum_point.mean()
errors_cum_point = r2s_cum_point.std(ddof = 1) / np.sqrt(NUM_JOBS)

r2s_q_value = np.zeros(NUM_JOBS)
for j, jobid in enumerate(range(NUM_JOBS)):
    r2s_q_value[j] = np.mean(data[j]['r2s_q_value_cv'])
means_q_value = r2s_q_value.mean()
errors_q_value = r2s_q_value.std(ddof = 1) / np.sqrt(NUM_JOBS)

means = [means_point, means_cum_point, means_q_value]
errors = [errors_point, errors_cum_point, errors_q_value]





"""
Plot errors
"""

plt.figure(figsize = (3.1, 2.8)) # (3.1, 2.8)
bars = plt.bar(x = ['Reward', 'Path value', 'Q value'], height = means, width = 0.5, color = 'lightblue', yerr = errors, ecolor = 'black', capsize = 0)
for j, bar in enumerate(bars):
    bar_height = 0.
    plt.hlines(y = bar_height, xmin = bar.get_x(), xmax = bar.get_x() + bar.get_width(), color = 'black', linestyle = (0, (4, 3.)), linewidth = 1)
plt.ylim((-0.05, 1.05))
plt.xlim((-0.5, 2.5))
plt.xticks(rotation = 30)
# plt.xlabel('Noise')
plt.ylabel(r'$r^2$')
# plt.title('Reward', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_decoding.pdf'), bbox_inches = 'tight')


