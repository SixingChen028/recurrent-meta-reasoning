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

    exp_path = os.path.join(args.path, f'logger_{jobid}')

    with open(os.path.join(exp_path, 'logger_rsa.pkl'), 'rb') as file:
        data_jobid = pickle.load(file)

    data.append(data_jobid)

# print(data[0].keys())





"""
Set plot path
"""

# set experiment path
exp_path = os.path.join(args.path, f'figure')
if not os.path.exists(exp_path):
    os.makedirs(exp_path)





"""
N vs. N RSA
"""

matrix_n = np.zeros((NUM_JOBS, 3, 3))
for jobid in range(NUM_JOBS):
    data_jobid = data[jobid]
    matrix_n[jobid] = data_jobid['matrix_n']

matrix_n = matrix_n.mean(axis = 0)

plt.figure(figsize = (3, 3))
sc = plt.imshow(matrix_n.T, origin = 'lower', vmin = 0.2, vmax = 1.0)
plt.xlabel(r'Time, Start $N$')
plt.ylabel(r'Time, Start $N$')
plt.xticks(ticks = [0, 1, 2], labels = [1, 2, 3])
plt.yticks(ticks = [0, 1, 2], labels = [1, 2, 3])
plt.title('Agent', pad = 15)
cbar = plt.colorbar(sc, shrink = 0.82)
cbar.set_label(r'Correlation coefficient', rotation = 270, labelpad = 10, va = 'center')
plt.savefig(os.path.join(exp_path, 'p_rsa_n.pdf'), bbox_inches = 'tight')





"""
N vs. N-1 RSA
"""

matrix_n_minus_1 = np.zeros((NUM_JOBS, 3, 3))
for jobid in range(NUM_JOBS):
    data_jobid = data[jobid]
    matrix_n_minus_1[jobid] = data_jobid['matrix_n_minus_1']

print(matrix_n_minus_1)

matrix_n_minus_1 = matrix_n_minus_1.mean(axis = 0)

plt.figure(figsize = (3, 3))
sc = plt.imshow(matrix_n_minus_1.T, origin = 'lower', vmax = 0.4)
plt.xlabel(r'Time, Start $N$')
plt.ylabel(r'Time, Start $N - 1$')
plt.xticks(ticks = [0, 1, 2], labels = [1, 2, 3])
plt.yticks(ticks = [0, 1, 2], labels = [1, 2, 3])
plt.title('Agent', pad = 15)
cbar = plt.colorbar(sc, shrink = 0.82)
cbar.set_label(r'Correlation coefficient', rotation = 270, labelpad = 10, va = 'center')
plt.savefig(os.path.join(exp_path, 'p_rsa_n_minus_1.pdf'), bbox_inches = 'tight')