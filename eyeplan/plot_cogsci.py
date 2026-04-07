import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.formula.api as smf
import seaborn as sns
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
    exp_path = os.path.join(args.path, f'logger_{args.cost}_{args.beta_e_final}_{args.kappa_squared}_{jobid}')

    with open(os.path.join(exp_path, 'logger_cog.pkl'), 'rb') as file:
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
Fixation by depth
"""

fixation_counts = np.array([data_jobid['fixation_count'] for data_jobid in data])
fixation_counts_by_depth = np.array([data_jobid['fixation_counts_by_depth'] for data_jobid in data])

colors = ['#016981', '#00868E', '#00A195', '#0BB795', '#71C897', '#A3D2A0']
means = fixation_counts_by_depth.mean(axis = 0)
errors = fixation_counts_by_depth.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)

print(means, errors)

# plt.figure(figsize = (2.4, 2.6))
plt.figure(figsize = (2.75, 2.8))
plt.bar(x = np.arange(len(means)), height = means, color = colors, yerr = errors, capsize = 0)  # Adjust cap size for error bars)
plt.axhline(y = fixation_counts.mean() / args.num_nodes, color = 'k', linestyle = '--', linewidth = 1, zorder = 0)
plt.xticks(range(len(means)))
plt.ylim((0, 1.8))
plt.xlabel('Depth')
plt.title('Agent')
plt.ylabel('Queries per node')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_fixation_by_depth.pdf'), bbox_inches = 'tight')





"""
Fixation by type
"""

fixation_proportions_by_type = np.array([data_jobid['fixation_proportions_by_type'] for data_jobid in data])
means = fixation_proportions_by_type.mean(axis = 0)
errors = fixation_proportions_by_type.std(axis = 0, ddof = 1) / np.sqrt(NUM_JOBS)

G = nx.DiGraph()
for parent, children in {0: [1, 2], 1: [3, 4]}.items():
    for child in children:
        G.add_edge(parent, child)
pos = {
    0: [0.9, 0],
    1: [0, 0.9],
    2: [1.8, 0.9],
    3: [-0.9, 1.8],
    4: [0.9, 1.8],
}
labels = {
    0: f"{round(means[1] * 100)}%",
    1: '',
    2: f"{round(means[2] * 100)}%",
    3: f"{round(means[0] * 100 / 2)}%",
    4: f"{round(means[0] * 100 / 2)}%",
}
plt.figure(figsize = (3, 3))
nx.draw(
    G = G,
    pos = pos,
    labels = labels,
    with_labels = True,
    node_size = 2500,
    node_color = ['#59B5D9', '#CCCCCC', '#F2AE30', '#F25244', '#F25244'],
    edgecolors = 'black',
    linewidths = 3,
    width = 3,
    font_size = 16,
    arrowsize = 15,
    font_color = 'white',
    # font_weight = 'bold',
)
plt.xlim((-1.5, 2.5))
plt.ylim((-0.5, 2.5))
# plt.title('Agent')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_fixation_by_type.pdf'), bbox_inches = 'tight')


plt.figure(figsize = (3.5, 2.8))
plt.bar(x = np.arange(len(means[:-1])), height = means[:-1], yerr = errors[:-1], capsize = 0)
# plt.axhline(y = 1, color = 'k', linestyle = '--', linewidth = 1, zorder = 0)
plt.xticks(range(len(means[:-1])), labels = ['Child', 'Parent', 'Sibling', 'Other'])
# plt.ylim((0, 1.8))
plt.xlabel('Fixation type')
plt.ylabel('Proportion')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_fixation_type_bar.pdf'), bbox_inches = 'tight')





"""
Continuation policy
"""

plt.figure(figsize = (2.5, 2.5))
df = pd.concat([data_jobid['df_continuation'] for data_jobid in data])
df_filtered = df[df['points'] != 0]
df_grouped = df_filtered.groupby(['jobid', 'points'])['continuations'].mean().reset_index()
df_summary = df_grouped.groupby(['points'])['continuations'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])
plt.errorbar(df_summary['points'], df_summary['mean'], yerr = df_summary['se'], fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
sns.regplot(data = df_grouped, x = 'points', y = 'continuations', ci = False, scatter = False, color = 'k')
plt.axhline(y = 0.5, color = 'k', linestyle = '--', linewidth = 1)
plt.xticks(range(-5, 6, 5))
plt.ylim((0, 1))
plt.xlabel('Point')
plt.ylabel('Fixate a child')
plt.title('Agent')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_continuation_by_point.pdf'), bbox_inches = 'tight')

plt.figure(figsize = (2.5, 2.5))
df = pd.concat([data_jobid['df_continuation'] for data_jobid in data])
df_grouped = df.groupby(['jobid', 'depths'])['continuations'].mean().reset_index()
df_summary = df_grouped.groupby(['depths'])['continuations'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])
plt.errorbar(df_summary['depths'], df_summary['mean'], yerr = df_summary['se'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.axhline(y = 0.5, color = 'k', linestyle = '--', linewidth = 1)
plt.xticks(range(0, 5, 1))
plt.ylim((0, 1))
plt.xlabel('Depth')
plt.ylabel('Fixate a child')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_continuation_by_depth.pdf'), bbox_inches = 'tight')





"""
Exploitation policy
"""

plt.figure(figsize = (2.6, 2.8))
df = pd.concat([data_jobid['df_exploitation'] for data_jobid in data])
bins = [-12.5, -7.5, -3.5, 0, 3.5, 7.5, 12.5]
labels = [-12.5, -7, -2, 2, 7, 12.5]
df['group'] = pd.cut(df['relative_q_values'], bins = bins, labels = labels, include_lowest = False, right = False)
df_filtered = df[(df['relative_q_values'] >= -20) & (df['relative_q_values'] <= 20)]
# ---- aggregate for plotting ----
df_grouped = df_filtered.groupby(['jobid', 'group'])['child1_fixation_counts'].mean().reset_index()
df_summary = df_grouped.groupby(['group'])['child1_fixation_counts'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])

# df_summary.to_csv(os.path.join(exp_path, 'df_exploitation.csv'), index = False)

# ---- fit linear mixed model ----
# random intercepts per jobid; add "+ (relative_q_values|jobid)" if you also want random slopes
model = smf.mixedlm('child1_fixation_counts ~ relative_q_values', df_filtered, groups = df_filtered['jobid'])
result = model.fit(reml = False)
print(result.summary())
# ---- get fitted values for plotting ----
x_pred = np.linspace(-20, 20, 200)
df_pred = pd.DataFrame({'relative_q_values': x_pred})
df_pred["child1_fixation_counts"] = result.predict(df_pred)

# df_pred.to_csv(os.path.join(exp_path, 'df_exploitation_fit.csv'), index = False)

# ---- plotting ----
plt.errorbar(df_summary['group'], df_summary['mean'], yerr = df_summary['se'], fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.plot(df_pred['relative_q_values'], df_pred['child1_fixation_counts'], color = 'k', linewidth = 2)
plt.axhline(y = 0.5, color = 'k', linestyle = '--', linewidth = 1)
plt.xlim((-21, 21))
plt.ylim((0, 1))
plt.xlabel('Relative Q value')
plt.ylabel('Fixate child 1')
plt.title('Agent', pad = 15)
# plt.title(f'Noise = {args.kappa_squared:.1f}', pad = 15)
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_cog_exploitation.pdf'), bbox_inches = 'tight')


plt.figure(figsize = (3.5, 2.5))
colors = ['#2493BF', '#D94E4E', '#8E6AA6']
df_grouped = df_filtered.groupby(['jobid', 'group', 'q_groups'])['child1_fixation_counts'].mean().reset_index()
df_summary = df_grouped.groupby(['group', 'q_groups'])['child1_fixation_counts'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])
for i, q_group in enumerate(['child 1', 'child 2', 'both']):
    df_filtered = df_summary[df_summary['q_groups'] == q_group]
    plt.errorbar(df_filtered['group'], df_filtered['mean'], yerr = df_filtered['se'], fmt = 'o', label = q_group, color = colors[i], ecolor = colors[i], elinewidth = 1, capsize = 0)
    df_filtered = df[df['q_groups'] == q_group]
    # df_grouped = df_filtered.groupby(['jobid', 'relative_q_values'])['child1_fixation_counts'].mean().reset_index()
    sns.regplot(data = df_filtered, x = 'relative_q_values', y = 'child1_fixation_counts', ci = False, scatter = False, color = colors[i]) ######################## raw data
plt.axhline(y = 0.5, color = 'k', linestyle = '--', linewidth = 1)
plt.xlim((-21, 21))
plt.ylim((0, 1))
plt.xlabel('Relative action value')
plt.ylabel('Fixate child 1')
plt.title('Agent')
plt.legend(frameon = False, bbox_to_anchor = (1.05, 1), loc = 'upper left', fontsize = 'small')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_exploitation_by_type.pdf'), bbox_inches = 'tight')





"""
Exploration policy
"""

plt.figure(figsize = (2.5, 2.5))
df = pd.concat([data_jobid['df_exploration'] for data_jobid in data])
df_filtered = df[(df['relative_fixation_counts'] >= -3) & (df['relative_fixation_counts'] <= 3)]
df_grouped = df_filtered.groupby(['jobid', 'relative_fixation_counts'])['child1_fixation_counts'].mean().reset_index()
df_summary = df_grouped.groupby(['relative_fixation_counts'])['child1_fixation_counts'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])
plt.errorbar(df_summary['relative_fixation_counts'], df_summary['mean'], yerr = df_summary['se'], fmt = 'o-', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.xticks(range(-3, 4, 1))
plt.ylim((0, 1))
plt.xlabel('Relative # fixations')
plt.ylabel('Fixate child 1')
plt.title('Agent')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_exploration.pdf'), bbox_inches = 'tight')

plt.figure(figsize = (2.5, 2.5))
colors = ['#2493BF', '#D94E4E', '#8E6AA6']
df_grouped = df_filtered.groupby(['jobid', 'seen_groups', 'relative_fixation_counts'])['child1_fixation_counts'].mean().reset_index()
df_summary = df_grouped.groupby(['seen_groups', 'relative_fixation_counts'])['child1_fixation_counts'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])
for i, group in enumerate(['first', 'second', 'both']):
    df_filtered = df_summary[df_summary['seen_groups'] == group]
    plt.errorbar(df_filtered['relative_fixation_counts'], df_filtered['mean'], yerr = df_filtered['se'], fmt = 'o', color = colors[i], ecolor = colors[i], elinewidth = 1, capsize = 0)
for i, group in enumerate(['first', 'second', 'both']):
    df_filtered = df_grouped[df_grouped['seen_groups'] == group]
    sns.regplot(data = df_filtered, x = 'relative_fixation_counts', y = 'child1_fixation_counts', ci = False, scatter = False, color = colors[i])
plt.xticks(range(-3, 4, 1))
plt.ylim((0, 1))
plt.xlabel('Relative # fixations')
plt.ylabel('Fixate child 1')
plt.title('Agent')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_exploration_by_type.pdf'), bbox_inches = 'tight')





"""
Switching policy
"""

df = pd.concat([data_jobid['df_jump'] for data_jobid in data])

df_grouped = df.groupby(['jobid', 'jump_depths']).size().reset_index(name = 'counts')
df_grouped['proportions'] = df_grouped.groupby('jobid')['counts'].transform(lambda x: x / x.sum())
df_summary = df_grouped.groupby(['jump_depths'])['proportions'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])

df_grouped_baseline = df.groupby(['jobid', 'jump_depths_baseline']).size().reset_index(name = 'counts')
df_grouped_baseline['proportions'] = df_grouped_baseline.groupby('jobid')['counts'].transform(lambda x: x / x.sum())
df_summary_baseline = df_grouped_baseline.groupby(['jump_depths_baseline'])['proportions'].agg(['mean', 'std', 'count']).reset_index()

colors = ['#016981', '#00868E', '#00A195', '#0BB795', '#71C897', '#A3D2A0']
plt.figure(figsize = (2.6, 2.8))
bars = plt.bar(x = df_summary['jump_depths'], height = df_summary['mean'], color = colors)
plt.errorbar(df_summary['jump_depths'], df_summary['mean'], yerr = df_summary['se'], fmt = 'none', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
for j, bar in enumerate(bars):
    bar_height = df_summary_baseline['mean'][j] #bar.get_height()
    plt.hlines(y = bar_height, xmin = bar.get_x(), xmax = bar.get_x() + bar.get_width(), color = 'black', linestyle = (0, (6.5, 3.)), linewidth = 1)
plt.xticks(np.arange(0, 6))
plt.ylim((0, 0.6))
plt.xlabel('Depth')
plt.ylabel('Proportion of jumps')
plt.title('Agent', pad = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_jump.pdf'), bbox_inches = 'tight')


for i, seen in enumerate(['unseen', 'seen']):
    df_filtered = df[df['jump_seens'] == seen]

    df_grouped = df_filtered.groupby(['jobid', 'jump_depths']).size().reset_index(name = 'counts')
    df_grouped['proportions'] = df_grouped.groupby('jobid')['counts'].transform(lambda x: x / x.sum())
    df_summary = df_grouped.groupby(['jump_depths'])['proportions'].agg(['mean', 'std', 'count']).reset_index()
    df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])

    df_grouped_baseline = df_filtered.groupby(['jobid', 'jump_depths_baseline']).size().reset_index(name = 'counts')
    df_grouped_baseline['proportions'] = df_grouped_baseline.groupby('jobid')['counts'].transform(lambda x: x / x.sum())
    df_summary_baseline = df_grouped_baseline.groupby(['jump_depths_baseline'])['proportions'].agg(['mean', 'std', 'count']).reset_index()

    colors = ['#016981', '#00868E', '#00A195', '#0BB795', '#71C897', '#A3D2A0']
    plt.figure(figsize = (2.75, 2.5))
    bars = plt.bar(x = df_summary['jump_depths'], height = df_summary['mean'], color = colors)
    plt.errorbar(df_summary['jump_depths'], df_summary['mean'], yerr = df_summary['se'], fmt = 'none', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
    for j, bar in enumerate(bars):
        bar_height = df_summary_baseline['mean'][j] #bar.get_height()
        plt.hlines(y = bar_height, xmin = bar.get_x(), xmax = bar.get_x() + bar.get_width(), color = 'black', linestyle = (0, (6.5, 3.)), linewidth = 1)
    plt.xticks(np.arange(0, 6))
    plt.ylim((0, 0.6))
    plt.xlabel('Depth')
    plt.ylabel('Proportion of jumps')
    plt.title('Agent')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(exp_path, f'p_cog_jump_{seen}.pdf'), bbox_inches = 'tight')





"""
Evidence accumulation
"""

df = pd.concat([data_jobid['df_evidence_accumulation'] for data_jobid in data])
df_filtered = df[df['fixation_counts_chosen'] <= 3]
df_grouped = df_filtered.groupby(['jobid', 'fixation_counts_chosen', 'points'])['chosens'].mean().reset_index()
df_summary = df_grouped.groupby(['fixation_counts_chosen', 'points'])['chosens'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])

colors = ['#F5191D', '#E97000', '#E79912', '#E9BA20', '#C1C88D', '#8ABD94', '#4CAFA1', '#3B99B1']
plt.figure(figsize = (3, 2.5))
for i, point in enumerate(df_summary['points'].unique()):
    df_point = df_summary[df_summary['points'] == point]
    plt.errorbar(df_point['fixation_counts_chosen'], df_point['mean'], yerr = df_point['se'], fmt = 'o-', label = point, color = colors[i], ecolor = colors[i], elinewidth = 1, capsize = 0)
plt.xticks(np.arange(4))
plt.ylim((0, 1))
plt.xlabel('# fixations')
plt.ylabel('Chosen probability')
plt.title('Agent', pad = 15)
plt.legend(title = 'Point', bbox_to_anchor = (1.05, 1), loc = 'upper left', frameon = False, fontsize = 'small')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(exp_path, 'p_cog_evidence_accumulation.pdf'), bbox_inches = 'tight')





"""
Frontier fixation
"""

plt.figure(figsize = (2.6, 2.8))
df = pd.concat([data_jobid['df_frontier'] for data_jobid in data])
bins = [-100, -9.5, -4.5, 0, 4.5, 9.5, 100]
labels = [-13.5, -7.5, -2.5, 2.5, 7.5, 13.5]
df['group'] = pd.cut(df['cum_points'], bins = bins, labels = labels, include_lowest = False, right = False)
df_filtered = df[(df['cum_points'] != 0) & (df['cum_points'] >= -20) & (df['cum_points'] <= 20)]
# ---- aggregate for errorbar plot ----
df_grouped = df_filtered.groupby(['jobid', 'group'])['frontier_fixation_counts'].mean().reset_index()
df_summary = df_grouped.groupby(['group'])['frontier_fixation_counts'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])

# ---- fit linear mixed model ----
# random intercepts per jobid
model = smf.mixedlm('frontier_fixation_counts ~ cum_points', df_filtered, groups = df_filtered['jobid'])
result = model.fit(reml = False)
print(result.summary())
# ---- get fitted values for regression line ----
x_pred = np.linspace(-20, 20, 200)
df_pred = pd.DataFrame({'cum_points': x_pred})
df_pred['frontier_fixation_counts'] = result.predict(df_pred)

# ---- plotting ----
plt.errorbar(df_summary['group'], df_summary['mean'], yerr = df_summary['se'], fmt = 'o', color = 'black', ecolor = 'black', elinewidth = 1, capsize = 0)
plt.plot(df_pred['cum_points'], df_pred['frontier_fixation_counts'], color = 'k', linewidth = 2)
plt.ylim((0, 1))
plt.xlabel('Path value')
plt.ylabel('p(query frontier node)')
plt.title('Agent', pad = 15)
# plt.title(f'Cost = {args.cost * 8:.2f}', pad = 15)
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_cog_frontier.pdf'), bbox_inches = 'tight')




plt.figure(figsize=(2.6, 2.8))
df = pd.concat([data_jobid['df_refixation_q'] for data_jobid in data])
bins = [-100, -9.5, -4.5, 0, 4.5, 9.5, 100]
labels = [-13.5, -7.5, -2.5, 2.5, 7.5, 13.5]
df['group'] = pd.cut(df['q_values'], bins=bins, labels=labels, include_lowest=False, right=False)
df_filtered = df[(df['q_values'] >= -20) & (df['q_values'] <= 20)]
# ---- aggregate for errorbar plot ----
df_grouped = df_filtered.groupby(['jobid', 'group'])['refixation_counts'].mean().reset_index()
df_summary = df_grouped.groupby(['group'])['refixation_counts'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])

# ---- fit linear mixed model ----
# random intercepts per jobid
model = smf.mixedlm('refixation_counts ~ q_values', df_filtered, groups=df_filtered['jobid'])
result = model.fit(reml=False)
print(result.summary())
# ---- get fitted values for regression line ----
x_pred = np.linspace(-20, 20, 200)
df_pred = pd.DataFrame({'q_values': x_pred})
df_pred['refixation_counts'] = result.predict(df_pred)

# ---- plotting ----
plt.errorbar(df_summary['group'], df_summary['mean'], yerr=df_summary['se'], fmt='o', color='black', ecolor='black', elinewidth=1, capsize=0)
plt.plot(df_pred['q_values'], df_pred['refixation_counts'], color='k', linewidth=2)
plt.ylim((0, 0.4))
plt.xlabel('Q value')
plt.ylabel('p(re-query node)')
plt.title('Agent', pad = 15)
# plt.title(f'Noise = {args.kappa_squared:.1f}', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(exp_path, 'p_cog_refixation_q.pdf'), bbox_inches='tight')