# -*- coding: utf-8 -*-
# @author: Elie
#%% ==========================================================
# Import libraries set library params
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "sup_fig8")
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
prob_path = os.path.join(datadir, "darcsign_probability.tsv")

sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False).query('(cancer == "PC")')
prob = pd.read_csv(prob_path, sep='\t', low_memory=False)
common_cols = list(np.intersect1d(sample_labels.columns, prob.columns))
df = pd.merge(sample_labels, prob, how="left", on=common_cols).reset_index(drop=True)

#%% define some variables for all plots
#color schemes
#define all the colors
color_list = list(sns.color_palette().as_hex())
blue = color_list[0] #drp
orange = color_list[1] #atm
green = color_list[2] #cdk12
red = color_list[3] #brca2
purple = color_list[4] #mmr

#seperate out dfs for order of plotting
brc = df.query('(label == "BRCA2d")')
cdk = df.query('(label == "CDK12d")')
mmr = df.query('(label == "MMRd")')
drp = df.query('(label == "DRwt")')
fs=6 #fontsize
scatsize = 9
alpha=0.6

#%%BRCA2d plot
goi = "BRCA2d"
prob_column = f"{goi}_prob_of_true"

fig, ax = plt.subplots(figsize=(2.4,2))
ax.scatter(x=drp[prob_column], y=drp["cellularity_sequenza"], color=blue, s=scatsize, alpha=alpha, zorder=10, linewidth=0, label="DRwt")
ax.scatter(x=cdk[prob_column], y=cdk["cellularity_sequenza"], color=green, s=scatsize, alpha=alpha, zorder=20, linewidth=0, label="CDK12d")
ax.scatter(mmr[prob_column], mmr["cellularity_sequenza"], color=purple, s=scatsize, alpha=alpha, zorder=30, linewidth=0, label="MMRd")
ax.scatter(x=brc[prob_column], y=brc["cellularity_sequenza"], color=red, s=scatsize, alpha=alpha, zorder=40, linewidth=0, label="BRCA2d")

ax.tick_params(axis='x', which='both', length=3, pad=2, labelsize=fs)
ax.tick_params(axis='y', which='both', length=3, pad=2, labelsize=fs)
ax.grid(b=False, which='both', axis='both', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-100)
ax.set_ylabel("Tumor cellularity", fontsize=fs, labelpad=4, verticalalignment='center')
ax.yaxis.set_label_coords(-0.12, 0.5)
ax.set_xlabel(f"Probability of {goi}", fontsize=fs, labelpad=4, verticalalignment='center')
ax.set_xlim(-0.001,1.01)
ax.set_ylim(0,1.01)
sns.despine(ax=ax, top=True, right=True)

slope, intercept, r_value, p_value, std_err = linregress(df[prob_column].values, df["cellularity_sequenza"].values)
rval = r_value**2
ax.text(0.81, 0.98, f"R={r_value.round(3)}", fontsize=fs, color="k", va="top", ha="left", alpha=0.9)
ax.text(0.81, 0.93, f"P={p_value.round(3)}", fontsize=fs, color="k", va="top", ha="left", alpha=0.9)

fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.125, hspace=0.05, wspace=0)
plt.savefig(os.path.join(figdir, f"{goi}_cellularity_vs_prob.pdf"))
plt.savefig(os.path.join(figdir, f"{goi}_cellularity_vs_prob.png"), dpi=500, transparent=False, facecolor="w")
# plt.close()
#%% CDK12d plot
goi = "CDK12d"
prob_column = f"{goi}_prob_of_true"

fig, ax = plt.subplots(figsize=(2.4,2))
ax.scatter(x=drp[prob_column], y=drp["cellularity_sequenza"], color=blue, s=scatsize, alpha=alpha, zorder=10, linewidth=0, label="DRwt")
ax.scatter(mmr[prob_column], mmr["cellularity_sequenza"], color=purple, s=scatsize, alpha=alpha, zorder=30, linewidth=0, label="MMRd")
ax.scatter(x=brc[prob_column], y=brc["cellularity_sequenza"], color=red, s=scatsize, alpha=alpha, zorder=40, linewidth=0, label="BRCA2d")
ax.scatter(x=cdk[prob_column], y=cdk["cellularity_sequenza"], color=green, s=scatsize, alpha=alpha, zorder=20, linewidth=0, label="CDK12d")

ax.tick_params(axis='x', which='both', length=3, pad=2, labelsize=fs)
ax.tick_params(axis='y', which='both', length=3, pad=2, labelsize=fs)
ax.grid(b=False, which='both', axis='both', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-100)
ax.set_ylabel("Tumor cellularity", fontsize=fs, labelpad=4, verticalalignment='center')
ax.yaxis.set_label_coords(-0.12, 0.5)
ax.set_xlabel(f"Probability of {goi}", fontsize=fs, labelpad=4, verticalalignment='center')
ax.set_xlim(-0.001,1.01)
ax.set_ylim(0,1.01)
sns.despine(ax=ax, top=True, right=True)

slope, intercept, r_value, p_value, std_err = linregress(df[prob_column].values, df["cellularity_sequenza"].values)
rval = r_value**2
ax.text(0.81, 0.98, f"R={r_value.round(3)}", fontsize=fs, color="k", va="top", ha="left", alpha=0.9)
ax.text(0.81, 0.93, f"P={p_value.round(3)}", fontsize=fs, color="k", va="top", ha="left", alpha=0.9)

fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.125, hspace=0.05, wspace=0)
plt.savefig(os.path.join(figdir, f"{goi}_cellularity_vs_prob.pdf"))
plt.savefig(os.path.join(figdir, f"{goi}_cellularity_vs_prob.png"), dpi=500, transparent=False, facecolor="w")
# plt.close()
#%% MMRd plot
goi = "MMRd"
prob_column = f"{goi}_prob_of_true"

fig, ax = plt.subplots(figsize=(2.4,2))
ax.scatter(x=drp[prob_column], y=drp["cellularity_sequenza"], color=blue, s=scatsize, alpha=alpha, zorder=10, linewidth=0, label="DRp")
ax.scatter(x=cdk[prob_column], y=cdk["cellularity_sequenza"], color=green, s=scatsize, alpha=alpha, zorder=20, linewidth=0, label="CDK12d")
ax.scatter(x=brc[prob_column], y=brc["cellularity_sequenza"], color=red, s=scatsize, alpha=alpha, zorder=40, linewidth=0, label="BRCA2d")
ax.scatter(mmr[prob_column], mmr["cellularity_sequenza"], color=purple, s=scatsize, alpha=alpha, zorder=30, linewidth=0, label="MMRd")

ax.tick_params(axis='x', which='both', length=3, pad=2, labelsize=fs)
ax.tick_params(axis='y', which='both', length=3, pad=2, labelsize=fs)
ax.grid(b=False, which='both', axis='both', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-100)
ax.set_ylabel("Tumor cellularity", fontsize=fs, labelpad=4, verticalalignment='center')
ax.yaxis.set_label_coords(-0.12, 0.5)
ax.set_xlabel(f"Probability of {goi}", fontsize=fs, labelpad=4, verticalalignment='center')
ax.set_xlim(-0.001,1.01)
ax.set_ylim(0,1.01)
sns.despine(ax=ax, top=True, right=True)

slope, intercept, r_value, p_value, std_err = linregress(df[prob_column].values, df["cellularity_sequenza"].values)
rval = r_value**2
ax.text(0.81, 0.93, f"P={p_value.round(3)}", fontsize=fs, color="k", va="top", ha="left", alpha=0.9)
ax.text(0.81, 0.98, f"R={r_value.round(3)}", fontsize=fs, color="k", va="top", ha="left", alpha=0.9)

fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.125, hspace=0.05, wspace=0)
plt.savefig(os.path.join(figdir, f"{goi}_cellularity_vs_prob.pdf"))
plt.savefig(os.path.join(figdir, f"{goi}_cellularity_vs_prob.png"), dpi=500, transparent=False, facecolor="w")
# plt.close()

#%% legend
fig, ax = plt.subplots(figsize=(2,0.1))
ax.axis('off')
plt.grid(b=False, which='both')
plt.scatter([], [], c=blue, alpha=alpha, s=scatsize, label="DRwt", marker='o', linewidth=0)
plt.scatter([], [], c=red, alpha=alpha, s=scatsize, label="BRCA2d", marker='o', linewidth=0)
plt.scatter([], [], c=green, alpha=alpha, s=scatsize, label="CDK12d", marker='o', linewidth=0)
plt.scatter([], [], c=purple, alpha=alpha, s=scatsize, label="MMRd", marker='o', linewidth=0)
plt.legend(loc='center', ncol=4, borderaxespad=0., fontsize=6, labelspacing=0, columnspacing=0.7, handletextpad=0, borderpad=0.2, handlelength=1.5, framealpha=0, markerscale=2)
plt.savefig(os.path.join(figdir, "gene_legend.pdf"))
plt.savefig(os.path.join(figdir, "gene_legend.png"), dpi=500, transparent=True, facecolor="w")
# plt.close()

#%%
