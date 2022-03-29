# -*- coding: utf-8 -*-
# @author: Elie
#%% ==========================================================
# Import libraries set library params
# ============================================================
import pandas as pd
import numpy as np
import os
#plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

# ================ Toggle variables =======================================
pd.options.mode.chained_assignment = None
pd.set_option('max_columns', None)
mpl.rcParams['savefig.transparent'] = "False"
mpl.rcParams['axes.facecolor'] = "white"
mpl.rcParams['figure.facecolor'] = "white"
output_dir = os.path.dirname(__file__)

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "fig3")
datadir = os.path.join(rootdir, "data")
probs_data = os.path.join(datadir, "darcsign_probability.tsv")
probs = pd.read_csv(probs_data, sep='\t', low_memory=False)

#%% ==========================================================
# define standard gene color pairs
# ============================================================

color_list = list(sns.color_palette().as_hex())
blue = color_list[0] #drwt
orange = color_list[1] #atm
green = color_list[2] #cdk12
red = color_list[3] #brca2
purple = color_list[4] #mmr
brown = color_list[5] # bladder
grey = "#383838"
# print(blue,orange,green,red,purple,brown)
# #1f77b4 #ff7f0e #2ca02c #d62728 #9467bd #8c564b #383838
# all_probs["alt_color"]
probs.loc[(probs["label"] == "ATMd"), 'alt_color'] = orange
probs.loc[(probs["label"] == "BRCA2d"), 'alt_color'] = red
probs.loc[(probs["label"] == "CDK12d"), 'alt_color'] = green
probs.loc[(probs["label"] == "MMRd"), 'alt_color'] = purple

#%% ==========================================================
# dfs sorted for gene and probability
# ============================================================
brca2_pos = probs.query('(label == "BRCA2d")').sort_values("BRCA2d_prob_of_true", ascending=False)
brca2_pos["colour"] = red
brca2_neg = probs.query('(label != "BRCA2d")').sort_values("BRCA2d_prob_of_true", ascending=False)
brca2_neg["colour"] = blue
brca2_graph = pd.concat([brca2_pos, brca2_neg]).reset_index(drop=True)

cdk12_pos = probs.query('(label == "CDK12d")').sort_values("CDK12d_prob_of_true", ascending=False)
cdk12_pos["colour"] = green
cdk12_neg = probs.query('(label != "CDK12d")').sort_values("CDK12d_prob_of_true", ascending=False)
cdk12_neg["colour"] = blue
cdk12_graph = pd.concat([cdk12_pos, cdk12_neg]).reset_index(drop=True)

mmrd_pos = probs.query('(label == "MMRd")').sort_values("MMRd_prob_of_true", ascending=False)
mmrd_pos["colour"] = purple
mmrd_neg = probs.query('(label != "MMRd")').sort_values("MMRd_prob_of_true", ascending=False)
mmrd_neg["colour"] = blue
mmrd_graph = pd.concat([mmrd_pos, mmrd_neg]).reset_index(drop=True)

#%% ==========================================================
# make the barplots
# ============================================================
#%%
def make_graph(df, gene):

	prob_col = f"{gene}_prob_of_true"
	annotation = df.query('(label != @gene) and (label != "DRwt")')
	fig, ax = plt.subplots(figsize=(3.2,1.5))
	ax.bar(x=df.index, height=df[prob_col], width=0.8, edgecolor=None, linewidth=0, color=df["colour"], zorder=10)
	ax.scatter(annotation.index-0.3, annotation[prob_col]+0.025, marker=r'$\downarrow$', color = annotation["alt_color"], s=12, zorder = 3, linewidths=0.2, edgecolors=None)

	ax.set_ylim(0,1)
	ax.set_xlim(df.index[0]-0.5,df.index[-1]+1)
	ax.grid(b=False, which='both', axis='y', color='0.4', linewidth=0.9, linestyle='dotted', zorder=0)
	ax.tick_params(axis='both', which="major", length=3, labelsize=5, pad=1, reset=False)
	ax.set_xticks([])
	ax.set_yticks([0.25, 0.50, 0.75])
	ax.set_xlabel("Samples", fontsize=5, labelpad=1)
	ax.set_ylabel("Probability", fontsize=5, horizontalalignment="center", labelpad=0.6)
	ax.yaxis.set_label_coords(-0.08, 0.5)
	sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
	fig.subplots_adjust(hspace=0.05, wspace=0.0, left=0.1, right=0.995, top=0.99, bottom=0.06)
	return fig, ax

gene="BRCA2d"
fig, ax = make_graph(brca2_graph, gene)
handles = []
handles.append(mlines.Line2D([], [], color=red, markeredgecolor=red, marker='s', lw=0, markersize=5, label='BRCA2d'))
handles.append(mlines.Line2D([], [], color=blue, markeredgecolor=blue, marker='s', lw=0, markersize=5, label='BRCA2wt'))
ax.legend(handles=handles,loc='upper left', edgecolor='0.5', frameon=False, ncol=2, fontsize=5, handletextpad=0.001, bbox_to_anchor=(0.45, 0.72), borderpad=0, columnspacing=0.9)
plt.savefig(os.path.join(figdir, f"{gene}_prob_barplot.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_prob_barplot.pdf"))
# plt.close()

gene="CDK12d"
fig, ax = make_graph(cdk12_graph, gene)
handles = []
handles.append(mlines.Line2D([], [], color=green, markeredgecolor=green, marker='s', lw=0, markersize=5, label='CDK12d'))
handles.append(mlines.Line2D([], [], color=blue, markeredgecolor=blue, marker='s', lw=0, markersize=5, label='CDK12wt'))
ax.legend(handles=handles,loc='upper left', edgecolor='0.5', frameon=False, ncol=2, fontsize=5, handletextpad=0.001, bbox_to_anchor=(0.45, 0.72), borderpad=0, columnspacing=0.9)
plt.savefig(os.path.join(figdir, f"{gene}_prob_barplot.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_prob_barplot.pdf"))
# plt.close()

gene="MMRd"
fig, ax = make_graph(mmrd_graph, gene)
handles = []
handles.append(mlines.Line2D([], [], color=purple, markeredgecolor=purple, marker='s', lw=0, markersize=5, label=f"MMRd   "))
handles.append(mlines.Line2D([], [], color=blue, markeredgecolor=blue, marker='s', lw=0, markersize=5, label=f'MMRwt'))
ax.legend(handles=handles,loc='upper left', edgecolor='0.5', frameon=False, ncol=2, fontsize=5, handletextpad=0.001, bbox_to_anchor=(0.45, 0.72), borderpad=0, columnspacing=0.9)
plt.savefig(os.path.join(figdir, f"{gene}_prob_barplot.png"), dpi=500)
plt.savefig(os.path.join(figdir, f"{gene}_prob_barplot.pdf"))
# plt.close()


#%% ==========================================================
#  annotation legends for the arrows above bars
# ============================================================

#brca2d plot arrows
handles = []
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=green, color=green, label='CDK12d'))
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=orange, color=orange, label='ATMd'))
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=purple, color=purple, label='MMRd'))
fig, ax = plt.subplots(figsize=(1,0.5))
ax.axis('off')
plt.grid(b=False, which='both')
plt.legend(handles=handles,loc='center left', edgecolor='0.5', frameon=False, facecolor='white', ncol=3, fontsize=5, labelspacing=0.3, handletextpad=-0.3, columnspacing=0.5, borderpad=0, borderaxespad=0)
plt.savefig(os.path.join(figdir, "brca2_annot.png"), dpi=500, transparent=True, bbox_inches = 'tight', pad_inches = 0)
plt.savefig(os.path.join(figdir, "brca2_annot.pdf"))
plt.close()
#cdk12d plot arrows
handles = []
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=red, color=red, label='BRCA2d'))
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=orange, color=orange, label='ATMd'))
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=purple, color=purple, label='MMRd'))
fig, ax = plt.subplots(figsize=(1,0.5))
ax.axis('off')
plt.grid(b=False, which='both')
plt.legend(handles=handles,loc='center left', edgecolor='0.5', frameon=False, facecolor='white', ncol=3, fontsize=5, labelspacing=0.3, handletextpad=-0.3, columnspacing=0.5, borderpad=0, borderaxespad=0)
plt.savefig(os.path.join(figdir, "cdk12_annot.png"), dpi=500, transparent=True, bbox_inches = 'tight', pad_inches = 0)
plt.savefig(os.path.join(figdir, "cdk12_annot.pdf"))
plt.close()
#mmrd plot arrows
handles = []
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=red, color=red, label='BRCA2d'))
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=green, color=green, label='CDK12d'))
handles.append(mlines.Line2D([], [], markeredgewidth=0.1, markersize=5, linewidth=0, marker=r'$\downarrow$', markeredgecolor=orange, color=orange, label='ATMd'))
fig, ax = plt.subplots(figsize=(1,0.5))
ax.axis('off')
plt.grid(b=False, which='both')
plt.legend(handles=handles,loc='center left', edgecolor='0.5', frameon=False, facecolor='white', ncol=3, fontsize=5, labelspacing=0.3, handletextpad=-0.3, columnspacing=0.5, borderpad=0, borderaxespad=0)
plt.savefig(os.path.join(figdir, "mmrd_annot.png"), dpi=500, transparent=True, bbox_inches = 'tight', pad_inches = 0)
plt.savefig(os.path.join(figdir, "mmrd_annot.pdf"))
plt.close()


# %%
