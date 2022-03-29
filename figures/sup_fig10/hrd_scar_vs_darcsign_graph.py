# -*- coding: utf-8 -*-
# @author: Elie
#%% ==========================================================
# Import libraries set library params
# ============================================================

# Libraries
import pandas as pd
import numpy as np
from numpy import std, mean, sqrt
from scipy.stats import linregress
import os

#plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator

pd.options.mode.chained_assignment = None

#%% ==========================================================
# scarhrd vs probability plot
# ============================================================
def make_graph(df, goi, other_val, fs):

	scatsize = 10
	prob_column = f"{goi}_prob_of_true"
	#color schemes
	#define all the colors
	color_list = list(sns.color_palette().as_hex())
	blue = color_list[0] #DRwt
	orange = color_list[1] #atm
	green = color_list[2] #cdk12
	red = color_list[3] #brca2
	purple = color_list[4] #mmr

	brc = df.query('(label == "BRCA2d")')
	cdk = df.query('(label == "CDK12d")')
	atm = df.query('(label == "ATMd")')
	mmr = df.query('(label == "MMRd")')
	DRwt = df.query('(label == "DRwt")')
	all_brcap = df.query('(label != "BRCA2d")')

	def cohen_d(x,y):
		nx = len(x)
		ny = len(y)
		dof = nx + ny - 2
		return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

	fig, ax = plt.subplots(figsize=(3.25,2.5), ncols=2, nrows=2, gridspec_kw={'height_ratios':[1,5],'width_ratios':[5,1]})

	#first do scatter plot
	ax[1,0].scatter(x=DRwt[prob_column], y=DRwt[other_val], color=blue, s=scatsize, alpha=0.6, zorder=10, linewidth=0, label="DRwt")
	ax[1,0].scatter(x=atm[prob_column], y=atm[other_val], color=orange, s=scatsize, alpha=0.7, zorder=20, linewidth=0, label="ATMd")
	ax[1,0].scatter(x=cdk[prob_column], y=cdk[other_val], color=green, s=scatsize, alpha=0.7, zorder=20, linewidth=0, label="CDK12d")
	ax[1,0].scatter(mmr[prob_column], mmr[other_val], color=purple, s=scatsize, alpha=0.8, zorder=30, linewidth=0, label="MMRd")
	ax[1,0].scatter(x=brc[prob_column], y=brc[other_val], color=red, s=scatsize, alpha=0.9, zorder=40, linewidth=0, label="BRCA2d")
	ax[1,0].tick_params(axis='x', which='both', length=3, pad=2, labelsize=fs)
	ax[1,0].tick_params(axis='y', which='both', length=3, pad=2, labelsize=fs)
	ax[1,0].grid(which='both', axis='both', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-100)
	ax[1,0].set_ylabel(other_val, fontsize=fs, labelpad=4, verticalalignment='center')
	# ax[1,0].yaxis.set_label_coords(-0.12, 0.5)
	ax[1,0].set_xlabel(f"Probability of {goi}", fontsize=fs, labelpad=5, verticalalignment='center')

	slope_, intercept_, r_value, p_value, std_err_ = linregress(df[prob_column].values, df[other_val].values)

	ax[1,0].text(0.5, 0, f"R={r_value.round(2)}", fontsize=fs, color="k", va="bottom", ha="center", ma='center', alpha=0.9, transform=ax[1,0].transAxes)

	# do ax[0,0] top plot
	ax[0,0].get_shared_x_axes().join(ax[0,0], ax[1,0])
	ax[0,0].xaxis.set_ticklabels([])
	ax[0,0].get_xaxis().set_visible(False)
	ax[0,0].yaxis.set_ticklabels([])
	ax[0,0].get_yaxis().set_visible(False)
	sns.kdeplot(all_brcap[prob_column], color="grey", shade=True, ax=ax[0,0])
	sns.kdeplot(brc[prob_column], color=red, shade=True, ax=ax[0,0])
	# test conditions
	c0 = all_brcap[prob_column]
	c1 = brc[prob_column]
	cd = cohen_d(c1,c0)
	ax[0,0].text(0.5, 0.15, f"Cohen's D: {round(cd, 2)}", fontsize=fs, color="k", va="bottom", ha="center", ma='center', alpha=0.9, transform=ax[0,0].transAxes)

	ax[1,1].get_shared_y_axes().join(ax[1,1], ax[1,0])
	ax[1,1].xaxis.set_ticklabels([])
	ax[1,1].get_xaxis().set_visible(False)
	ax[1,1].yaxis.set_ticklabels([])
	ax[1,1].get_yaxis().set_visible(False)
	sns.kdeplot(y=all_brcap[otherval], color="grey", shade=True, ax=ax[1,1])
	sns.kdeplot(y=brc[otherval], color=red, shade=True, ax=ax[1,1])
	c0 = all_brcap[otherval]
	c1 = brc[otherval]
	cd = cohen_d(c1,c0)
	ax[1,1].text(0.01, 0.01, f"Cohen's D:\n{round(cd, 2)}", fontsize=fs, color="k", va="bottom", ha="left", ma='center', alpha=0.9, transform=ax[1,1].transAxes)

	ax[1,0].yaxis.set_label_coords(-0.12, 0.5)
	ax[1,0].set_xlim(-0.001,1.01)
	ax[1,0].set_ylim(-0.001,df[other_val].max()+1)
	ax[1,0].yaxis.set_major_locator(MaxNLocator(integer=True))
	ax[0,1].xaxis.set_ticklabels([])
	ax[0,1].get_xaxis().set_visible(False)
	ax[0,1].yaxis.set_ticklabels([])
	ax[0,1].get_yaxis().set_visible(False)

	sns.despine(ax=ax[0,0], top=True, right=True, left=True, bottom=True)
	sns.despine(ax=ax[1,1], top=True, right=True, left=True, bottom=True)
	sns.despine(ax=ax[0,1], top=True, right=True, left=True, bottom=True)
	sns.despine(ax=ax[1,0], top=True, right=True, left=False, bottom=False)
	fig.subplots_adjust(hspace=0.01, wspace=0.01, left=0.11, right=0.98, top=0.99, bottom=0.11)
	return fig,ax

def plot_legend_scatter(figdir, fs=6, ss=6):
	color_list = list(sns.color_palette().as_hex())
	blue = color_list[0] #DRwt
	orange = color_list[1] #atm
	green = color_list[2] #cdk12
	red = color_list[3] #brca2
	purple = color_list[4] #mmr
	fig, ax = plt.subplots(figsize=(3.1,0.15))
	handles = []
	handles.append(mlines.Line2D([], [], color=blue, markeredgecolor=blue, marker='o', lw=0, markersize=ss, label='DRwt'))
	handles.append(mlines.Line2D([], [], color=orange, markeredgecolor=orange, marker='o', lw=0, markersize=ss, label='ATMd'))
	handles.append(mlines.Line2D([], [], color=green, markeredgecolor=green, marker='o', lw=0, markersize=ss, label='CDK12d'))
	handles.append(mlines.Line2D([], [], color=red, markeredgecolor=red, marker='o', lw=0, markersize=ss, label='BRCA2d'))
	handles.append(mlines.Line2D([], [], color=purple, markeredgecolor=purple, marker='o', lw=0, markersize=ss, label='MMRd'))
	ax.axis('off')
	plt.grid(b=False, which='both')
	plt.legend(handles=handles,loc='center', edgecolor='0.5', fancybox=True, frameon=False, facecolor='white', ncol=5, fontsize=fs, labelspacing=0.1, handletextpad=-0.2, columnspacing=0.5)
	fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0, wspace=0)
	plt.savefig(os.path.join(figdir, "scarhrd_vs_darcsign_scatter_legend.pdf"))
	plt.savefig(os.path.join(figdir, "scarhrd_vs_darcsign_scatter_legend.png"), dpi=500, transparent=False, facecolor="w")

def plot_legend_kde(figdir, fs=6, ss=7):
	color_list = list(sns.color_palette().as_hex())
	red = color_list[3] #brca2
	# purple = color_list[4] #mmr
	fig, ax = plt.subplots(figsize=(1.5,0.15))
	handles = []
	handles.append(mlines.Line2D([], [], color=red, markeredgecolor=red, marker='s', lw=0, markersize=ss, label='BRCA2d'))
	handles.append(mlines.Line2D([], [], color="grey", markeredgecolor="grey", marker='s', lw=0, markersize=ss, label='BRCA2p'))
	ax.axis('off')
	plt.grid(b=False, which='both')
	plt.legend(handles=handles,loc='center', edgecolor='0.5', fancybox=True, frameon=False, facecolor='white', ncol=2, fontsize=fs, labelspacing=0.1, handletextpad=-0.2, columnspacing=0.5)
	fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0, wspace=0)
	plt.savefig(os.path.join(figdir, "scarhrd_vs_darcsign_kde_legend.pdf"))
	plt.savefig(os.path.join(figdir, "scarhrd_vs_darcsign_kde_legend.png"), dpi=500, transparent=False, facecolor="w")

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "sup_fig10")
datadir = os.path.join(rootdir, "data")
# cohort_data = os.path.join(datadir, "cohort.tsv")
shrd_path = os.path.join(datadir, "scarhrd.tsv")
prob_path = os.path.join(datadir, "darcsign_probability.tsv")

shrd = pd.read_csv(shrd_path, sep='\t', low_memory=False)
prob = pd.read_csv(prob_path, sep='\t', low_memory=False)
df = pd.merge(prob, shrd, how="left", on="sample")

#%%
otherval = "HRD"
fig, ax = make_graph(df, "BRCA2d", otherval, 6)
plt.savefig(os.path.join(figdir, f"{otherval}_vs_darcsign.pdf"))
plt.savefig(os.path.join(figdir, f"{otherval}_vs_darcsign.png"), dpi=500, transparent=False, facecolor="w")
# plt.close()

otherval = "TelomericAI"
fig, ax = make_graph(df, "BRCA2d", otherval, 6)
plt.savefig(os.path.join(figdir, f"{otherval}_vs_darcsign.pdf"))
plt.savefig(os.path.join(figdir, f"{otherval}_vs_darcsign.png"), dpi=500, transparent=False, facecolor="w")
# plt.close()

otherval = "LST"
fig, ax = make_graph(df, "BRCA2d", otherval, 6)
plt.savefig(os.path.join(figdir, f"{otherval}_vs_darcsign.pdf"))
plt.savefig(os.path.join(figdir, f"{otherval}_vs_darcsign.png"), dpi=500, transparent=False, facecolor="w")
# plt.close()

otherval = "HRDsum"
fig, ax = make_graph(df, "BRCA2d", otherval, 6)
plt.savefig(os.path.join(figdir, f"{otherval}_vs_darcsign.pdf"))
plt.savefig(os.path.join(figdir, f"{otherval}_vs_darcsign.png"), dpi=500, transparent=False, facecolor="w")
# plt.close()

plot_legend_scatter(figdir)
plot_legend_kde(figdir)
#%%

