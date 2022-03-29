# -*- coding: utf-8 -*-
# @author: Elie
#%%
# Libraries
import pandas as pd
# from copy import copy, deepcopy
import numpy as np
from joblib import load, dump
from functools import partial, reduce
import datetime
import os
# import sys
#plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
# from matplotlib.patches import Rectangle
import umap
# import math
import xgboost
from xgboost import XGBClassifier
pd.options.mode.chained_assignment = None

#%% ==========================================================
# define these feature/headers here in case the headers 
# are out of order in input files (often the case)
# ============================================================

snv_categories = ["sample", 
				"A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T", 
				"C[C>A]A", "C[C>A]C", "C[C>A]G", "C[C>A]T", 
				"G[C>A]A", "G[C>A]C", "G[C>A]G", "G[C>A]T", 
				"T[C>A]A", "T[C>A]C", "T[C>A]G", "T[C>A]T", 
				"A[C>G]A", "A[C>G]C", "A[C>G]G", "A[C>G]T", 
				"C[C>G]A", "C[C>G]C", "C[C>G]G", "C[C>G]T", 
				"G[C>G]A", "G[C>G]C", "G[C>G]G", "G[C>G]T", 
				"T[C>G]A", "T[C>G]C", "T[C>G]G", "T[C>G]T", 
				"A[C>T]A", "A[C>T]C", "A[C>T]G", "A[C>T]T", 
				"C[C>T]A", "C[C>T]C", "C[C>T]G", "C[C>T]T", 
				"G[C>T]A", "G[C>T]C", "G[C>T]G", "G[C>T]T", 
				"T[C>T]A", "T[C>T]C", "T[C>T]G", "T[C>T]T", 
				"A[T>A]A", "A[T>A]C", "A[T>A]G", "A[T>A]T", 
				"C[T>A]A", "C[T>A]C", "C[T>A]G", "C[T>A]T", 
				"G[T>A]A", "G[T>A]C", "G[T>A]G", "G[T>A]T", 
				"T[T>A]A", "T[T>A]C", "T[T>A]G", "T[T>A]T", 
				"A[T>C]A", "A[T>C]C", "A[T>C]G", "A[T>C]T", 
				"C[T>C]A", "C[T>C]C", "C[T>C]G", "C[T>C]T", 
				"G[T>C]A", "G[T>C]C", "G[T>C]G", "G[T>C]T", 
				"T[T>C]A", "T[T>C]C", "T[T>C]G", "T[T>C]T", 
				"A[T>G]A", "A[T>G]C", "A[T>G]G", "A[T>G]T", 
				"C[T>G]A", "C[T>G]C", "C[T>G]G", "C[T>G]T", 
				"G[T>G]A", "G[T>G]C", "G[T>G]G", "G[T>G]T", 
				"T[T>G]A", "T[T>G]C", "T[T>G]G", "T[T>G]T"]

indel_categories = ["sample", 
				"1:Del:C:0", "1:Del:C:1", "1:Del:C:2", "1:Del:C:3", "1:Del:C:4", "1:Del:C:5", 
				"1:Del:T:0", "1:Del:T:1", "1:Del:T:2", "1:Del:T:3", "1:Del:T:4", "1:Del:T:5", 
				"1:Ins:C:0", "1:Ins:C:1", "1:Ins:C:2", "1:Ins:C:3", "1:Ins:C:4", "1:Ins:C:5", 
				"1:Ins:T:0", "1:Ins:T:1", "1:Ins:T:2", "1:Ins:T:3", "1:Ins:T:4", "1:Ins:T:5", 
				"2:Del:R:0", "2:Del:R:1", "2:Del:R:2", "2:Del:R:3", "2:Del:R:4", "2:Del:R:5", 
				"3:Del:R:0", "3:Del:R:1", "3:Del:R:2", "3:Del:R:3", "3:Del:R:4", "3:Del:R:5", 
				"4:Del:R:0", "4:Del:R:1", "4:Del:R:2", "4:Del:R:3", "4:Del:R:4", "4:Del:R:5", 
				"5:Del:R:0", "5:Del:R:1", "5:Del:R:2", "5:Del:R:3", "5:Del:R:4", "5:Del:R:5", 
				"2:Ins:R:0", "2:Ins:R:1", "2:Ins:R:2", "2:Ins:R:3", "2:Ins:R:4", "2:Ins:R:5", 
				"3:Ins:R:0", "3:Ins:R:1", "3:Ins:R:2", "3:Ins:R:3", "3:Ins:R:4", "3:Ins:R:5", 
				"4:Ins:R:0", "4:Ins:R:1", "4:Ins:R:2", "4:Ins:R:3", "4:Ins:R:4", "4:Ins:R:5", 
				"5:Ins:R:0", "5:Ins:R:1", "5:Ins:R:2", "5:Ins:R:3", "5:Ins:R:4", "5:Ins:R:5", 
				"2:Del:M:1", "3:Del:M:1", "3:Del:M:2", "4:Del:M:1", "4:Del:M:2", "4:Del:M:3", 
				"5:Del:M:1", "5:Del:M:2", "5:Del:M:3", "5:Del:M:4", "5:Del:M:5"]
				
cnv_categories = ["sample", 
				"BCper10mb_0", "BCper10mb_1", "BCper10mb_2", "BCper10mb_3", 
				"CN_0", "CN_1", "CN_2", "CN_3", "CN_4", "CN_5", "CN_6", "CN_7", "CN_8", 
				"CNCP_0", "CNCP_1", "CNCP_2", "CNCP_3", "CNCP_4", "CNCP_5", "CNCP_6", "CNCP_7", 
				"BCperCA_0", "BCperCA_1", "BCperCA_2", "BCperCA_3", "BCperCA_4", "BCperCA_5", 
				"SegSize_0", "SegSize_1", "SegSize_2", "SegSize_3", "SegSize_4", "SegSize_5", 
				"SegSize_6", "SegSize_7", "SegSize_8", "SegSize_9", "SegSize_10", 
				"CopyFraction_0", "CopyFraction_1", "CopyFraction_2", "CopyFraction_3", "CopyFraction_4", 
				"CopyFraction_5", "CopyFraction_6"]

#%% ==========================================================
# make concat sig dataframe  and umap functions. 
# differnt aesthetics for big / small plot
# ============================================================

def load_data(snv_counts_path, indel_counts_path, cnv_counts_path):
	df_snv = pd.read_csv(snv_counts_path, sep='\t', low_memory=False)
	df_snv = df_snv[snv_categories]
	df_snv["sample"] = df_snv["sample"].astype(str)

	df_indel = pd.read_csv(indel_counts_path, sep='\t', low_memory=False)
	df_indel = df_indel[indel_categories]
	df_indel["sample"] = df_indel["sample"].astype(str)

	df_cnv = pd.read_csv(cnv_counts_path, sep='\t', low_memory=False)
	df_cnv = df_cnv[cnv_categories]
	df_cnv["sample"] = df_cnv["sample"].astype(str)

	df_sigs = pd.merge(df_snv, df_indel, on="sample", how='left').fillna(0)
	df_sigs = pd.merge(df_sigs, df_cnv, on="sample", how='left').reset_index(drop=True)
	return df_sigs

def predict_bladder(modelpath, alldata, gene):
	print(f"start predicting {gene} deficiency at {str(datetime.datetime.now())}")
	alldatacopy = alldata.copy(deep=True)
	modelpath = os.path.expanduser(modelpath)

	features_list = snv_categories[1:] + indel_categories[1:] + cnv_categories[1:]
	X_data = alldatacopy[features_list]
	X_data.columns = X_data.columns.str.replace("[", "mm").str.replace("]", "nn").str.replace(">", "rr")
	alldatacopy['sample_labels'] = [1 if x == gene else 0 for x in alldatacopy['label']]
	Y_labels = alldatacopy["sample_labels"]
	xgbmodel = xgboost.XGBClassifier()
	xgbmodel.load_model(modelpath)
	prediction_prob = xgbmodel.predict_proba(X_data, ntree_limit=1000000)
	# prediction_prob[:,1]
	df_probs = pd.DataFrame(data={"labels":Y_labels.values, "prob_of_true": prediction_prob[:,1]})
	df_probs.index = Y_labels.index
	df_probs["sample"] = alldatacopy["sample"]
	prob_col = f"prob_of_{gene}"
	preds = df_probs.rename(columns={"prob_of_true": prob_col})
	preds = preds[["sample", prob_col]]
	print(f"finished predicting {gene} deficiency at {str(datetime.datetime.now())}")
	return preds

def barplot_aesthetics(prob_table, axis, gene_column, fs=6):
	axis.set_ylim(0,1)
	axis.set_xlim(prob_table.index[0]-0.5,prob_table.index[-1]+0.5)
	# axis.grid(axis='y', color='0.4', linewidth=0.9, linestyle='dotted', zorder=0)
	axis.grid(axis='y', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-100)
	# sample_xlabels = prob_table["sample"].values
	# axis.tick_params(axis='y', which="major", length=3, labelsize=fs, pad=1, reset=False)
	axis.tick_params(axis='y', which='major', length=3, pad=2, labelsize=fs)
	# axis.set_xticks(prob_table.index)
	# axis.set_xticklabels(sample_xlabels, rotation=45, fontsize=8, ha='right', va='top', rotation_mode="anchor")
	# axis.tick_params(axis='x', which="major", length=3, labelsize=8, pad=0, reset=False)
	axis.set_xticks([])
	axis.set_ylabel(f"Probability of\n{gene_column}", fontsize=fs, ha="center", ma="center", labelpad=4)
	axis.yaxis.set_label_coords(-0.09, 0.5)
	axis.set_yticks([0.2, 0.4, 0.6, 0.8])
	# ax.set_xlabel("")
	sns.despine(ax=axis, top=True, right=True, left=False, bottom=False)
	return axis

def barplot_legend(gened, genep, gened_color, axis, fs=6):
	handles = []
	handles.append(mlines.Line2D([], [], color=gened_color, markeredgecolor=gened_color, marker='s', lw=0, markersize=6, label=f"Predicted {gened}"))
	handles.append(mlines.Line2D([], [], color=blue, markeredgecolor=blue, marker='s', lw=0, markersize=6, label=f"Predicted {genep}"))
	axis.legend(handles=handles,loc='center left', edgecolor='0.5', frameon=False, ncol=2, fontsize=fs, handletextpad=-0.1, labelspacing=0.1, columnspacing=0.5,borderpad=0, borderaxespad=0, bbox_to_anchor=(0.2, 0.9))
	return axis

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "fig4")
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")

sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df_bc_only = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer != "PC")').reset_index(drop=True)

#%% ==========================================================
# predict bladder ccancer samples
# ============================================================

all_probabilites_list = []
gene = "BRCA2d"
modeldir = os.path.join(rootdir, "models")
gene_model = os.path.join(modeldir, f"{gene}.xgb_py37_xgboost_ml.model.json")
model_path = os.path.expanduser(gene_model)
predictions = predict_bladder(model_path, df_bc_only, gene)
all_probabilites_list.append(predictions)

gene = "CDK12d"
modeldir = os.path.join(rootdir, "models")
gene_model = os.path.join(modeldir, f"{gene}.xgb_py37_xgboost_ml.model.json")
model_path = os.path.expanduser(gene_model)
predictions = predict_bladder(model_path, df_bc_only, gene)
all_probabilites_list.append(predictions)

gene = "MMRd"
modeldir = os.path.join(rootdir, "models")
gene_model = os.path.join(modeldir, f"{gene}.xgb_py37_xgboost_ml.model.json")
model_path = os.path.expanduser(gene_model)
predictions = predict_bladder(model_path, df_bc_only, gene)
all_probabilites_list.append(predictions)

my_reduce = partial(pd.merge, on="sample", how='outer')
all_prob_table = reduce(my_reduce, all_probabilites_list)
all_prob_table = all_prob_table.sort_values(by="prob_of_BRCA2d", ascending=False).reset_index(drop=True)

#%% ==========================================================
# barplot with each sample aligned
# ============================================================

color_list = list(sns.color_palette().as_hex())
blue = color_list[0] #drp '#1f77b4'
orange = color_list[1] #atm '#ff7f0e'
green = color_list[2] #cdk12 '#2ca02c'
red = color_list[3] #brca2 '#d62728'
purple = color_list[4] #mmr '#9467bd'
bladder_brown = color_list[5] # '#8c564b'
bladder_pink = color_list[6] # '#e377c2'
bladder_grey = color_list[7] # '#7f7f7f'

fig, ax = plt.subplots(figsize=(2.8,2.6), nrows=3, gridspec_kw={'height_ratios':[1,1,1]})

gened="BRCA2d"
all_prob_table["color"] = '#1f77b4'
all_prob_table.loc[all_prob_table[f"prob_of_{gened}"] > 0.8, "color"] = '#d62728'
ax[0].bar(x=all_prob_table.index, height=all_prob_table[f"prob_of_{gened}"], width=0.8, edgecolor=None, linewidth=0, color=all_prob_table["color"], zorder=10)
barplot_aesthetics(all_prob_table, ax[0], gened)
barplot_legend(gened, "BRCA2p", '#d62728', ax[0], fs=6)

gened="CDK12d"
all_prob_table["color"] = '#1f77b4'
all_prob_table.loc[all_prob_table[f"prob_of_{gened}"] > 0.5, "color"] = '#2ca02c'
ax[1].bar(x=all_prob_table.index, height=all_prob_table[f"prob_of_{gened}"], width=0.8, edgecolor=None, linewidth=0, color=all_prob_table["color"], zorder=10)
barplot_aesthetics(all_prob_table, ax[1], gened)
barplot_legend(gened, "CDK12p", '#2ca02c', ax[1], fs=6)

gened="MMRd"
all_prob_table["color"] = '#1f77b4'
all_prob_table.loc[all_prob_table[f"prob_of_{gened}"] > 0.5, "color"] = '#9467bd'
ax[2].bar(x=all_prob_table.index, height=all_prob_table[f"prob_of_{gened}"], width=0.8, edgecolor=None, linewidth=0, color=all_prob_table["color"], zorder=10)
barplot_aesthetics(all_prob_table, ax[2], gened)
barplot_legend("MMRd   ", "MMRp", '#9467bd', ax[2], fs=6)

fig.subplots_adjust(hspace=0.05, wspace=0.0, left=0.145, right=0.995, top=0.99, bottom=0.01)
plt.savefig(os.path.join(figdir, f"bladder_probabilities.png"), dpi=500, transparent=False, facecolor="w")
plt.savefig(os.path.join(figdir, f"bladder_probabilities.pdf"))
# plt.close()

#%% ==========================================================
# umap of all samples BC and PC
# ============================================================

df_allctdna = pd.merge(sample_labels, sigs, how='left', on='sample')

all_categories = snv_categories[1:] + indel_categories[1:] + cnv_categories[1:]
features = df_allctdna[all_categories].columns
x = df_allctdna.loc[:, features].values
y = df_allctdna.loc[:,['sample']].values
rs=19
neighbor=13
md=0.35
standard_embedding = umap.UMAP(random_state=rs, n_neighbors=neighbor, metric="euclidean", n_epochs=500, min_dist=md, n_components=2).fit_transform(x)
mu = pd.DataFrame(data = standard_embedding , columns = ['umap X', 'umap Y'])
mu["sample"] = y
mu = pd.merge(mu, df_allctdna, how='left', on="sample").drop_duplicates(subset=['sample']).reset_index(drop=True)
mu["cellularity_sequenza"] = mu["cellularity_sequenza"]*200

#seperate out each for seperate graphing
bcscat = mu.query('(label == "BRCA2d") and (cancer == "PC")')
cdk12scat = mu.query('(label == "CDK12d") and (cancer == "PC")')
mmrdscat = mu.query('(label == "MMRd") and (cancer == "PC")')
ddrpscat = mu.query('(label == "DRwt") and (cancer == "PC")')

# bladder_brca_scat = mu.query('(label == "BRCA2") and (cancer == "BC")')
not_predicted_brca2d = all_prob_table[all_prob_table[f"prob_of_BRCA2d"] < 0.8]
predicted_brca2d = all_prob_table[all_prob_table[f"prob_of_BRCA2d"] > 0.8]
bladder_suspected = pd.merge(predicted_brca2d, mu, how="left", on="sample")
bladder_notsuspected = pd.merge(not_predicted_brca2d, mu, how="left", on="sample")

color_list = list(sns.color_palette().as_hex())
blue = color_list[0] #drp '#1f77b4'
orange = color_list[1] #atm '#ff7f0e'
green = color_list[2] #cdk12 '#2ca02c'
red = color_list[3] #brca2 '#d62728'
purple = color_list[4] #mmr '#9467bd'
bladder_brown = color_list[5] # '#8c564b' # '#a65628
bladder_pink = color_list[6] # '#e377c2'
bladder_grey = color_list[7] # '#7f7f7f'

fs=6
fig, ax = plt.subplots(figsize=(3.9,2.8))
ax.scatter(x=ddrpscat["umap X"], y=ddrpscat["umap Y"], color=blue, s=ddrpscat["cellularity_sequenza"], alpha=0.6, label="DRp", zorder=100, linewidth=0)
ax.scatter(x=bcscat["umap X"], y=bcscat["umap Y"], color=red, s=bcscat["cellularity_sequenza"], alpha=0.6, label="BRCA2d", zorder=200, linewidth=0)
ax.scatter(x=cdk12scat["umap X"], y=cdk12scat["umap Y"], color=green, s=cdk12scat["cellularity_sequenza"], alpha=0.6, label="CDK12d", zorder=300, linewidth=0)
ax.scatter(x=mmrdscat["umap X"], y=mmrdscat["umap Y"], color=purple, s=mmrdscat["cellularity_sequenza"], alpha=0.6, label="MMRd", zorder=400, linewidth=0)

ax.scatter(x=bladder_notsuspected["umap X"], y=bladder_notsuspected["umap Y"], color='#7f7f7f', s=bladder_notsuspected["cellularity_sequenza"], alpha=0.8, label="bladdermaybe", zorder=500, linewidth=0, marker="*")
ax.scatter(x=bladder_suspected["umap X"], y=bladder_suspected["umap Y"], color='k', s=bladder_suspected["cellularity_sequenza"], alpha=0.8, label="bladdernotmaybe", zorder=600, linewidth=0, marker="*")

ax.tick_params(axis='x', which='both', length=3, pad=2, labelsize=fs)
ax.tick_params(axis='y', which='both', length=3, pad=2, labelsize=fs)
ax.grid(b=False, which='both', axis='both', color='0.6', linewidth=0.7, linestyle='dotted', zorder=-100)

ax.set_ylabel("Umap Y", fontsize=fs, labelpad=5, verticalalignment='center')
# ax.yaxis.set_label_coords(-0.08, 0.5)
ax.set_xlabel("Umap X", fontsize=fs, labelpad=5, verticalalignment='center')
sns.despine(ax=ax, top=True, right=True)
fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.1, hspace=0.05, wspace=0)
plt.savefig(os.path.join(figdir, f"bladder_prostate_umap.png"), dpi=500, transparent=False, facecolor="w")
plt.savefig(os.path.join(figdir, f"bladder_prostate_umap.pdf"))
# plt.close()

fig, ax = plt.subplots(figsize=(1.5,1), nrows=2, gridspec_kw={'height_ratios':[2,1]})
ax[0].axis('off')
ax[0].grid(b=False, which='both')
ax[0].scatter([], [], c=blue, alpha=0.7, s=80, label="PC DRwt", marker='o', linewidth=0)
ax[0].scatter([], [], c=red, alpha=0.7, s=80, label="PC BRCA2d", marker='o', linewidth=0)
ax[0].scatter([], [], c=green, alpha=0.7, s=80, label="PC CDK12d", marker='o', linewidth=0)
ax[0].scatter([], [], c=purple, alpha=0.7, s=80, label="PC MMRd", marker='o', linewidth=0)
ax[0].legend(loc='lower left', ncol=1, borderaxespad=0., fontsize=6, labelspacing=0.7, handletextpad=0.1, borderpad=0.3, handlelength=2, framealpha=0, markerscale=1, columnspacing=1.5)
ax[1].axis('off')
ax[1].grid(b=False, which='both')
ax[1].scatter([], [], c="#7f7f7f", alpha=0.7, s=105, label="BC Predicted DRwt", marker='*', linewidth=0)
ax[1].scatter([], [], c='k', alpha=0.7, s=105, label="BC Predicted BRCA2d", marker='*', linewidth=0)
ax[1].legend(loc='upper left', ncol=1, borderaxespad=0., fontsize=6, labelspacing=0.7, handletextpad=0.1, borderpad=0.3, handlelength=2, framealpha=0, markerscale=1, columnspacing=1.2)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.1, wspace=0)
plt.savefig(os.path.join(figdir, f"bladder_prostate_umap_legend.png"), dpi=500, transparent=False, facecolor="w")
plt.savefig(os.path.join(figdir, f"bladder_prostate_umap_legend.pdf"))
plt.close()

#%%

