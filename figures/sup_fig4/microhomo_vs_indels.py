# -*- coding: utf-8 -*-
# @author: Elie
#%% ==========================================================
# Import libraries set library params
# ============================================================

# Libraries
import pandas as pd
import numpy as np
import os
#plotting
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import Locator
import matplotlib.lines as mlines
# stats
from scipy import stats
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
# make concat sig dataframe 
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
#%% ==========================================================
# figure functions and calculations
# ============================================================

def make_seabornstyle_table(df, feature_list):
	table_list = []
	for i, j in enumerate(feature_list):
		df_feature = df[["sample", "label", "primary_label_actual", feature_list[i]]]
		df_feature["feature"] = feature_list[i]
		df_feature = df_feature.rename(columns={feature_list[i]:"feature_value"})
		table_list.append(df_feature)
	graphtable = pd.concat(table_list)
	return graphtable

#same color list as always
color_list = list(sns.color_palette().as_hex())
blue = color_list[0] #DRwt
orange = color_list[1] #atm
green = color_list[2] #cdk12
red = color_list[3] #brca2
purple = color_list[4] #mmr

def common_settings(fig, ax, fs):
	# fig.set_size_inches(6, 2)
	# ax.set_yscale('symlog', linthresh=10, base=10)
	# yaxis = plt.gca().yaxis
	# yaxis.set_minor_locator(MinorSymLogLocator(10e1))
	sns.set_theme(style="whitegrid")
	ax.set_xticklabels(ax.get_xticklabels(), rotation=20, fontsize=fs, ha='right', va='top', rotation_mode="anchor")
	ax.tick_params(axis='both', which="major", length=2, labelsize=fs, pad=0.5, reset=False)
	ax.grid(b=False, which='both', axis='y', color='0.5', linewidth=0.6,linestyle='dotted', zorder=-100)
	# ax.set_ylim([-2, 35])
	ax.set_xlabel("")
	ax.set_ylabel("Feature value", labelpad=0, fontsize=fs)
	ax.set_ylim(bottom=0)
	ax.set_yticks(np.arange(0,19,2))
	sns.despine(top=True, right=True)
	fig.subplots_adjust(left=0.07, top=0.96, bottom=0.16, right=0.99)
	return fig, ax
	
def make_feature_pvalues(graphtable, feature_list, gene_def, gene_pos):
	pvalues = []
	gene_def = str(gene_def)
	gene_pos = str(gene_pos)
	for feature in feature_list:
		# print(f"Mann-Whitney of BRCA2 for {feature}")
		feature_table = graphtable.query('(feature == @feature)').reset_index(drop=True)
		deficient = feature_table.query('(label == @gene_def)')["feature_value"]
		proficient = feature_table.query('(label == @gene_pos)')["feature_value"]
		# print(stats.mannwhitneyu(deficient, proficient))
		u, p = stats.mannwhitneyu(deficient, proficient) 
		pvalues.append(p)
	feature_pvalues = dict(zip(feature_list, pvalues))
	return feature_pvalues
	
def plot_legend(fs=10):
	title = "DR labels"
	handles = []
	handles.append(mlines.Line2D([], [], color=blue, markeredgecolor=blue, marker='o', lw=0, markersize=5, label='DRwt'))
	handles.append(mlines.Line2D([], [], color=orange, markeredgecolor=orange, marker='o', lw=0, markersize=5, label='ATMd'))
	handles.append(mlines.Line2D([], [], color=green, markeredgecolor=green, marker='o', lw=0, markersize=5, label='CDK12d'))
	handles.append(mlines.Line2D([], [], color=red, markeredgecolor=red, marker='o', lw=0, markersize=5, label='BRCA2d'))
	handles.append(mlines.Line2D([], [], color=purple, markeredgecolor=purple, marker='o', lw=0, markersize=5, label='MMRd'))
	# plt.legend(handles=handles,loc=2, edgecolor='0.5', fancybox=True, frameon=False, facecolor='white', ncol=1, fontsize=10, labelspacing=0.1, handletextpad=-0.2, columnspacing=0.5, bbox_to_anchor=(0.94,0.95), title=legendtitle,title_fontsize=fs,)
	l = ax.legend(handles=handles, bbox_to_anchor=(0.85,1.02), loc=2, borderaxespad=0.,fontsize=fs, labelspacing=0.4, handletextpad=0, borderpad=0.3,handlelength=2, framealpha=0.6, title=legendtitle,title_fontsize=fs, ncol=1)
	l.get_title().set_multialignment('center')

#%% ==========================================================
# get paths, load data and make df with each file merged
# ============================================================

#files from paths relative to this script
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
figdir = os.path.join(rootdir, "figures", "sup_fig4")
datadir = os.path.join(rootdir, "data")
cohort_data = os.path.join(datadir, "cohort.tsv")
snv_features = os.path.join(datadir, "tns_features.tsv")
ndl_features = os.path.join(datadir, "ndl_features.tsv")
cnv_features = os.path.join(datadir, "cnv_features.tsv")

sigs = load_data(snv_features, ndl_features, cnv_features)
sample_labels = pd.read_csv(cohort_data, sep='\t', low_memory=False)
df = pd.merge(sample_labels, sigs, how='left', on='sample').query('(cancer == "PC")').reset_index(drop=True)


# =============================================================================
#%% bits of calcs
# =============================================================================

#order to display the fraction differences make them into percentages
cols = df.columns[df.columns.str.contains('CopyFraction')]
df[cols] = df[cols] * 100

brca2_shap_features = ["5:Del:R:0", "5:Del:M:1", "5:Del:M:2", "5:Del:M:3", "5:Del:M:4", "5:Del:M:5"]
brca2_table = df.copy(deep=True)
brca2_table["primary_label_actual"] = brca2_table["label"]
brca2_table.loc[brca2_table["label"] != "BRCA2d", "label"] = "BRCA2p"
brca_graphtable = make_seabornstyle_table(brca2_table, brca2_shap_features)

#%% ==========================================================
# BRCA2d plot
# =============================================================================

grey="#CACACA"
face_pal = {'BRCA2d': grey, 'BRCA2p': grey}
hue_order = ['BRCA2d', 'BRCA2p']
stripplot_kwargs = {'linewidth': 0, 'size': 4, 'alpha': 0.6, 'hue_order': hue_order}
legendtitle="DR labels"
fig, ax = plt.subplots(figsize=(7,2.5))
fs=10
sns.set_theme(style="whitegrid")
# sns.boxplot(x='feature', y='feature_value', hue='label', data=brca_graphtable, ax=ax, fliersize=0, **boxplot_kwargs)
sns.violinplot(x='feature', y='feature_value', hue='label', data=brca_graphtable, ax=ax, color="lightgrey", scale="width", hue_order=hue_order, width=0.8, cut=0, bw=.3, linewidth=0, inner=None, split=False, palette=face_pal)
ax.set_alpha(0.55)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "BRCA2d")'), ax=ax, jitter=0.15, dodge=True, palette={'BRCA2d': red, 'BRCA2p': "white"}, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "DRwt")'), ax=ax, jitter=0.15, dodge=True, color=blue, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "ATMd")'), ax=ax, jitter=0.15, dodge=True, color=orange, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "CDK12d")'), ax=ax, jitter=0.15, dodge=True, color=green, **stripplot_kwargs)
sns.stripplot(x='feature', y='feature_value', hue='label', data=brca_graphtable.query('(primary_label_actual == "MMRd")'), ax=ax, jitter=0.15, dodge=True, color=purple, **stripplot_kwargs)

brca_feature_pvalues = make_feature_pvalues(brca_graphtable, brca2_shap_features, "BRCA2d", "BRCA2p")

for i in ax.get_xticks():
	xstart = i-0.2
	xend = i+0.2
	feat=brca2_shap_features[i]
	height=brca_graphtable.query('(feature == @feat)')["feature_value"].max()
	if height > 15:
		y=17
		h=0.5
	if height < 10:
		y=10
		h=0.5
	if height < 5:
		y=5
		h=0.5
	# y=17
	# h=0.5
	col='k'
	ax.plot([xstart, xstart, xend, xend], [y, y+h, y+h, y], lw=1.1, c=col)
	if brca_feature_pvalues[feat] > 0.05:
		annot="n.s."
	if brca_feature_pvalues[feat] <= 0.05:
		annot="*"
	if brca_feature_pvalues[feat] <= 0.01:
		annot="**"
	if brca_feature_pvalues[feat] <= 0.001:
		annot="***"
	ax.text((xstart+xend)*.5, y+h, annot, ha='center', va='bottom', color=col, fontsize=fs, linespacing=0)

plot_legend()
common_settings(fig, ax, 10)
plt.savefig(os.path.join(figdir, "violin_long_del_vs_microhomo.pdf"))
plt.savefig(os.path.join(figdir, "violin_long_del_vs_microhomo.png"), dpi=500, transparent=False, facecolor="w")

#save stats
with open(os.path.join(figdir, "feature_comparison_stats_mhVSlongindel.txt"), "w") as f:
	print("#==============================================", file=f)
	print("BRCA2 statistics:", file=f)
	for feature in brca2_shap_features:
		print(f"Mann-Whitney of BRCA2 for {feature}", file=f)
		feature_table = brca_graphtable.query('(feature == @feature)').reset_index(drop=True)
		deficient = feature_table.query('(label == "BRCA2d")')["feature_value"]
		proficient = feature_table.query('(label == "BRCA2p")')["feature_value"]
		print(f"{stats.mannwhitneyu(deficient, proficient)}", file=f)
	print("#==============================================", file=f)

#%% ==========================================================
# 5:Del:M:1 on yaxis vs 5:Del:R:0 on xaxis
# ============================================================
df_5dm1_vs_5d = df[["sample", "label", "5:Del:R:0", "5:Del:M:1"]]
sns.set_theme(style="whitegrid")
fs=10
legendtitle="DR labels"
# Load the example tips dataset
bcscat = df_5dm1_vs_5d.query('(label == "BRCA2d")')
cdk12scat = df_5dm1_vs_5d.query('(label == "CDK12d")')
mmrdscat = df_5dm1_vs_5d.query('(label == "MMRd")')
dDRwtscat = df_5dm1_vs_5d.query('(label == "DRwt")')
atmscat = df_5dm1_vs_5d.query('(label == "ATMd")')

fig, ax = plt.subplots(figsize=(7,3))
ax.scatter(y=bcscat["5:Del:M:1"], x=bcscat["5:Del:R:0"], color=red, s=120, alpha=1, label="BRCA2d", zorder=100,linewidth=0)
ax.scatter(y=mmrdscat["5:Del:M:1"], x=mmrdscat["5:Del:R:0"], color=purple,s=80, alpha=1, label="MMRd", zorder=100,linewidth=0)
ax.scatter(y=cdk12scat["5:Del:M:1"], x=cdk12scat["5:Del:R:0"], color=green,s=50, alpha=1, label="CDK12d",zorder=100, linewidth=0)
ax.scatter(y=dDRwtscat["5:Del:M:1"], x=dDRwtscat["5:Del:R:0"], color=blue,s=20, alpha=1, label="DRwt", zorder=100,linewidth=0)

ax.grid(b=False, which='both', axis='both', color='0.6', linewidth=0.7,linestyle='dotted', zorder=-100)
ax.set_xlabel("5:Del:R:0 (#)", fontsize=fs, labelpad=9,verticalalignment='center')
ax.set_ylabel("5:Del:M:1 (#)", fontsize=fs, labelpad=9,verticalalignment='center')
ax.set_xticks(np.arange(0,df_5dm1_vs_5d["5:Del:R:0"].max()+1,2))
ax.set_yticks(np.arange(0,df_5dm1_vs_5d["5:Del:M:1"].max()+1,2))
ax.tick_params(axis='both', which="major", length=3, labelsize=fs, pad=0.5, reset=False)

l = ax.legend(bbox_to_anchor=(0.02, 1.02), loc=2, borderaxespad=0.,fontsize=fs, labelspacing=0.4, handletextpad=0, borderpad=0.3,handlelength=2, framealpha=0.6, title=legendtitle,title_fontsize=fs, ncol=1)
l.get_title().set_multialignment('center')

sns.despine(ax=ax, top=True, right=True)
fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.15,hspace=0.05, wspace=0)
# fig.subplots_adjust(left=0.07, top=0.96, bottom=0.16, right=0.99)
plt.savefig(os.path.join(figdir, "scatter_long_del_vs_microhomo_del.pdf"))
plt.savefig(os.path.join(figdir, "scatter_long_del_vs_microhomo_del.png"), dpi=500, transparent=False, facecolor="w")
# %%

# color_list = list(sns.color_palette().as_hex())
# print(f"{color_list[0]}") #DRwt
# print(f"{color_list[1]}") #atm
# print(f"{color_list[2]}") #cdk12
# print(f"{color_list[3]}") #brca2
# print(f"{color_list[4]}") #mmr
# print(f"{color_list[5]}") #bladder

# #4c72b0
# #dd8452
# #55a868
# #c44e52
# #8172b3
# #937860
